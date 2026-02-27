"""vLLM server-side metrics scraper (Prometheus format).

Scrapes /metrics endpoint before and after a benchmark run to compute
server-side TTFT, TPOT, E2E latency, and throughput — free of network latency.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx


@dataclass
class HistogramSnapshot:
    """Parsed Prometheus histogram at a point in time."""
    buckets: Dict[float, float] = field(default_factory=dict)  # le → count
    count: float = 0.0
    total: float = 0.0  # _sum


@dataclass
class ServerMetricsSnapshot:
    """All vLLM metrics at a single point in time."""
    ts: float = 0.0
    ttft: Optional[HistogramSnapshot] = None
    tpot: Optional[HistogramSnapshot] = None
    e2e: Optional[HistogramSnapshot] = None
    generation_tokens: float = 0.0
    prompt_tokens: float = 0.0
    requests_running: float = 0.0
    kv_cache_usage: float = 0.0
    raw_text: str = ""


@dataclass
class ServerMetricsDelta:
    """Delta between two snapshots: before/after benchmark run."""
    duration_s: float = 0.0

    # TTFT (server-side, no network)
    ttft_mean_ms: Optional[float] = None
    ttft_count: int = 0
    ttft_p50_ms: Optional[float] = None
    ttft_p95_ms: Optional[float] = None
    ttft_p99_ms: Optional[float] = None

    # TPOT (server-side inter-token latency)
    tpot_mean_ms: Optional[float] = None
    tpot_count: int = 0
    tpot_p50_ms: Optional[float] = None
    tpot_p95_ms: Optional[float] = None
    tpot_p99_ms: Optional[float] = None

    # E2E request latency
    e2e_mean_ms: Optional[float] = None
    e2e_count: int = 0

    # Throughput
    generation_tokens_delta: float = 0.0
    prompt_tokens_delta: float = 0.0
    gen_tps: float = 0.0  # generation tokens / second
    prompt_tps: float = 0.0

    # Peak utilization
    peak_kv_cache_usage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def _parse_prometheus(text: str) -> Dict[str, List[Tuple[Dict[str, str], float]]]:
    """Parse Prometheus text format into {metric_name: [(labels, value), ...]}."""
    result: Dict[str, List[Tuple[Dict[str, str], float]]] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        # metric_name{label="val",...} value
        m = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]*)\}\s+(.+)$', line)
        if m:
            name, labels_str, val_str = m.group(1), m.group(2), m.group(3)
            labels = {}
            for pair in re.findall(r'(\w+)="([^"]*)"', labels_str):
                labels[pair[0]] = pair[1]
            try:
                val = float(val_str)
            except ValueError:
                continue
            result.setdefault(name, []).append((labels, val))
            continue
        # metric_name value (no labels)
        m2 = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+(.+)$', line)
        if m2:
            name, val_str = m2.group(1), m2.group(2)
            try:
                val = float(val_str)
            except ValueError:
                continue
            result.setdefault(name, []).append(({}, val))
    return result


def _extract_histogram(parsed: Dict, prefix: str) -> Optional[HistogramSnapshot]:
    """Extract a histogram from parsed Prometheus data."""
    h = HistogramSnapshot()
    bucket_key = f"{prefix}_bucket"
    count_key = f"{prefix}_count"
    sum_key = f"{prefix}_sum"

    if bucket_key in parsed:
        for labels, val in parsed[bucket_key]:
            le = labels.get("le", "")
            try:
                le_f = float(le) if le != "+Inf" else float("inf")
            except ValueError:
                continue
            h.buckets[le_f] = val

    if count_key in parsed:
        for _, val in parsed[count_key]:
            h.count = val
    if sum_key in parsed:
        for _, val in parsed[sum_key]:
            h.total = val

    if h.count > 0 or h.buckets:
        return h
    return None


def _extract_gauge(parsed: Dict, name: str) -> float:
    """Extract a simple gauge value."""
    if name in parsed:
        vals = parsed[name]
        if vals:
            return vals[-1][1]
    return 0.0


def _extract_counter(parsed: Dict, name: str) -> float:
    """Extract a counter total."""
    # Try _total suffix first, then bare name
    for key in [f"{name}_total", name]:
        if key in parsed:
            total = 0.0
            for _, val in parsed[key]:
                total += val
            return total
    return 0.0


def _histogram_percentile(before: Optional[HistogramSnapshot],
                          after: Optional[HistogramSnapshot],
                          p: float) -> Optional[float]:
    """Estimate percentile from delta of two histogram bucket snapshots.

    Uses linear interpolation within histogram buckets.
    """
    if not before or not after:
        return None

    # Compute delta buckets
    all_les = sorted(set(before.buckets.keys()) | set(after.buckets.keys()))
    if not all_les:
        return None

    delta_count = after.count - before.count
    if delta_count <= 0:
        return None

    target = p / 100.0 * delta_count
    prev_le = 0.0
    prev_count = 0.0

    for le in all_les:
        if le == float("inf"):
            continue
        b_val = before.buckets.get(le, 0.0)
        a_val = after.buckets.get(le, 0.0)
        delta_bucket = a_val - b_val
        if delta_bucket >= target:
            # Linear interpolation
            bucket_width = le - prev_le
            remaining = target - prev_count
            fraction = remaining / max(delta_bucket - prev_count, 1e-9)
            return prev_le + fraction * bucket_width
        prev_le = le
        prev_count = delta_bucket

    return prev_le  # Above all buckets


def parse_snapshot(text: str) -> ServerMetricsSnapshot:
    """Parse raw Prometheus metrics text into a snapshot."""
    parsed = _parse_prometheus(text)
    snap = ServerMetricsSnapshot(
        ts=time.time(),
        raw_text=text,
    )

    # Histograms — vLLM and SGLang use different metric name formats
    for prefix in ["vllm:time_to_first_token_seconds",
                   "vllm_time_to_first_token_seconds",
                   "sglang:time_to_first_token_seconds"]:
        h = _extract_histogram(parsed, prefix)
        if h:
            snap.ttft = h
            break

    for prefix in ["vllm:inter_token_latency_seconds",
                   "vllm_inter_token_latency_seconds",
                   "sglang:inter_token_latency_seconds"]:
        h = _extract_histogram(parsed, prefix)
        if h:
            snap.tpot = h
            break

    for prefix in ["vllm:e2e_request_latency_seconds",
                   "vllm_e2e_request_latency_seconds",
                   "sglang:e2e_request_latency_seconds"]:
        h = _extract_histogram(parsed, prefix)
        if h:
            snap.e2e = h
            break

    # Counters
    for name in ["vllm:generation_tokens", "vllm_generation_tokens",
                 "sglang:generation_tokens"]:
        v = _extract_counter(parsed, name)
        if v > 0:
            snap.generation_tokens = v
            break

    for name in ["vllm:prompt_tokens", "vllm_prompt_tokens",
                 "sglang:prompt_tokens"]:
        v = _extract_counter(parsed, name)
        if v > 0:
            snap.prompt_tokens = v
            break

    # Gauges
    for name in ["vllm:num_requests_running", "vllm_num_requests_running",
                 "sglang:num_running_reqs"]:
        v = _extract_gauge(parsed, name)
        if v > 0:
            snap.requests_running = v
            break

    for name in ["vllm:gpu_cache_usage_perc", "vllm_gpu_cache_usage_perc",
                 "vllm:kv_cache_usage_perc", "sglang:token_usage"]:
        v = _extract_gauge(parsed, name)
        if v > 0:
            snap.kv_cache_usage = v
            break

    return snap


def compute_delta(before: ServerMetricsSnapshot,
                  after: ServerMetricsSnapshot) -> ServerMetricsDelta:
    """Compute server-side metrics delta between before/after snapshots."""
    d = ServerMetricsDelta()
    d.duration_s = after.ts - before.ts

    # TTFT
    if before.ttft and after.ttft:
        count_delta = after.ttft.count - before.ttft.count
        sum_delta = after.ttft.total - before.ttft.total
        d.ttft_count = int(count_delta)
        if count_delta > 0:
            d.ttft_mean_ms = (sum_delta / count_delta) * 1000.0
            d.ttft_p50_ms = (_histogram_percentile(before.ttft, after.ttft, 50) or 0) * 1000
            d.ttft_p95_ms = (_histogram_percentile(before.ttft, after.ttft, 95) or 0) * 1000
            d.ttft_p99_ms = (_histogram_percentile(before.ttft, after.ttft, 99) or 0) * 1000

    # TPOT
    if before.tpot and after.tpot:
        count_delta = after.tpot.count - before.tpot.count
        sum_delta = after.tpot.total - before.tpot.total
        d.tpot_count = int(count_delta)
        if count_delta > 0:
            d.tpot_mean_ms = (sum_delta / count_delta) * 1000.0
            d.tpot_p50_ms = (_histogram_percentile(before.tpot, after.tpot, 50) or 0) * 1000
            d.tpot_p95_ms = (_histogram_percentile(before.tpot, after.tpot, 95) or 0) * 1000
            d.tpot_p99_ms = (_histogram_percentile(before.tpot, after.tpot, 99) or 0) * 1000

    # E2E
    if before.e2e and after.e2e:
        count_delta = after.e2e.count - before.e2e.count
        sum_delta = after.e2e.total - before.e2e.total
        d.e2e_count = int(count_delta)
        if count_delta > 0:
            d.e2e_mean_ms = (sum_delta / count_delta) * 1000.0

    # Throughput
    d.generation_tokens_delta = after.generation_tokens - before.generation_tokens
    d.prompt_tokens_delta = after.prompt_tokens - before.prompt_tokens
    if d.duration_s > 0:
        d.gen_tps = d.generation_tokens_delta / d.duration_s
        d.prompt_tps = d.prompt_tokens_delta / d.duration_s

    d.peak_kv_cache_usage = max(before.kv_cache_usage, after.kv_cache_usage)

    return d


async def scrape_metrics(base_url: str, client: httpx.AsyncClient,
                         timeout: float = 10.0) -> Optional[ServerMetricsSnapshot]:
    """Scrape vLLM /metrics endpoint. Returns None if unavailable."""
    metrics_url = base_url.rstrip("/").replace("/v1", "") + "/metrics"
    try:
        resp = await client.get(metrics_url, timeout=timeout)
        resp.raise_for_status()
        text = resp.text
        snap = parse_snapshot(text)
        return snap
    except Exception as e:
        print(f"[metrics] Warning: could not scrape {metrics_url}: {e}")
        return None


def save_server_metrics(delta: ServerMetricsDelta, output_dir: str,
                        suffix: str = "") -> str:
    """Save server metrics delta as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"server_metrics{suffix}.json")
    with open(path, "w") as f:
        json.dump(delta.to_dict(), f, indent=2)
    return path


def print_server_metrics(delta: ServerMetricsDelta) -> None:
    """Print server-side metrics summary."""
    print("\n-- Server-side metrics (vLLM /metrics, no network latency) --")
    if delta.ttft_count > 0:
        print(f"Server TTFT: mean={delta.ttft_mean_ms:.2f}ms  "
              f"p50={delta.ttft_p50_ms:.2f}ms  "
              f"p95={delta.ttft_p95_ms:.2f}ms  "
              f"p99={delta.ttft_p99_ms:.2f}ms  "
              f"(n={delta.ttft_count})")
    if delta.tpot_count > 0:
        print(f"Server TPOT: mean={delta.tpot_mean_ms:.2f}ms  "
              f"p50={delta.tpot_p50_ms:.2f}ms  "
              f"p95={delta.tpot_p95_ms:.2f}ms  "
              f"p99={delta.tpot_p99_ms:.2f}ms  "
              f"(n={delta.tpot_count})")
    if delta.e2e_count > 0:
        print(f"Server E2E:  mean={delta.e2e_mean_ms:.2f}ms  (n={delta.e2e_count})")
    print(f"Server TPS:  gen={delta.gen_tps:.1f} tok/s  "
          f"prompt={delta.prompt_tps:.1f} tok/s  "
          f"(total gen={delta.generation_tokens_delta:.0f} tok in {delta.duration_s:.1f}s)")
    if delta.peak_kv_cache_usage > 0:
        print(f"KV cache:    peak={delta.peak_kv_cache_usage:.1%}")
