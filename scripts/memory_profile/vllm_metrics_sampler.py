#!/usr/bin/env python3
"""Poll vLLM /metrics (Prometheus) endpoint and write to CSV.

Captures gpu_cache_usage_perc, running/waiting request counts,
preemption counts, and throughput metrics at configurable intervals.
"""

import csv
import re
import threading
import time
from pathlib import Path

import requests


# Metrics we want to capture (vLLM Prometheus names)
METRICS_OF_INTEREST = [
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
]


def parse_prometheus_text(text: str, metrics: list[str] | None = None) -> dict:
    """Parse Prometheus text exposition format into {metric_name: value}."""
    result = {}
    targets = set(metrics) if metrics else None
    for line in text.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        # metric_name{labels} value [timestamp]
        # or: metric_name value [timestamp]
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{")[0]
        if targets and name not in targets:
            continue
        try:
            result[name] = float(parts[1])
        except (ValueError, IndexError):
            continue
    return result


class VllmMetricsSampler:
    """Background thread that polls vLLM /metrics and records samples."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        output_path: str | Path = "metrics_trace.csv",
        interval_s: float = 0.5,
        metrics: list[str] | None = None,
    ):
        # Strip /v1 suffix if present
        url = base_url.rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        self.url = f"{url}/metrics"
        self.output_path = Path(output_path)
        self.interval = interval_s
        self.metrics = metrics or METRICS_OF_INTEREST
        self._stop = threading.Event()
        self._thread = None
        self._samples: list[dict] = []
        self._t0 = 0.0

    @property
    def samples(self) -> list[dict]:
        return list(self._samples)

    def _poll_loop(self):
        while not self._stop.is_set():
            try:
                resp = requests.get(self.url, timeout=2)
                if resp.status_code == 200:
                    parsed = parse_prometheus_text(resp.text, self.metrics)
                    parsed["timestamp"] = time.time()
                    parsed["elapsed_s"] = parsed["timestamp"] - self._t0
                    self._samples.append(parsed)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._t0 = time.time()
        self._stop.clear()
        self._samples.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print(f"  [metrics] polling {self.url} every {self.interval}s")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._write_csv()
        print(f"  [metrics] {len(self._samples)} samples written to {self.output_path}")

    def _write_csv(self):
        if not self._samples:
            return
        fieldnames = ["timestamp", "elapsed_s"] + self.metrics
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(self._samples)

    def peak_cache_usage(self) -> float:
        """Return peak kv_cache_usage_perc observed."""
        vals = [
            s.get("vllm:kv_cache_usage_perc", 0.0) for s in self._samples
        ]
        return max(vals) if vals else 0.0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="vLLM metrics sampler")
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000",
        help="vLLM server base URL",
    )
    parser.add_argument(
        "-o", "--output", default="metrics_trace.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--interval", type=float, default=0.5,
        help="Polling interval in seconds",
    )
    parser.add_argument(
        "--duration", type=float, default=30,
        help="Total sampling duration in seconds",
    )
    args = parser.parse_args()

    sampler = VllmMetricsSampler(args.base_url, args.output, args.interval)
    sampler.start()
    print(f"Sampling for {args.duration}s ...")
    time.sleep(args.duration)
    sampler.stop()
    print(f"Peak cache usage: {sampler.peak_cache_usage():.1%}")
