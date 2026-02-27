#!/usr/bin/env python3
"""
Prefix-heavy multi-turn workload benchmark for KV cache experiments.

Simulates coding-agent-style conversations where:
- All sessions share a large system prompt (prefix)
- Each turn appends tool results + assistant response (context grows)
- Multiple concurrent sessions compete for KV cache space

Measures: TTFT, TPOT, throughput, and scrapes vLLM Prometheus metrics
for prefix_cache_hits, kv_cache_usage, preemptions.

Usage:
  python benchmark_prefix_workload.py --concurrency 8 --num_sessions 32 --num_turns 8
  python benchmark_prefix_workload.py --concurrency 1 --num_turns 5 --output_dir results/
"""

import argparse
import asyncio
import json
import os
import random
import string
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

try:
    import httpx
except ImportError:
    print("ERROR: httpx required. Install: pip install httpx")
    raise


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_random_text(num_chars: int) -> str:
    """Generate pseudo-random text (deterministic per length for reproducibility)."""
    rng = random.Random(num_chars)
    words = []
    while len(" ".join(words)) < num_chars:
        word_len = rng.randint(3, 12)
        words.append("".join(rng.choices(string.ascii_lowercase, k=word_len)))
    return " ".join(words)[:num_chars]


def build_system_prompt(target_tokens: int) -> str:
    """Build a system prompt of approximately target_tokens tokens.

    Random lowercase words tokenize at ~2 chars/token (verified with Qwen3 tokenizer).
    """
    target_chars = target_tokens * 2
    base = (
        "You are an expert software engineering assistant. "
        "You help with code review, debugging, refactoring, and implementation tasks. "
        "Always provide clear, concise code with explanations.\n\n"
        "## Project Context\n"
    )
    filler = generate_random_text(max(0, target_chars - len(base)))
    return base + filler


def build_turn_message(turn_idx: int, context_growth_tokens: int) -> str:
    """Build a user message for turn N, simulating growing context.

    Each turn adds ~context_growth_tokens of new context (tool output, file content).
    """
    target_chars = context_growth_tokens * 2
    base = f"## Turn {turn_idx + 1}\n"
    base += f"Here is the latest file content and tool output:\n\n```\n"
    filler = generate_random_text(max(0, target_chars - len(base) - 10))
    base += filler + "\n```\n\nPlease review and fix any issues."
    return base


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class RequestMetrics:
    session_id: int
    turn_idx: int
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    start_time: float = 0.0
    error: Optional[str] = None


@dataclass
class ServerMetricsSnapshot:
    timestamp: float = 0.0
    prefix_cache_hit_rate: float = -1.0
    prefix_cache_queries: int = -1
    prefix_cache_hits: int = -1
    kv_cache_usage_pct: float = -1.0
    num_preemptions: int = -1
    num_requests_running: int = -1
    num_requests_waiting: int = -1
    raw: str = ""


def parse_prometheus_metrics(text: str) -> ServerMetricsSnapshot:
    """Parse vLLM Prometheus /metrics text into a snapshot."""
    snap = ServerMetricsSnapshot(timestamp=time.time())

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        # Parse "metric_name{labels} value" or "metric_name value"
        try:
            parts = line.split()
            if len(parts) < 2:
                continue
            name_labels = parts[0]
            value = float(parts[-1])
            name = name_labels.split("{")[0]

            if "prefix_cache_hit_rate" in name:
                snap.prefix_cache_hit_rate = value
            elif "prefix_cache_queries_total" in name:
                snap.prefix_cache_queries = int(value)
            elif "prefix_cache_hits_total" in name:
                snap.prefix_cache_hits = int(value)
            elif "gpu_cache_usage_perc" in name:
                snap.kv_cache_usage_pct = value
            elif "num_preemptions_total" in name:
                snap.num_preemptions = int(value)
            elif "num_requests_running" in name and "gauge" not in name:
                snap.num_requests_running = int(value)
            elif "num_requests_waiting" in name and "gauge" not in name:
                snap.num_requests_waiting = int(value)
        except (ValueError, IndexError):
            continue

    snap.raw = text
    return snap


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

async def send_chat_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: list,
    max_tokens: int = 64,
    temperature: float = 0.2,
) -> RequestMetrics:
    """Send a streaming chat completion request and measure TTFT/TPOT."""
    metrics = RequestMetrics(session_id=-1, turn_idx=-1)
    metrics.start_time = time.time()

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    first_token_time = None
    output_tokens = 0
    start_ns = time.monotonic_ns()

    try:
        async with client.stream(
            "POST", f"{base_url}/chat/completions", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Usage in final chunk
                if "usage" in chunk and chunk["usage"]:
                    metrics.input_tokens = chunk["usage"].get("prompt_tokens", 0)
                    metrics.output_tokens = chunk["usage"].get("completion_tokens", 0)

                # Detect content tokens
                choices = chunk.get("choices", [])
                if choices and choices[0].get("delta", {}).get("content"):
                    output_tokens += 1
                    if first_token_time is None:
                        first_token_time = time.monotonic_ns()

        end_ns = time.monotonic_ns()
        metrics.latency_ms = (end_ns - start_ns) / 1e6

        if first_token_time is not None:
            metrics.ttft_ms = (first_token_time - start_ns) / 1e6
            if output_tokens > 1:
                metrics.tpot_ms = (end_ns - first_token_time) / 1e6 / (output_tokens - 1)

        if metrics.output_tokens == 0:
            metrics.output_tokens = output_tokens

    except Exception as e:
        metrics.error = str(e)
        metrics.latency_ms = (time.monotonic_ns() - start_ns) / 1e6

    return metrics


async def run_session(
    session_id: int,
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    system_prompt: str,
    num_turns: int,
    context_growth_tokens: int,
    max_output_tokens: int,
    semaphore: asyncio.Semaphore,
) -> List[RequestMetrics]:
    """Run a single multi-turn session."""
    results = []
    messages = [{"role": "system", "content": system_prompt}]

    for turn in range(num_turns):
        # Build user message with growing context
        user_msg = build_turn_message(turn, context_growth_tokens)
        messages.append({"role": "user", "content": user_msg})

        async with semaphore:
            metrics = await send_chat_request(
                client, base_url, model, messages, max_tokens=max_output_tokens
            )

        metrics.session_id = session_id
        metrics.turn_idx = turn
        results.append(metrics)

        if metrics.error:
            print(f"  [session {session_id}] turn {turn}: ERROR {metrics.error}")
            break

        # Append a fake assistant response to grow context for next turn
        assistant_content = generate_random_text(max_output_tokens * 4)
        messages.append({"role": "assistant", "content": assistant_content})

    return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def run_benchmark(args) -> dict:
    """Run the full benchmark: N sessions x M turns at given concurrency."""
    system_prompt = build_system_prompt(args.system_prompt_tokens)
    all_results: List[RequestMetrics] = []

    print(f"\n{'='*60}")
    print(f"  Benchmark: {args.num_sessions} sessions x {args.num_turns} turns")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  System prompt: ~{args.system_prompt_tokens} tokens")
    print(f"  Context growth: ~{args.context_growth_tokens} tokens/turn")
    print(f"  Max output: {args.max_output_tokens} tokens")
    print(f"  Model: {args.model}")
    print(f"  Server: {args.base_url}")
    print(f"{'='*60}\n")

    semaphore = asyncio.Semaphore(args.concurrency)

    # Scrape server metrics before
    metrics_before = None
    try:
        async with httpx.AsyncClient(timeout=5) as mc:
            resp = await mc.get(f"{args.base_url.replace('/v1', '')}/metrics")
            metrics_before = parse_prometheus_metrics(resp.text)
    except Exception:
        pass

    t_start = time.time()

    async with httpx.AsyncClient(timeout=300) as client:
        tasks = []
        for sid in range(args.num_sessions):
            tasks.append(
                run_session(
                    session_id=sid,
                    client=client,
                    base_url=args.base_url,
                    model=args.model,
                    system_prompt=system_prompt,
                    num_turns=args.num_turns,
                    context_growth_tokens=args.context_growth_tokens,
                    max_output_tokens=args.max_output_tokens,
                    semaphore=semaphore,
                )
            )
        session_results = await asyncio.gather(*tasks, return_exceptions=True)

    t_end = time.time()

    for sr in session_results:
        if isinstance(sr, Exception):
            print(f"  [ERROR] Session exception: {sr}")
            continue
        all_results.extend(sr)

    # Scrape server metrics after
    metrics_after = None
    try:
        async with httpx.AsyncClient(timeout=5) as mc:
            resp = await mc.get(f"{args.base_url.replace('/v1', '')}/metrics")
            metrics_after = parse_prometheus_metrics(resp.text)
    except Exception:
        pass

    # Compute summary
    ok_results = [r for r in all_results if r.error is None]
    errors = [r for r in all_results if r.error is not None]

    summary = {
        "config": {
            "num_sessions": args.num_sessions,
            "num_turns": args.num_turns,
            "concurrency": args.concurrency,
            "system_prompt_tokens": args.system_prompt_tokens,
            "context_growth_tokens": args.context_growth_tokens,
            "max_output_tokens": args.max_output_tokens,
            "model": args.model,
            "base_url": args.base_url,
            "mode": args.mode,
        },
        "total_requests": len(all_results),
        "successful": len(ok_results),
        "errors": len(errors),
        "wall_time_s": t_end - t_start,
    }

    if ok_results:
        ttfts = [r.ttft_ms for r in ok_results if r.ttft_ms > 0]
        tpots = [r.tpot_ms for r in ok_results if r.tpot_ms > 0]
        lats = [r.latency_ms for r in ok_results]

        def percentiles(vals):
            if not vals:
                return {}
            s = sorted(vals)
            n = len(s)
            return {
                "mean": sum(s) / n,
                "p50": s[n // 2],
                "p90": s[int(n * 0.9)],
                "p99": s[int(n * 0.99)],
                "min": s[0],
                "max": s[-1],
            }

        summary["ttft_ms"] = percentiles(ttfts)
        summary["tpot_ms"] = percentiles(tpots)
        summary["latency_ms"] = percentiles(lats)

        total_output = sum(r.output_tokens for r in ok_results)
        summary["throughput_tps"] = total_output / (t_end - t_start) if (t_end > t_start) else 0

        # Per-turn breakdown (shows prefix cache effect as turns progress)
        per_turn = {}
        for turn in range(args.num_turns):
            turn_results = [r for r in ok_results if r.turn_idx == turn]
            if turn_results:
                turn_ttfts = [r.ttft_ms for r in turn_results if r.ttft_ms > 0]
                per_turn[f"turn_{turn}"] = {
                    "count": len(turn_results),
                    "ttft_mean_ms": sum(turn_ttfts) / len(turn_ttfts) if turn_ttfts else 0,
                    "ttft_p50_ms": sorted(turn_ttfts)[len(turn_ttfts) // 2] if turn_ttfts else 0,
                    "avg_input_tokens": sum(r.input_tokens for r in turn_results) / len(turn_results),
                }
        summary["per_turn"] = per_turn

    # Server metrics delta
    if metrics_before and metrics_after:
        summary["server_metrics"] = {
            "prefix_cache_hit_rate": metrics_after.prefix_cache_hit_rate,
            "kv_cache_usage_pct": metrics_after.kv_cache_usage_pct,
            "preemptions_delta": (
                metrics_after.num_preemptions - metrics_before.num_preemptions
                if metrics_before.num_preemptions >= 0 and metrics_after.num_preemptions >= 0
                else -1
            ),
        }

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Results: mode={args.mode}")
    print(f"{'='*60}")
    print(f"  Requests: {summary['successful']}/{summary['total_requests']} OK, {summary['errors']} errors")
    print(f"  Wall time: {summary['wall_time_s']:.1f}s")
    if "ttft_ms" in summary:
        print(f"  TTFT:  mean={summary['ttft_ms']['mean']:.1f}ms  p50={summary['ttft_ms']['p50']:.1f}ms  p99={summary['ttft_ms']['p99']:.1f}ms")
    if "tpot_ms" in summary:
        print(f"  TPOT:  mean={summary['tpot_ms']['mean']:.1f}ms  p50={summary['tpot_ms']['p50']:.1f}ms")
    if "throughput_tps" in summary:
        print(f"  TPS:   {summary['throughput_tps']:.1f}")
    if "per_turn" in summary:
        print(f"\n  Per-turn TTFT (prefix cache effect):")
        for turn_key, tv in sorted(summary["per_turn"].items()):
            print(f"    {turn_key}: TTFT_mean={tv['ttft_mean_ms']:.1f}ms  input_tokens={tv['avg_input_tokens']:.0f}")
    if "server_metrics" in summary:
        sm = summary["server_metrics"]
        print(f"\n  Server: prefix_hit_rate={sm['prefix_cache_hit_rate']:.3f}  kv_usage={sm['kv_cache_usage_pct']:.3f}  preemptions={sm['preemptions_delta']}")

    # Save detailed results
    output = {
        "summary": summary,
        "requests": [asdict(r) for r in all_results],
    }
    if metrics_after:
        output["server_metrics_raw"] = metrics_after.raw

    return output


def main():
    parser = argparse.ArgumentParser(description="Prefix-heavy KV cache benchmark")
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-Next-80B-A3B-Instruct")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--num_sessions", type=int, default=32)
    parser.add_argument("--num_turns", type=int, default=8)
    parser.add_argument("--system_prompt_tokens", type=int, default=2000)
    parser.add_argument("--context_growth_tokens", type=int, default=1500)
    parser.add_argument("--max_output_tokens", type=int, default=64)
    parser.add_argument("--mode", default="unknown", help="Label for this config (e.g., baseline, prefix, prefix_offload)")
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output = asyncio.run(run_benchmark(args))

    out_path = os.path.join(args.output_dir, f"results_{args.mode}_c{args.concurrency}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
