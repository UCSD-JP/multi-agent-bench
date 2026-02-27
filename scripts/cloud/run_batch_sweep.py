#!/usr/bin/env python3
"""
Batch sweep: measure TPOT/TTFT/TPS across (input_len × batch_size) grid.

Sends `batch_size` concurrent requests to vLLM, each with `input_len` prompt
tokens and `output_len` max_tokens.  Collects streaming token timestamps to
compute per-request TPOT, then aggregates.

Output format matches existing H200 TP8 batch_sweep_v4 JSON schema.

Usage:
    python run_batch_sweep.py --preset tp2-fp16
    python run_batch_sweep.py --preset tp4-fp16 --output-dir /data/results/batch
    python run_batch_sweep.py --input-lens 128 512 --batch-sizes 1 8 16
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path

import aiohttp


async def make_request(session, base_url, api_key, model, prompt_text,
                       output_len, request_id):
    """Send one streaming chat completion, return token-level timing."""
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": output_len,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    token_times = []
    first_token_time = None
    start_time = time.perf_counter()
    gen_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0

    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                return {"error": f"HTTP {resp.status}: {text}", "id": request_id}

            # Read line-by-line to get individual SSE events
            buffer = ""
            async for raw_chunk in resp.content.iter_any():
                buffer += raw_chunk.decode("utf-8")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices", [])
                    content = ""
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                    if content:
                        now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = now
                        token_times.append(now)
                        gen_tokens += 1  # each SSE content chunk = 1 token

                    # Extract usage if available
                    usage = chunk.get("usage")
                    if usage:
                        if usage.get("prompt_tokens"):
                            prompt_tokens = usage["prompt_tokens"]
                        if usage.get("completion_tokens"):
                            completion_tokens = usage["completion_tokens"]

    except Exception as e:
        return {"error": str(e), "id": request_id}

    end_time = time.perf_counter()

    # Compute TPOT from inter-token intervals
    intervals = []
    for i in range(1, len(token_times)):
        intervals.append((token_times[i] - token_times[i - 1]) * 1000)  # ms

    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
    e2e_ms = (end_time - start_time) * 1000

    # Prefer server-reported completion_tokens over SSE chunk count
    actual_gen = completion_tokens if completion_tokens > 0 else gen_tokens

    return {
        "id": request_id,
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "gen_tokens": actual_gen,
        "gen_tokens_sse": gen_tokens,
        "prompt_tokens": prompt_tokens,
        "token_intervals_ms": intervals,
        "n_intervals": len(intervals),
    }


def generate_prompt(input_len: int) -> str:
    """Generate a prompt that produces approximately `input_len` tokens."""
    # Use repeating pattern; vLLM tokenizer will produce ~input_len tokens.
    # "Hello world " ≈ 2-3 tokens, so we overshoot and rely on max_model_len.
    base = "Repeat the following text exactly: "
    # ~1 token per 4 chars for typical tokenizers
    filler = "abcd " * (input_len // 1)
    return base + filler[:input_len * 4]


def percentile(data, p):
    """Compute p-th percentile."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


async def run_batch(base_url, api_key, model, input_len, batch_size,
                    output_len):
    """Run one batch of concurrent requests and return aggregated metrics."""
    prompt = generate_prompt(input_len)

    connector = aiohttp.TCPConnector(limit=batch_size + 10)
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(connector=connector,
                                     timeout=timeout) as session:
        tasks = [
            make_request(session, base_url, api_key, model, prompt,
                         output_len, i)
            for i in range(batch_size)
        ]
        wall_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        wall_ms = (time.perf_counter() - wall_start) * 1000

    # Filter successful results
    ok_results = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    if errors:
        print(f"    [WARN] {len(errors)} errors: {errors[0].get('error', '')[:80]}")

    if not ok_results:
        return None

    # Aggregate TPOT: use median of per-request median inter-token intervals
    all_tpots = []
    all_intervals = []
    for r in ok_results:
        intervals = r["token_intervals_ms"]
        if intervals:
            all_tpots.append(statistics.median(intervals))
            all_intervals.extend(intervals)

    # Aggregate other metrics
    e2e_times = [r["e2e_ms"] for r in ok_results]
    ttft_times = [r["ttft_ms"] for r in ok_results]
    total_gen = sum(r["gen_tokens"] for r in ok_results)
    total_gen_sse = sum(r.get("gen_tokens_sse", 0) for r in ok_results)
    total_prompt = sum(r["prompt_tokens"] for r in ok_results)

    # sys_tpot: wall_time / total_gen_tokens — system-level TPOT
    sys_tpot = wall_ms / max(total_gen, 1)

    # Per-request TPOT: (e2e - ttft) / (gen_tokens - 1) for each request
    per_req_tpots = []
    for r in ok_results:
        if r["gen_tokens"] > 1 and r["ttft_ms"] > 0:
            decode_ms = r["e2e_ms"] - r["ttft_ms"]
            per_req_tpots.append(decode_ms / (r["gen_tokens"] - 1))

    # Primary TPOT: per-request (e2e-ttft)/(tokens-1), most reliable
    tpot_primary = statistics.mean(per_req_tpots) if per_req_tpots else 0
    tpot_p50_primary = percentile(per_req_tpots, 50) if per_req_tpots else 0

    return {
        "input_len": input_len,
        "batch_size": batch_size,
        "output_len": output_len,
        "n_ok": len(ok_results),
        "wall_ms": round(wall_ms, 2),
        # Primary TPOT: (e2e - ttft) / (gen_tokens - 1) per request, then mean
        "tpot_mean_ms": round(tpot_primary, 2),
        "tpot_p50_ms": round(tpot_p50_primary, 2),
        # SSE interval-based TPOT (may be affected by network buffering)
        "tpot_sse_mean_ms": round(statistics.mean(all_tpots), 2) if all_tpots else 0,
        "tpot_sse_p50_ms": round(percentile(all_tpots, 50), 2) if all_tpots else 0,
        # System TPOT: wall_time / total_tokens
        "sys_tpot_ms": round(sys_tpot, 2),
        "ttft_single_ms": round(ttft_times[0], 2) if ttft_times else 0,
        "ttft_mean_ms": round(statistics.mean(ttft_times), 2) if ttft_times else 0,
        "e2e_mean_ms": round(statistics.mean(e2e_times), 2),
        "e2e_p50_ms": round(percentile(e2e_times, 50), 2),
        "e2e_p95_ms": round(percentile(e2e_times, 95), 2),
        "e2e_p99_ms": round(percentile(e2e_times, 99), 2),
        "gen_tokens": total_gen,
        "gen_tokens_sse": total_gen_sse,
        "prompt_tokens": total_prompt,
        "gen_tps": round(total_gen / (wall_ms / 1000), 1) if wall_ms > 0 else 0,
        "prompt_tps": round(total_prompt / (wall_ms / 1000), 1) if wall_ms > 0 else 0,
    }


async def main():
    parser = argparse.ArgumentParser(description="vLLM batch sweep")
    parser.add_argument("--base-url", default="http://localhost:18000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--preset", default="tp2-fp16",
                        help="Preset label for output metadata")
    parser.add_argument("--input-lens", nargs="+", type=int,
                        default=[128, 512, 2048])
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=[1, 8, 16, 32, 64])
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--output-dir", default="batch_sweep_v4")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Warmup requests before sweep")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"=== Batch Sweep ===")
    print(f"  Server:     {args.base_url}")
    print(f"  Model:      {args.model}")
    print(f"  Preset:     {args.preset}")
    print(f"  Input lens: {args.input_lens}")
    print(f"  Batch sizes:{args.batch_sizes}")
    print(f"  Output len: {args.output_len}")
    print(f"  Output dir: {outdir}")
    print()

    # Warmup
    if args.warmup > 0:
        print(f"[warmup] Sending {args.warmup} warmup request(s)...")
        connector = aiohttp.TCPConnector(limit=10)
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(connector=connector,
                                         timeout=timeout) as session:
            prompt = generate_prompt(128)
            for _ in range(args.warmup):
                r = await make_request(session, args.base_url, args.api_key,
                                       args.model, prompt, 16, 0)
                if "error" in r:
                    print(f"  [WARN] Warmup error: {r['error'][:80]}")
                else:
                    print(f"  [OK] Warmup done (TTFT={r['ttft_ms']:.0f}ms)")
        await asyncio.sleep(2)

    all_results = []
    for input_len in args.input_lens:
        for batch_size in args.batch_sizes:
            print(f"[sweep] input_len={input_len} batch_size={batch_size} ...",
                  end=" ", flush=True)

            result = await run_batch(
                args.base_url, args.api_key, args.model,
                input_len, batch_size, args.output_len
            )

            if result is None:
                print("FAILED (all requests errored)")
                continue

            result["preset"] = args.preset
            print(f"TPOT={result['tpot_mean_ms']:.2f}ms  "
                  f"TPS={result['gen_tps']:.1f}  "
                  f"TTFT={result['ttft_single_ms']:.0f}ms")

            # Save individual file
            fname = f"batch_i{input_len}_b{batch_size}.json"
            with open(outdir / fname, "w") as f:
                json.dump(result, f, indent=2)

            all_results.append(result)

            # Cool down between batches
            await asyncio.sleep(2)

    # Save summary
    with open(outdir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== Sweep Complete ===")
    print(f"  {len(all_results)} data points saved to {outdir}/")

    # Print summary table
    print(f"\n{'input':>8} {'batch':>6} {'TPOT':>8} {'TPS':>8} {'TTFT':>8}")
    print("-" * 42)
    for r in all_results:
        print(f"{r['input_len']:>8} {r['batch_size']:>6} "
              f"{r['tpot_mean_ms']:>7.2f}ms {r['gen_tps']:>7.1f} "
              f"{r['ttft_single_ms']:>7.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
