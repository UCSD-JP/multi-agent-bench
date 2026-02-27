#!/usr/bin/env python3
"""
EP2 vs TP2 Communication Profiling & GPU Kernel Analysis.

Profiles 3 scenarios (ordered by priority):
  Scenario 1: EP2 vs TP2 Communication Pattern comparison
  Scenario 4: Framework Burst Pattern Effect (AutoGen sequential vs A2A parallel)
  Scenario 2: Batch Size Scaling — Compute/Memory Bound Transition

Server: paladin.ucsd.edu:8000 (OpenAI-compatible, vLLM)
Model:  Qwen/Qwen3-Next-80B-A3B-Instruct (or FP8 variant)

Usage:
  # Run all scenarios (client-side load generation)
  python scripts/profile_ep2_vs_tp2.py --run-all

  # Run specific scenario
  python scripts/profile_ep2_vs_tp2.py --scenario 1

  # Parse existing nsys sqlite (post-profiling analysis)
  python scripts/profile_ep2_vs_tp2.py --parse-nsys /path/to/profile.sqlite --label ep2_batch64

  # Parse and compare two profiles
  python scripts/profile_ep2_vs_tp2.py --compare \
      --nsys-a /path/to/tp2_profile.sqlite --label-a tp2 \
      --nsys-b /path/to/ep2_profile.sqlite --label-b ep2

  # Analyze-only: process existing JSON results
  python scripts/profile_ep2_vs_tp2.py --analyze-only

Server-side nsys commands are printed but NOT executed (must run on paladin).

Output: results_multiagent/profiling/
"""

import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

# ===========================================================================
# Configuration
# ===========================================================================

SERVER_HOST = "paladin.ucsd.edu"
SERVER_PORT = 8000
DEFAULT_BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/v1"
DEFAULT_API_KEY = "vllm-key"
DEFAULT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results_multiagent" / "profiling"

# Runtime config (set from CLI args in main)
_config = {
    "base_url": DEFAULT_BASE_URL,
    "api_key": DEFAULT_API_KEY,
    "model": DEFAULT_MODEL,
}

# ===========================================================================
# Kernel classification (v2, ported from gpusim analyze_fp8_decode_hypothesis)
# ===========================================================================
# Includes EP2-specific AllToAll patterns alongside existing TP2 AllReduce.

KERNEL_CATEGORIES = {
    # Attention kernels (KV cache readers)
    "attn": [
        "BatchPrefillWithPagedKVCache",
        "BatchDecodeWithPagedKVCache",
        "flashinfer",
        "flash_attn",
        "FlashAttention",
        "PrefillWithKVCache",
        # DeltaNet (hybrid attention) -- Qwen3-Next specific
        "fused_recurrent_gated_delta_rule",
        "chunk_gated_delta_rule",
        "chunk_fwd_kernel",
        "chunk_scaled_dot_kkt",
        "merge_16x16_to_64x64",
        "recompute_w_u_fwd_kernel",
        "causal_conv1d",
    ],
    # GEMM kernels (weight matmul)
    "gemm": [
        "cutlass",
        "gemm",
        "cublas",
        "nvjet",
        "dot_kernel",
        "reduce_1Block_kernel",
    ],
    # MoE routing/dispatch
    "moe": [
        "fused_moe",
        "topkGating",
        "moe_align_block",
        "count_and_sort_expert",
    ],
    # FP8 quantization/dequantization
    "quant": [
        "per_token_group_quant",
        "quant_8bit",
        "dequant",
        "fp8_quant",
    ],
    # Memory copy/move
    "copy": [
        "direct_copy",
        "CatArrayBatchedCopy",
        "bfloat16_copy",
        "reshape_and_cache",
        "rotary_embedding",
        "elementwise_kernel",
    ],
    # Communication: TP AllReduce + EP AllToAll
    "comm": [
        "ncclKernel",
        "ncclDev",
        "nccl",
        "AllReduce",
        "AllGather",
        "all_reduce",
        "cross_device_reduce",      # vLLM custom AllReduce (non-NCCL)
        "custom_ar",                # vLLM custom all-reduce alias
        # EP2-specific: AllToAll for expert dispatch/combine
        "AllToAll",
        "all_to_all",
        "alltoall",
        "a2a_kernel",              # potential custom EP AllToAll
        "ReduceScatter",
        "reduce_scatter",
    ],
    # Activation / elementwise
    "activation": [
        "act_and_mul",
        "silu_kernel",
        "sigmoid",
        "layer_norm",
        "triton_red_fused",
        "rsqrt",
        "pow_tensor",
        "l2norm",
        "fused_gdn_gating",
    ],
    # Reduction
    "reduce": [
        "reduce_kernel",
        "MeanOps",
        "sum_functor",
    ],
    # Index / scatter / gather
    "index": [
        "index_elementwise",
        "index_select",
        "scatter",
        "gather",
    ],
}


def categorize_kernel(kernel_name: str) -> str:
    """Classify a CUDA kernel name into a category (v2 rules)."""
    name_lower = kernel_name.lower()
    for category, patterns in KERNEL_CATEGORIES.items():
        for pattern in patterns:
            if pattern.lower() in name_lower:
                return category
    return "other"


def categorize_comm_subtype(kernel_name: str) -> str:
    """Further classify comm kernels into AllReduce vs AllToAll vs other."""
    name_lower = kernel_name.lower()
    alltoall_kw = ["alltoall", "all_to_all", "a2a_kernel"]
    allreduce_kw = ["allreduce", "all_reduce", "cross_device_reduce", "custom_ar"]
    allgather_kw = ["allgather", "all_gather"]
    reducescatter_kw = ["reducescatter", "reduce_scatter"]

    for kw in alltoall_kw:
        if kw in name_lower:
            return "AllToAll"
    for kw in allreduce_kw:
        if kw in name_lower:
            return "AllReduce"
    for kw in allgather_kw:
        if kw in name_lower:
            return "AllGather"
    for kw in reducescatter_kw:
        if kw in name_lower:
            return "ReduceScatter"
    if "nccl" in name_lower:
        return "NCCL_other"
    return "comm_other"


# ===========================================================================
# NSys SQLite Parsing
# ===========================================================================

def parse_nsys_sqlite(sqlite_path: str) -> List[Dict]:
    """
    Parse nsys-generated .sqlite file to extract CUDA kernel events.

    Handles both nsys 2024 (direct column names) and nsys 2025+ (StringIds JOIN).
    """
    import sqlite3

    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # Check available tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]
    print(f"  SQLite tables: {tables}")

    has_string_ids = "StringIds" in tables

    # Try JOIN approach first (nsys 2025+)
    if has_string_ids:
        query = """
        SELECT
            k.start,
            k.end,
            (k.end - k.start) as duration_ns,
            COALESCE(sd.value, ss.value, 'unknown') as name
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN StringIds sd ON k.demangledName = sd.id
        LEFT JOIN StringIds ss ON k.shortName = ss.id
        ORDER BY k.start
        """
    else:
        query = """
        SELECT
            start,
            end,
            (end - start) as duration_ns,
            COALESCE(demangledName, shortName, 'unknown') as name
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        ORDER BY start
        """

    try:
        cursor.execute(query)
        rows = cursor.fetchall()
    except Exception as e:
        print(f"  ERROR: SQLite query failed: {e}")
        conn.close()
        return []

    kernels = []
    for row in rows:
        start_ns, end_ns, duration_ns, name = row
        kernels.append({
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_us": duration_ns / 1000.0,
            "name": str(name),
            "category": categorize_kernel(str(name)),
        })

    conn.close()
    print(f"  Parsed {len(kernels)} kernel events from {sqlite_path}")
    return kernels


def extract_decode_window(kernels: List[Dict], skip_first_n: int = 5000) -> List[Dict]:
    """
    Extract decode-phase kernels by skipping prefill.

    Uses step-boundary heuristic: large time gaps between consecutive kernels
    indicate scheduler step boundaries. Prefill occupies the first 1-2 steps.
    """
    if len(kernels) < 100:
        return kernels

    STEP_GAP_THRESHOLD_US = 500
    steps: List[List[Dict]] = []
    current_step: List[Dict] = []

    for i, k in enumerate(kernels):
        if i > 0:
            gap_us = (k["start_ns"] - kernels[i - 1]["start_ns"]) / 1000
            if gap_us > STEP_GAP_THRESHOLD_US:
                if current_step:
                    steps.append(current_step)
                current_step = []
        current_step.append(k)
    if current_step:
        steps.append(current_step)

    if len(steps) < 3:
        # Not enough steps to separate prefill/decode; use simple skip
        return kernels[skip_first_n:] if len(kernels) > skip_first_n else kernels

    # Skip first 2 steps (likely prefill) and return the rest as decode
    decode_kernels = []
    for step in steps[2:]:
        decode_kernels.extend(step)

    return decode_kernels


def aggregate_by_category(kernels: List[Dict]) -> Dict[str, Dict]:
    """Aggregate kernel list by category."""
    categories: Dict[str, Dict] = {}
    for k in kernels:
        cat = k["category"]
        if cat not in categories:
            categories[cat] = {"count": 0, "total_us": 0.0, "kernels": {}}
        categories[cat]["count"] += 1
        categories[cat]["total_us"] += k["duration_us"]

        name = k["name"][:100]
        if name not in categories[cat]["kernels"]:
            categories[cat]["kernels"][name] = {"count": 0, "total_us": 0.0}
        categories[cat]["kernels"][name]["count"] += 1
        categories[cat]["kernels"][name]["total_us"] += k["duration_us"]

    return categories


def aggregate_comm_subtypes(kernels: List[Dict]) -> Dict[str, Dict]:
    """Aggregate communication kernels by subtype (AllReduce, AllToAll, etc.)."""
    comm_kernels = [k for k in kernels if k["category"] == "comm"]
    subtypes: Dict[str, Dict] = {}
    for k in comm_kernels:
        subtype = categorize_comm_subtype(k["name"])
        if subtype not in subtypes:
            subtypes[subtype] = {"count": 0, "total_us": 0.0}
        subtypes[subtype]["count"] += 1
        subtypes[subtype]["total_us"] += k["duration_us"]
    return subtypes


# ===========================================================================
# Client-side load generator (OpenAI-compatible streaming)
# ===========================================================================

async def send_requests(
    num_requests: int,
    input_len: int,
    output_len: int,
    concurrency: int,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    pattern: str = "burst",       # "burst" (all at once), "sequential" (one by one)
    inter_request_delay_s: float = 0.0,
) -> List[Dict]:
    """
    Send requests to the vLLM server and collect per-request metrics.

    Args:
        num_requests: total requests
        input_len: approximate prompt token count (padded with repeated text)
        output_len: max tokens to generate
        concurrency: max in-flight requests
        pattern: "burst" = all at once; "sequential" = one after another
        inter_request_delay_s: delay between sequential sends

    Returns:
        List of per-request timing dicts.
    """
    if base_url is None:
        base_url = _config["base_url"]
    if model is None:
        model = _config["model"]
    if api_key is None:
        api_key = _config["api_key"]

    # Build a prompt of approximately input_len tokens (2 chars per token estimate)
    base_text = (
        "You are a helpful AI assistant. Please analyze the following passage "
        "carefully and provide a detailed summary: "
    )
    # Pad to approximate input_len tokens
    char_target = max(input_len * 2, 100)
    padding_unit = "The quick brown fox jumps over the lazy dog. "
    prompt_text = base_text + (padding_unit * (char_target // len(padding_unit) + 1))
    prompt_text = prompt_text[:char_target]

    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}

    sem = asyncio.Semaphore(concurrency)
    results: List[Dict] = []

    async def single_request(req_id: int, client: httpx.AsyncClient) -> Dict:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0.0,
            "max_tokens": output_len,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        t_send = time.time()
        first_token_time = None
        token_count = 0
        usage_info = {}

        try:
            async with sem:
                async with client.stream(
                    "POST", url, headers=headers, json=payload,
                    timeout=httpx.Timeout(120.0),
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data = line[len("data: "):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        choices = obj.get("choices", [])
                        delta = (
                            choices[0].get("delta", {}).get("content", None)
                            if choices else None
                        )
                        if delta:
                            if first_token_time is None:
                                first_token_time = time.time()
                            token_count += 1

                        usage = obj.get("usage")
                        if usage:
                            usage_info = usage

            t_end = time.time()
            ttft_ms = (first_token_time - t_send) * 1000 if first_token_time else None
            tpot_ms = None
            if first_token_time and token_count > 1:
                tpot_ms = ((t_end - first_token_time) * 1000) / (token_count - 1)

            return {
                "req_id": req_id,
                "ok": True,
                "t_send": t_send,
                "t_first_token": first_token_time,
                "t_end": t_end,
                "ttft_ms": ttft_ms,
                "tpot_ms": tpot_ms,
                "token_count": token_count,
                "prompt_tokens": usage_info.get("prompt_tokens"),
                "completion_tokens": usage_info.get("completion_tokens"),
            }
        except Exception as e:
            return {
                "req_id": req_id,
                "ok": False,
                "error": str(e),
                "t_send": t_send,
                "t_end": time.time(),
            }

    async with httpx.AsyncClient() as client:
        if pattern == "burst":
            tasks = [
                asyncio.create_task(single_request(i, client))
                for i in range(num_requests)
            ]
            results = await asyncio.gather(*tasks)
        elif pattern == "sequential":
            for i in range(num_requests):
                r = await single_request(i, client)
                results.append(r)
                if inter_request_delay_s > 0:
                    await asyncio.sleep(inter_request_delay_s)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

    return list(results)


def print_request_summary(results: List[Dict], label: str) -> Dict:
    """Print and return summary stats from request results."""
    ok_results = [r for r in results if r.get("ok")]
    failed = len(results) - len(ok_results)

    print(f"\n--- {label} ---")
    print(f"  Requests: {len(results)} total, {len(ok_results)} ok, {failed} failed")

    if not ok_results:
        print("  No successful requests.")
        return {}

    ttfts = [r["ttft_ms"] for r in ok_results if r.get("ttft_ms") is not None]
    tpots = [r["tpot_ms"] for r in ok_results if r.get("tpot_ms") is not None]
    e2es = [(r["t_end"] - r["t_send"]) * 1000 for r in ok_results]

    def stats(vals: List[float]) -> str:
        if not vals:
            return "N/A"
        return (
            f"mean={statistics.mean(vals):.1f}ms  "
            f"p50={sorted(vals)[len(vals)//2]:.1f}ms  "
            f"p95={sorted(vals)[int(len(vals)*0.95)]:.1f}ms"
        )

    print(f"  TTFT:  {stats(ttfts)}")
    print(f"  TPOT:  {stats(tpots)}")
    print(f"  E2E:   {stats(e2es)}")

    total_tokens = sum(r.get("token_count", 0) for r in ok_results)
    wall_s = max(r["t_end"] for r in ok_results) - min(r["t_send"] for r in ok_results)
    if wall_s > 0:
        print(f"  Throughput: {total_tokens / wall_s:.1f} tok/s (wall), {wall_s:.2f}s wall")

    return {
        "label": label,
        "total_requests": len(results),
        "ok_requests": len(ok_results),
        "failed": failed,
        "ttft_mean_ms": statistics.mean(ttfts) if ttfts else None,
        "tpot_mean_ms": statistics.mean(tpots) if tpots else None,
        "e2e_mean_ms": statistics.mean(e2es) if e2es else None,
        "throughput_tok_s": total_tokens / wall_s if wall_s > 0 else None,
        "wall_s": wall_s,
    }


# ===========================================================================
# Scenario 1: EP2 vs TP2 Communication Pattern
# ===========================================================================

def print_server_nsys_commands_scenario1():
    """Print nsys commands to run on the server for Scenario 1."""
    print("""
================================================================================
SCENARIO 1: EP2 vs TP2 Communication Pattern
================================================================================

SERVER-SIDE nsys COMMANDS (run on paladin.ucsd.edu):
----------------------------------------------------

The server must be started under nsys profiling. Stop the existing vLLM server
and restart it under nsys for each configuration.

--- TP2 Configuration ---
# 1. Start vLLM with TP=2 under nsys:
nsys profile \\
    --output /tmp/nsys_tp2_comm_b64 \\
    --force-overwrite true \\
    --trace cuda,cudnn,cublas,nvtx \\
    --cuda-memory-usage true \\
    --stats true \\
    --export sqlite \\
    -- python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen3-Next-80B-A3B-Instruct \\
        --tensor-parallel-size 2 \\
        --host 0.0.0.0 --port 8000 \\
        --max-model-len 4096

# 2. Then from the client, run:
#    python scripts/profile_ep2_vs_tp2.py --scenario 1
# 3. After client finishes, Ctrl+C the server to finalize nsys trace.
# 4. Copy: scp paladin:/tmp/nsys_tp2_comm_b64.sqlite results_multiagent/profiling/

--- EP2 Configuration ---
# 1. Start vLLM with EP=2 (expert parallelism via pipeline-parallel or EP flag):
nsys profile \\
    --output /tmp/nsys_ep2_comm_b64 \\
    --force-overwrite true \\
    --trace cuda,cudnn,cublas,nvtx \\
    --cuda-memory-usage true \\
    --stats true \\
    --export sqlite \\
    -- python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen3-Next-80B-A3B-Instruct \\
        --tensor-parallel-size 1 \\
        --num-scheduler-steps 1 \\
        --enable-expert-parallel \\
        --host 0.0.0.0 --port 8000 \\
        --max-model-len 4096

# NOTE: vLLM EP flag may vary by version. Check --help for:
#   --enable-expert-parallel, --expert-parallel-size 2, or
#   environment variable VLLM_EP_SIZE=2.
# For DeepSeek-style EP, try: --distributed-executor-backend mp
#   with CUDA_VISIBLE_DEVICES=0,1 and --expert-parallel-size 2

# 2. Then from the client, run:
#    python scripts/profile_ep2_vs_tp2.py --scenario 1
# 3. Ctrl+C server, copy sqlite as above.

EXPECTED DIFF:
  - TP2: comm kernels dominated by AllReduce (vLLM cross_device_reduce or NCCL)
  - EP2: comm kernels dominated by AllToAll (expert dispatch/combine)
  - EP2 should show more MoE kernel time, less GEMM (experts only on 1 GPU)
  - TP2 comm is 2x per-layer; EP2 comm is 2x per-MoE-layer only
================================================================================
""")


async def run_scenario1_client(output_dir: Path) -> Dict:
    """Scenario 1 client: send batch=64 requests to capture decode under profiling."""
    print("\n[Scenario 1] Sending batch=64 requests (input_len=2048, output_len=32)...")
    print("  Make sure the server is running under nsys profiling!\n")

    results = await send_requests(
        num_requests=64,
        input_len=2048,
        output_len=32,
        concurrency=64,
        pattern="burst",
    )

    summary = print_request_summary(results, "Scenario 1: EP2/TP2 comm comparison load")

    out_path = output_dir / "scenario1_client_results.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "requests": results}, f, indent=2, default=str)
    print(f"  Client results saved: {out_path}")

    return summary


def analyze_scenario1(
    tp2_sqlite: Optional[str],
    ep2_sqlite: Optional[str],
    output_dir: Path,
) -> None:
    """Analyze nsys profiles comparing TP2 vs EP2 communication patterns."""
    results = {}

    for label, path in [("tp2", tp2_sqlite), ("ep2", ep2_sqlite)]:
        if not path or not os.path.exists(path):
            print(f"  [SKIP] {label} sqlite not found: {path}")
            continue

        print(f"\n--- Analyzing {label}: {path} ---")
        kernels = parse_nsys_sqlite(path)
        decode_kernels = extract_decode_window(kernels)
        categories = aggregate_by_category(decode_kernels)
        comm_subtypes = aggregate_comm_subtypes(decode_kernels)

        total_us = sum(c["total_us"] for c in categories.values())

        print(f"\n  {'Category':<14} {'Count':>8} {'Total (ms)':>12} {'% of total':>10}")
        print(f"  {'-'*48}")
        for cat in sorted(categories.keys(), key=lambda c: -categories[c]["total_us"]):
            d = categories[cat]
            pct = d["total_us"] / total_us * 100 if total_us > 0 else 0
            print(f"  {cat:<14} {d['count']:>8} {d['total_us']/1000:>12.1f} {pct:>9.1f}%")

        print(f"\n  Communication subtypes:")
        for st in sorted(comm_subtypes.keys(), key=lambda s: -comm_subtypes[s]["total_us"]):
            d = comm_subtypes[st]
            comm_total = sum(v["total_us"] for v in comm_subtypes.values())
            pct = d["total_us"] / comm_total * 100 if comm_total > 0 else 0
            print(f"    {st:<20} {d['count']:>6} {d['total_us']/1000:>10.1f}ms ({pct:>5.1f}% of comm)")

        results[label] = {
            "categories": {
                cat: {"count": d["count"], "total_ms": round(d["total_us"] / 1000, 2)}
                for cat, d in categories.items()
            },
            "comm_subtypes": {
                st: {"count": d["count"], "total_ms": round(d["total_us"] / 1000, 2)}
                for st, d in comm_subtypes.items()
            },
            "total_decode_ms": round(total_us / 1000, 2),
            "top_kernels": {},
        }

        # Save top kernels per category
        for cat, data in categories.items():
            top = sorted(
                [{"name": k, **v} for k, v in data["kernels"].items()],
                key=lambda x: -x["total_us"],
            )[:5]
            results[label]["top_kernels"][cat] = [
                {"name": t["name"], "count": t["count"], "total_ms": round(t["total_us"] / 1000, 2)}
                for t in top
            ]

    # Comparison
    if "tp2" in results and "ep2" in results:
        print("\n" + "=" * 60)
        print("EP2 vs TP2 COMPARISON")
        print("=" * 60)

        all_cats = sorted(
            set(list(results["tp2"]["categories"].keys()) +
                list(results["ep2"]["categories"].keys()))
        )

        print(f"\n  {'Category':<14} {'TP2 (ms)':>10} {'EP2 (ms)':>10} {'EP2/TP2':>8}")
        print(f"  {'-'*46}")
        for cat in all_cats:
            tp2_ms = results["tp2"]["categories"].get(cat, {}).get("total_ms", 0)
            ep2_ms = results["ep2"]["categories"].get(cat, {}).get("total_ms", 0)
            ratio = ep2_ms / tp2_ms if tp2_ms > 0 else float("inf")
            print(f"  {cat:<14} {tp2_ms:>10.1f} {ep2_ms:>10.1f} {ratio:>7.2f}x")

        # Comm subtype comparison
        print(f"\n  Communication Subtype Detail:")
        tp2_subtypes = results["tp2"].get("comm_subtypes", {})
        ep2_subtypes = results["ep2"].get("comm_subtypes", {})
        all_subtypes = sorted(set(list(tp2_subtypes.keys()) + list(ep2_subtypes.keys())))

        for st in all_subtypes:
            tp2_ms = tp2_subtypes.get(st, {}).get("total_ms", 0)
            ep2_ms = ep2_subtypes.get(st, {}).get("total_ms", 0)
            print(f"    {st:<20} TP2={tp2_ms:>8.1f}ms  EP2={ep2_ms:>8.1f}ms")

    # Save
    out_path = output_dir / "scenario1_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Analysis saved: {out_path}")


# ===========================================================================
# Scenario 4: Framework Burst Pattern Effect
# ===========================================================================

def print_server_nsys_commands_scenario4():
    """Print nsys commands for Scenario 4."""
    print("""
================================================================================
SCENARIO 4: Framework Burst Pattern Effect (AutoGen vs A2A)
================================================================================

SERVER-SIDE nsys COMMANDS (run on paladin.ucsd.edu):
----------------------------------------------------

Start the vLLM server ONCE under nsys, then run both client patterns.
The nsys trace will capture the interleaved patterns from both runs.

# Approach A: Single long trace covering both patterns
nsys profile \\
    --output /tmp/nsys_framework_burst \\
    --force-overwrite true \\
    --trace cuda,cudnn,cublas,nvtx \\
    --cuda-memory-usage true \\
    --stats true \\
    --export sqlite \\
    -- python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen3-Next-80B-A3B-Instruct \\
        --tensor-parallel-size 2 \\
        --host 0.0.0.0 --port 8000 \\
        --max-model-len 4096

# Then from client:
#   python scripts/profile_ep2_vs_tp2.py --scenario 4

# Approach B: Separate traces per pattern (recommended)
# Run server under nsys for sequential pattern:
nsys profile \\
    --output /tmp/nsys_sequential_pattern \\
    --force-overwrite true \\
    --trace cuda,cudnn,cublas,nvtx \\
    --cuda-memory-usage true \\
    --stats true \\
    --export sqlite \\
    -- python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen3-Next-80B-A3B-Instruct \\
        --tensor-parallel-size 2 \\
        --host 0.0.0.0 --port 8000 \\
        --max-model-len 4096

# Client: python scripts/profile_ep2_vs_tp2.py --scenario 4 --pattern sequential
# Ctrl+C server, restart with new nsys output for parallel pattern.

EXPECTED DIFF:
  - Sequential (AutoGen): GPU utilization has idle gaps between steps
  - Parallel (A2A): Higher batch occupancy, fewer idle gaps
  - Sequential: smaller effective batch -> more memory-bound kernels
  - Parallel: larger effective batch -> more compute-bound kernels
================================================================================
""")


async def run_scenario4_client(output_dir: Path, pattern: str = "both") -> Dict:
    """
    Scenario 4 client: mimic AutoGen (sequential) vs A2A (parallel burst).

    AutoGen pattern: Sequential requests (1 at a time, chained like plan->exec->exec->agg)
    A2A pattern: Parallel burst (plan, then 2 execs in parallel, then aggregation)
    """
    results = {}
    input_len = 1024
    output_len = 128

    if pattern in ("both", "sequential"):
        print("\n[Scenario 4a] AutoGen-like sequential pattern")
        print("  Sending 8 tasks, each as 4 sequential calls (plan->exec1->exec2->agg)")

        all_seq_results = []
        for task_id in range(8):
            # 4 sequential calls per task (mimics autogen selector->executor->executor->aggregator)
            task_results = await send_requests(
                num_requests=4,
                input_len=input_len,
                output_len=output_len,
                concurrency=1,          # strictly sequential
                pattern="sequential",
                inter_request_delay_s=0.01,  # small gap between calls
            )
            all_seq_results.extend(task_results)

        seq_summary = print_request_summary(all_seq_results, "Scenario 4a: Sequential (AutoGen-like)")
        results["sequential"] = seq_summary

        # Wait for server to settle
        await asyncio.sleep(2.0)

    if pattern in ("both", "parallel"):
        print("\n[Scenario 4b] A2A-like parallel burst pattern")
        print("  Sending 8 tasks, each as: 1 plan + 2 parallel execs + 1 agg")

        all_par_results = []
        for task_id in range(8):
            # Step 1: Plan (1 call)
            plan = await send_requests(
                num_requests=1, input_len=input_len, output_len=output_len,
                concurrency=1, pattern="sequential",
            )
            all_par_results.extend(plan)

            # Step 2: 2 executor calls in parallel (A2A parallel dispatch)
            execs = await send_requests(
                num_requests=2, input_len=input_len, output_len=output_len,
                concurrency=2, pattern="burst",
            )
            all_par_results.extend(execs)

            # Step 3: Aggregation (1 call)
            agg = await send_requests(
                num_requests=1, input_len=input_len, output_len=output_len,
                concurrency=1, pattern="sequential",
            )
            all_par_results.extend(agg)

        par_summary = print_request_summary(all_par_results, "Scenario 4b: Parallel burst (A2A-like)")
        results["parallel"] = par_summary

    out_path = output_dir / "scenario4_client_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Client results saved: {out_path}")

    return results


def analyze_scenario4(
    sequential_sqlite: Optional[str],
    parallel_sqlite: Optional[str],
    output_dir: Path,
) -> None:
    """Analyze GPU utilization patterns from sequential vs parallel request arrivals."""
    results = {}

    for label, path in [("sequential", sequential_sqlite), ("parallel", parallel_sqlite)]:
        if not path or not os.path.exists(path):
            print(f"  [SKIP] {label} sqlite not found: {path}")
            continue

        print(f"\n--- Analyzing {label}: {path} ---")
        kernels = parse_nsys_sqlite(path)

        if not kernels:
            continue

        # Compute GPU utilization: fraction of wall time with active kernels
        first_ns = kernels[0]["start_ns"]
        last_ns = kernels[-1]["end_ns"]
        wall_ns = last_ns - first_ns

        # Merge overlapping kernel intervals to compute active time
        intervals = sorted([(k["start_ns"], k["end_ns"]) for k in kernels])
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        active_ns = sum(end - start for start, end in merged)
        utilization = active_ns / wall_ns * 100 if wall_ns > 0 else 0

        # Compute idle gaps between merged intervals
        gaps = []
        for i in range(1, len(merged)):
            gap_ns = merged[i][0] - merged[i - 1][1]
            if gap_ns > 0:
                gaps.append(gap_ns / 1000)  # in microseconds

        # Category breakdown
        categories = aggregate_by_category(kernels)
        total_us = sum(c["total_us"] for c in categories.values())

        print(f"  GPU utilization: {utilization:.1f}%")
        print(f"  Wall time: {wall_ns/1e9:.3f}s")
        print(f"  Active time: {active_ns/1e9:.3f}s")
        print(f"  Idle gaps: {len(gaps)} total")
        if gaps:
            print(f"    Mean gap: {statistics.mean(gaps):.1f}us")
            print(f"    Max gap: {max(gaps):.1f}us")
            print(f"    Gaps > 100us: {sum(1 for g in gaps if g > 100)}")
            print(f"    Gaps > 1ms: {sum(1 for g in gaps if g > 1000)}")

        print(f"\n  {'Category':<14} {'Count':>8} {'Total (ms)':>12} {'% of total':>10}")
        print(f"  {'-'*48}")
        for cat in sorted(categories.keys(), key=lambda c: -categories[c]["total_us"]):
            d = categories[cat]
            pct = d["total_us"] / total_us * 100 if total_us > 0 else 0
            print(f"  {cat:<14} {d['count']:>8} {d['total_us']/1000:>12.1f} {pct:>9.1f}%")

        results[label] = {
            "gpu_utilization_pct": round(utilization, 2),
            "wall_s": round(wall_ns / 1e9, 3),
            "active_s": round(active_ns / 1e9, 3),
            "total_kernels": len(kernels),
            "idle_gap_count": len(gaps),
            "idle_gap_mean_us": round(statistics.mean(gaps), 1) if gaps else 0,
            "idle_gap_max_us": round(max(gaps), 1) if gaps else 0,
            "gaps_over_100us": sum(1 for g in gaps if g > 100),
            "gaps_over_1ms": sum(1 for g in gaps if g > 1000),
            "categories": {
                cat: {"count": d["count"], "total_ms": round(d["total_us"] / 1000, 2)}
                for cat, d in categories.items()
            },
        }

    # Comparison
    if "sequential" in results and "parallel" in results:
        print("\n" + "=" * 60)
        print("SEQUENTIAL vs PARALLEL COMPARISON")
        print("=" * 60)

        s = results["sequential"]
        p = results["parallel"]
        print(f"  {'Metric':<30} {'Sequential':>12} {'Parallel':>12}")
        print(f"  {'-'*58}")
        print(f"  {'GPU utilization (%)':<30} {s['gpu_utilization_pct']:>12.1f} {p['gpu_utilization_pct']:>12.1f}")
        print(f"  {'Wall time (s)':<30} {s['wall_s']:>12.3f} {p['wall_s']:>12.3f}")
        print(f"  {'Active time (s)':<30} {s['active_s']:>12.3f} {p['active_s']:>12.3f}")
        print(f"  {'Idle gaps > 1ms':<30} {s['gaps_over_1ms']:>12} {p['gaps_over_1ms']:>12}")
        print(f"  {'Mean idle gap (us)':<30} {s['idle_gap_mean_us']:>12.1f} {p['idle_gap_mean_us']:>12.1f}")

    out_path = output_dir / "scenario4_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Analysis saved: {out_path}")


# ===========================================================================
# Scenario 2: Batch Size Scaling — Compute/Memory Bound Transition
# ===========================================================================

def print_server_ncu_commands_scenario2():
    """Print ncu (Nsight Compute) commands for Scenario 2."""
    print("""
================================================================================
SCENARIO 2: Batch Size Scaling — Compute/Memory Bound Transition
================================================================================

SERVER-SIDE ncu COMMANDS (run on paladin.ucsd.edu):
----------------------------------------------------

NCU requires running the inference OFFLINE (not as a server) because it injects
kernel replay and slows execution by 100-1000x. Use vLLM offline benchmark.

BATCH SIZES to test: 1, 8, 32, 64
Fixed: input_len=2048, output_len=16 (short decode for fast profiling)

--- GEMM kernel profiling (compute vs memory bound) ---

for BATCH in 1 8 32 64; do
  echo "=== NCU GEMM batch=$BATCH ==="
  ncu --target-processes all \\
      --set full \\
      --kernel-name-base function \\
      --kernel-name "cutlass_3x_gemm|cutlass.*gemm|nvjet_tst|cublas" \\
      --launch-skip 1000 \\
      --launch-count 200 \\
      --output /tmp/ncu_gemm_b${BATCH}_i2048 \\
      -- python -m vllm.entrypoints.cli.main bench throughput \\
          --model Qwen/Qwen3-Next-80B-A3B-Instruct \\
          --tensor-parallel-size 2 \\
          --input-len 2048 \\
          --output-len 16 \\
          --num-prompts $BATCH \\
          --dtype auto
done

--- Attention kernel profiling (KV cache pressure) ---

for BATCH in 1 8 32 64; do
  echo "=== NCU ATTN batch=$BATCH ==="
  ncu --target-processes all \\
      --set full \\
      --kernel-name-base function \\
      --kernel-name "BatchDecodeWithPagedKVCache|fused_recurrent_gated_delta" \\
      --launch-skip 200 \\
      --launch-count 100 \\
      --output /tmp/ncu_attn_b${BATCH}_i2048 \\
      -- python -m vllm.entrypoints.cli.main bench throughput \\
          --model Qwen/Qwen3-Next-80B-A3B-Instruct \\
          --tensor-parallel-size 2 \\
          --input-len 2048 \\
          --output-len 16 \\
          --num-prompts $BATCH \\
          --dtype auto
done

--- MoE kernel profiling ---

for BATCH in 1 8 32 64; do
  echo "=== NCU MoE batch=$BATCH ==="
  ncu --target-processes all \\
      --set full \\
      --kernel-name-base function \\
      --kernel-name "fused_moe" \\
      --launch-skip 500 \\
      --launch-count 100 \\
      --output /tmp/ncu_moe_b${BATCH}_i2048 \\
      -- python -m vllm.entrypoints.cli.main bench throughput \\
          --model Qwen/Qwen3-Next-80B-A3B-Instruct \\
          --tensor-parallel-size 2 \\
          --input-len 2048 \\
          --output-len 16 \\
          --num-prompts $BATCH \\
          --dtype auto
done

METRICS TO EXTRACT (after NCU profiling):
  ncu --import /tmp/ncu_gemm_b1_i2048.ncu-rep \\
      --csv --page raw \\
      --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\\
sm__warps_active.avg.pct_of_peak_sustained_elapsed,\\
smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_elapsed

KEY INDICATORS:
  - sm__throughput > dram__throughput  => compute-bound
  - dram__throughput > sm__throughput  => memory-bound
  - Crossover batch where ratio flips  => the transition point
================================================================================
""")


async def run_scenario2_client(output_dir: Path) -> Dict:
    """
    Scenario 2 client: send batch=1,8,32,64 requests sequentially.

    This is the CLIENT-SIDE load generator. For NCU profiling, use the
    offline benchmark commands printed above instead. This client variant
    is for nsys-based coarse analysis (category timing per batch size).
    """
    batch_sizes = [1, 8, 32, 64]
    input_len = 2048
    output_len = 32
    results_all = {}

    for batch in batch_sizes:
        print(f"\n[Scenario 2] batch={batch}, input_len={input_len}, output_len={output_len}")

        # Send all requests as a burst; concurrency = batch
        results = await send_requests(
            num_requests=batch,
            input_len=input_len,
            output_len=output_len,
            concurrency=batch,
            pattern="burst",
        )

        summary = print_request_summary(results, f"Scenario 2: batch={batch}")
        results_all[f"batch_{batch}"] = {"summary": summary, "requests": results}

        # Wait between batch sizes for clean separation
        await asyncio.sleep(3.0)

    out_path = output_dir / "scenario2_client_results.json"
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2, default=str)
    print(f"  Client results saved: {out_path}")

    return results_all


def print_server_nsys_commands_scenario2():
    """Print nsys commands for Scenario 2 (coarse, batch-size sweep)."""
    print("""
--- Alternative: nsys batch-size sweep (coarse category analysis) ---

Start vLLM under nsys, then send batch=1,8,32,64 from client.
Separate traces recommended for clean per-batch analysis:

for BATCH in 1 8 32 64; do
  echo "=== nsys batch=$BATCH ==="
  nsys profile \\
      --output /tmp/nsys_batch${BATCH}_i2048 \\
      --force-overwrite true \\
      --trace cuda,cudnn,cublas,nvtx \\
      --cuda-memory-usage true \\
      --stats true \\
      --export sqlite \\
      -- python -m vllm.entrypoints.cli.main bench throughput \\
          --model Qwen/Qwen3-Next-80B-A3B-Instruct \\
          --tensor-parallel-size 2 \\
          --input-len 2048 \\
          --output-len 32 \\
          --num-prompts $BATCH \\
          --dtype auto
done
""")


def analyze_scenario2_nsys(sqlite_paths: Dict[int, str], output_dir: Path) -> None:
    """Analyze nsys profiles across batch sizes for compute/memory bound transition."""
    results = {}

    for batch_size in sorted(sqlite_paths.keys()):
        path = sqlite_paths[batch_size]
        if not os.path.exists(path):
            print(f"  [SKIP] batch={batch_size}: {path} not found")
            continue

        print(f"\n--- batch={batch_size}: {path} ---")
        kernels = parse_nsys_sqlite(path)
        decode_kernels = extract_decode_window(kernels)
        categories = aggregate_by_category(decode_kernels)
        total_us = sum(c["total_us"] for c in categories.values())

        print(f"  {'Category':<14} {'Count':>8} {'Total (ms)':>12} {'% of total':>10}")
        print(f"  {'-'*48}")
        for cat in sorted(categories.keys(), key=lambda c: -categories[c]["total_us"]):
            d = categories[cat]
            pct = d["total_us"] / total_us * 100 if total_us > 0 else 0
            print(f"  {cat:<14} {d['count']:>8} {d['total_us']/1000:>12.1f} {pct:>9.1f}%")

        results[batch_size] = {
            "total_decode_ms": round(total_us / 1000, 2),
            "categories": {
                cat: {
                    "count": d["count"],
                    "total_ms": round(d["total_us"] / 1000, 2),
                    "pct": round(d["total_us"] / total_us * 100, 1) if total_us > 0 else 0,
                }
                for cat, d in categories.items()
            },
        }

    if len(results) > 1:
        print("\n" + "=" * 60)
        print("BATCH SIZE SCALING COMPARISON")
        print("=" * 60)

        all_cats = sorted(set(
            cat for r in results.values() for cat in r["categories"].keys()
        ))
        batch_sizes = sorted(results.keys())

        header = f"  {'Category':<14}" + "".join(f"  b={b:>3} (ms)" for b in batch_sizes)
        print(header)
        print(f"  {'-' * (14 + 14 * len(batch_sizes))}")

        for cat in all_cats:
            row = f"  {cat:<14}"
            for b in batch_sizes:
                ms = results[b]["categories"].get(cat, {}).get("total_ms", 0)
                row += f"  {ms:>10.1f}"
            print(row)

        # Compute gemm/attn ratio as proxy for compute vs memory bound
        print(f"\n  GEMM/ATTN ratio (higher = more compute-bound):")
        for b in batch_sizes:
            gemm_ms = results[b]["categories"].get("gemm", {}).get("total_ms", 0)
            attn_ms = results[b]["categories"].get("attn", {}).get("total_ms", 0)
            ratio = gemm_ms / attn_ms if attn_ms > 0 else float("inf")
            print(f"    batch={b:>3}: GEMM={gemm_ms:.1f}ms  ATTN={attn_ms:.1f}ms  ratio={ratio:.2f}")

    out_path = output_dir / "scenario2_analysis.json"
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\n  Analysis saved: {out_path}")


# ===========================================================================
# Generic parse & compare
# ===========================================================================

def parse_and_print(sqlite_path: str, label: str, output_dir: Path) -> Dict:
    """Parse a single nsys sqlite and print category breakdown."""
    print(f"\n{'='*60}")
    print(f"Parsing: {label} ({sqlite_path})")
    print(f"{'='*60}")

    kernels = parse_nsys_sqlite(sqlite_path)
    if not kernels:
        return {}

    decode_kernels = extract_decode_window(kernels)
    categories = aggregate_by_category(decode_kernels)
    comm_subtypes = aggregate_comm_subtypes(decode_kernels)
    total_us = sum(c["total_us"] for c in categories.values())

    print(f"\n  Total kernels: {len(kernels)}")
    print(f"  Decode kernels: {len(decode_kernels)}")
    print(f"  Total decode time: {total_us/1000:.1f}ms")

    print(f"\n  {'Category':<14} {'Count':>8} {'Total (ms)':>12} {'% of total':>10}")
    print(f"  {'-'*48}")
    for cat in sorted(categories.keys(), key=lambda c: -categories[c]["total_us"]):
        d = categories[cat]
        pct = d["total_us"] / total_us * 100 if total_us > 0 else 0
        print(f"  {cat:<14} {d['count']:>8} {d['total_us']/1000:>12.1f} {pct:>9.1f}%")

    if comm_subtypes:
        print(f"\n  Communication subtypes:")
        for st in sorted(comm_subtypes.keys(), key=lambda s: -comm_subtypes[s]["total_us"]):
            d = comm_subtypes[st]
            print(f"    {st:<20} {d['count']:>6} {d['total_us']/1000:>10.1f}ms")

    # Top kernels per category
    print(f"\n  Top kernels per category:")
    for cat in sorted(categories.keys(), key=lambda c: -categories[c]["total_us"]):
        data = categories[cat]
        top = sorted(
            [{"name": k, **v} for k, v in data["kernels"].items()],
            key=lambda x: -x["total_us"],
        )[:3]
        print(f"    [{cat}]")
        for t in top:
            print(f"      {t['name'][:70]:<72} {t['total_us']/1000:>8.2f}ms ({t['count']}x)")

    # Save result
    result = {
        "label": label,
        "source": sqlite_path,
        "total_kernels": len(kernels),
        "decode_kernels": len(decode_kernels),
        "total_decode_ms": round(total_us / 1000, 2),
        "categories": {
            cat: {
                "count": d["count"],
                "total_ms": round(d["total_us"] / 1000, 2),
                "pct": round(d["total_us"] / total_us * 100, 1) if total_us > 0 else 0,
                "top_kernels": sorted(
                    [{"name": k, "count": v["count"], "total_ms": round(v["total_us"]/1000, 2)}
                     for k, v in data["kernels"].items()],
                    key=lambda x: -x["total_ms"],
                )[:5],
            }
            for cat, d in categories.items()
        },
        "comm_subtypes": {
            st: {"count": d["count"], "total_ms": round(d["total_us"] / 1000, 2)}
            for st, d in comm_subtypes.items()
        },
    }

    out_path = output_dir / f"parsed_{label}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out_path}")

    return result


def compare_profiles(result_a: Dict, result_b: Dict) -> None:
    """Side-by-side comparison of two parsed profiles."""
    label_a = result_a.get("label", "A")
    label_b = result_b.get("label", "B")

    print(f"\n{'='*60}")
    print(f"COMPARISON: {label_a} vs {label_b}")
    print(f"{'='*60}")

    cats_a = result_a.get("categories", {})
    cats_b = result_b.get("categories", {})
    all_cats = sorted(set(list(cats_a.keys()) + list(cats_b.keys())))

    total_a = result_a.get("total_decode_ms", 0)
    total_b = result_b.get("total_decode_ms", 0)

    print(f"\n  {'Category':<14} {label_a+' (ms)':>12} {label_a+' %':>8} "
          f"{label_b+' (ms)':>12} {label_b+' %':>8} {'Ratio':>8}")
    print(f"  {'-'*66}")

    for cat in all_cats:
        ms_a = cats_a.get(cat, {}).get("total_ms", 0)
        ms_b = cats_b.get(cat, {}).get("total_ms", 0)
        pct_a = ms_a / total_a * 100 if total_a > 0 else 0
        pct_b = ms_b / total_b * 100 if total_b > 0 else 0
        ratio = ms_b / ms_a if ms_a > 0 else float("inf")
        print(f"  {cat:<14} {ms_a:>12.1f} {pct_a:>7.1f}% {ms_b:>12.1f} {pct_b:>7.1f}% {ratio:>7.2f}x")

    print(f"  {'TOTAL':<14} {total_a:>12.1f} {'100.0%':>8} {total_b:>12.1f} {'100.0%':>8} "
          f"{total_b/total_a if total_a > 0 else 0:>7.2f}x")

    # Comm subtype comparison
    comm_a = result_a.get("comm_subtypes", {})
    comm_b = result_b.get("comm_subtypes", {})
    if comm_a or comm_b:
        all_subtypes = sorted(set(list(comm_a.keys()) + list(comm_b.keys())))
        print(f"\n  Communication Subtypes:")
        for st in all_subtypes:
            ms_a = comm_a.get(st, {}).get("total_ms", 0)
            ms_b = comm_b.get(st, {}).get("total_ms", 0)
            print(f"    {st:<20} {label_a}={ms_a:>8.1f}ms  {label_b}={ms_b:>8.1f}ms")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EP2 vs TP2 Communication Profiling & GPU Kernel Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Print all server-side profiling commands
  python scripts/profile_ep2_vs_tp2.py --print-commands

  # Run client load for scenario 1 (while server is under nsys)
  python scripts/profile_ep2_vs_tp2.py --scenario 1

  # Run all client-side scenarios
  python scripts/profile_ep2_vs_tp2.py --run-all

  # Parse a single nsys sqlite
  python scripts/profile_ep2_vs_tp2.py --parse-nsys /path/to/profile.sqlite --label tp2_b64

  # Compare two profiles
  python scripts/profile_ep2_vs_tp2.py --compare \\
      --nsys-a /path/to/tp2.sqlite --label-a tp2 \\
      --nsys-b /path/to/ep2.sqlite --label-b ep2

  # Analyze scenario 1 with existing nsys data
  python scripts/profile_ep2_vs_tp2.py --analyze-scenario1 \\
      --tp2-sqlite results_multiagent/profiling/nsys_tp2_comm_b64.sqlite \\
      --ep2-sqlite results_multiagent/profiling/nsys_ep2_comm_b64.sqlite

  # Analyze scenario 2 with existing nsys data
  python scripts/profile_ep2_vs_tp2.py --analyze-scenario2 \\
      --batch-sqlites "1:/path/b1.sqlite,8:/path/b8.sqlite,32:/path/b32.sqlite,64:/path/b64.sqlite"
        """,
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run-all", action="store_true",
                      help="Run all scenarios (client-side load generation)")
    mode.add_argument("--scenario", type=int, choices=[1, 2, 4],
                      help="Run specific scenario (client-side)")
    mode.add_argument("--print-commands", action="store_true",
                      help="Print all server-side nsys/ncu commands")
    mode.add_argument("--parse-nsys", type=str, metavar="SQLITE_PATH",
                      help="Parse a single nsys sqlite file")
    mode.add_argument("--compare", action="store_true",
                      help="Compare two nsys profiles")
    mode.add_argument("--analyze-scenario1", action="store_true",
                      help="Analyze scenario 1 from existing nsys sqlite files")
    mode.add_argument("--analyze-scenario2", action="store_true",
                      help="Analyze scenario 2 from existing nsys sqlite files")
    mode.add_argument("--analyze-scenario4", action="store_true",
                      help="Analyze scenario 4 from existing nsys sqlite files")

    # Parse/compare options
    parser.add_argument("--label", type=str, default="profile",
                        help="Label for --parse-nsys output")
    parser.add_argument("--nsys-a", type=str, help="First sqlite for --compare")
    parser.add_argument("--label-a", type=str, default="A", help="Label for first profile")
    parser.add_argument("--nsys-b", type=str, help="Second sqlite for --compare")
    parser.add_argument("--label-b", type=str, default="B", help="Label for second profile")

    # Scenario 1 analysis
    parser.add_argument("--tp2-sqlite", type=str, help="TP2 nsys sqlite for scenario 1")
    parser.add_argument("--ep2-sqlite", type=str, help="EP2 nsys sqlite for scenario 1")

    # Scenario 2 analysis
    parser.add_argument("--batch-sqlites", type=str,
                        help="Comma-separated batch:path pairs, e.g. '1:/p/b1.sqlite,8:/p/b8.sqlite'")

    # Scenario 4 analysis
    parser.add_argument("--seq-sqlite", type=str,
                        help="Sequential pattern nsys sqlite for scenario 4")
    parser.add_argument("--par-sqlite", type=str,
                        help="Parallel pattern nsys sqlite for scenario 4")

    # Scenario 4 client options
    parser.add_argument("--pattern", type=str, default="both",
                        choices=["both", "sequential", "parallel"],
                        help="Pattern for scenario 4 client (default: both)")

    # General
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL,
                        help="Server base URL")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Model name")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Override runtime config from CLI args
    _config["base_url"] = args.base_url
    _config["model"] = args.model

    # ---------------------------------------------------------------
    # Print server commands
    # ---------------------------------------------------------------
    if args.print_commands:
        print_server_nsys_commands_scenario1()
        print_server_nsys_commands_scenario4()
        print_server_ncu_commands_scenario2()
        print_server_nsys_commands_scenario2()
        return

    # ---------------------------------------------------------------
    # Parse single nsys sqlite
    # ---------------------------------------------------------------
    if args.parse_nsys:
        parse_and_print(args.parse_nsys, args.label, output_dir)
        return

    # ---------------------------------------------------------------
    # Compare two profiles
    # ---------------------------------------------------------------
    if args.compare:
        if not args.nsys_a or not args.nsys_b:
            parser.error("--compare requires --nsys-a and --nsys-b")
        result_a = parse_and_print(args.nsys_a, args.label_a, output_dir)
        result_b = parse_and_print(args.nsys_b, args.label_b, output_dir)
        if result_a and result_b:
            compare_profiles(result_a, result_b)
        return

    # ---------------------------------------------------------------
    # Analyze scenario 1 from existing data
    # ---------------------------------------------------------------
    if args.analyze_scenario1:
        analyze_scenario1(args.tp2_sqlite, args.ep2_sqlite, output_dir)
        return

    # ---------------------------------------------------------------
    # Analyze scenario 2 from existing data
    # ---------------------------------------------------------------
    if args.analyze_scenario2:
        if not args.batch_sqlites:
            parser.error("--analyze-scenario2 requires --batch-sqlites")
        batch_map = {}
        for pair in args.batch_sqlites.split(","):
            batch_str, path = pair.split(":", 1)
            batch_map[int(batch_str)] = path
        analyze_scenario2_nsys(batch_map, output_dir)
        return

    # ---------------------------------------------------------------
    # Analyze scenario 4 from existing data
    # ---------------------------------------------------------------
    if args.analyze_scenario4:
        analyze_scenario4(args.seq_sqlite, args.par_sqlite, output_dir)
        return

    # ---------------------------------------------------------------
    # Client-side load generation
    # ---------------------------------------------------------------
    scenarios_to_run = []
    if args.run_all:
        scenarios_to_run = [1, 4, 2]  # priority order
    elif args.scenario:
        scenarios_to_run = [args.scenario]

    if not scenarios_to_run:
        parser.print_help()
        print("\nRun --print-commands to see server-side profiling instructions.")
        return

    async def run_scenarios():
        for s in scenarios_to_run:
            if s == 1:
                print_server_nsys_commands_scenario1()
                await run_scenario1_client(output_dir)
            elif s == 4:
                print_server_nsys_commands_scenario4()
                await run_scenario4_client(output_dir, pattern=args.pattern)
            elif s == 2:
                print_server_ncu_commands_scenario2()
                print_server_nsys_commands_scenario2()
                await run_scenario2_client(output_dir)

    asyncio.run(run_scenarios())


if __name__ == "__main__":
    main()
