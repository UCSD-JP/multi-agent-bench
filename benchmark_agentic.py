#!/usr/bin/env python3
"""
Unified multi-agent benchmark CLI.

Supports multiple framework runners via --framework flag:
  - autogen (default): SelectorGroupChat with dynamic LLM speaker selection
  - langgraph:         LangGraph StateGraph with fan-out parallel executors
  - a2a:               Google A2A protocol with parallel agent tasks

Example:
  python benchmark_agentic.py \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --dataset_path /path/to/sharegpt.json \
    --tasks 64 --concurrency 32 --executors 2

  python benchmark_agentic.py --framework langgraph \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --dataset_path /path/to/sharegpt.json \
    --tasks 3 --concurrency 1
"""

import asyncio
import os
import random
import statistics
import time
import argparse
from typing import Dict, List

import httpx

from benchmark.core.dataset import get_prompt_from_dataset, load_sharegpt_dataset
from benchmark.core.metrics import percentile
from benchmark.core.streaming_client import OpenAIStreamingClient
from benchmark.core.trace_writer import write_trace_jsonl
from benchmark.core.types import TaskRecord
from benchmark.runners.base import RunContext
from benchmark.runners.registry import available_frameworks, default_framework, get_runner


def print_summary(records: List[TaskRecord], total_s: float) -> None:
    """Print aggregated benchmark results."""
    task_throughput = len(records) / total_s if total_s > 0 else 0.0

    makespans = [r.makespan_ms for r in records]
    cps = [r.critical_path_ms for r in records]
    msg_counts = [r.messages_count for r in records]
    idle_waits = [r.total_idle_wait_ms for r in records]
    toks = [r.tokens_exchanged for r in records]
    bytes_x = [r.bytes_exchanged for r in records]

    per_role_lat: Dict[str, List[float]] = {}
    per_role_ttft: Dict[str, List[float]] = {}
    per_role_tpot: Dict[str, List[float]] = {}
    per_role_wait: Dict[str, List[float]] = {}

    total_steps = 0
    for tr in records:
        for s in tr.steps.values():
            total_steps += 1
            per_role_lat.setdefault(s.agent_role, []).append(s.latency_ms)
            per_role_wait.setdefault(s.agent_role, []).append(s.wait_ms)
            if s.ttft_ms is not None:
                per_role_ttft.setdefault(s.agent_role, []).append(s.ttft_ms)
            if s.tpot_ms is not None:
                per_role_tpot.setdefault(s.agent_role, []).append(s.tpot_ms)

    step_throughput = total_steps / total_s if total_s > 0 else 0.0

    def summarize(name: str, vals: List[float]) -> None:
        print(f"{name}: mean={statistics.mean(vals):.2f}  "
              f"p50={percentile(vals, 50):.2f}  "
              f"p95={percentile(vals, 95):.2f}  "
              f"p99={percentile(vals, 99):.2f}")

    framework = records[0].framework if records else "unknown"
    print(f"\n==== Multi-Agent Benchmark Results ({framework}) ====")
    print(f"Tasks: {len(records)}  Total time: {total_s:.2f}s  "
          f"Task throughput: {task_throughput:.2f} tasks/s")
    print(f"Total agent steps: {total_steps}  "
          f"Step throughput: {step_throughput:.2f} steps/s\n")

    summarize("E2E makespan (ms)", makespans)
    summarize("Critical path (ms)", cps)
    summarize("Total idle/sync wait per task (ms)", idle_waits)
    summarize("Messages per task", [float(x) for x in msg_counts])
    summarize("Tokens exchanged per task", [float(x) for x in toks])
    summarize("Bytes exchanged per task", [float(x) for x in bytes_x])

    # AutoGen-specific: selector overhead
    selector_overheads = [r.selector_overhead_ms for r in records if r.selector_overhead_ms > 0]
    if selector_overheads:
        summarize("Selector overhead (ms)", selector_overheads)

    print("\n-- Per-agent-role breakdown --")
    for role in sorted(per_role_lat.keys()):
        summarize(f"[{role}] latency (ms)", per_role_lat[role])
        if role in per_role_wait:
            summarize(f"[{role}] wait/idle (ms)", per_role_wait[role])
        if role in per_role_ttft and per_role_ttft[role]:
            summarize(f"[{role}] TTFT (ms)", per_role_ttft[role])
        if role in per_role_tpot and per_role_tpot[role]:
            summarize(f"[{role}] TPOT (ms)", per_role_tpot[role])
        print()


async def main():
    ap = argparse.ArgumentParser(description="Multi-framework agentic benchmark")
    default_fw = default_framework()
    ap.add_argument("--framework", type=str, default=default_fw,
                    choices=available_frameworks(),
                    help=f"Runner framework (default: {default_fw})")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--dataset_path", type=str, required=True,
                    help="REQUIRED: ShareGPT/ShareGPT4V JSON")
    ap.add_argument("--tasks", type=int, default=64,
                    help="Number of tasks (dataset prompts)")
    ap.add_argument("--concurrency", type=int, default=32,
                    help="LLM call concurrency (global)")
    ap.add_argument("--executors", type=int, default=2,
                    help="How many executor agents")
    ap.add_argument("--output_dir", type=str, default="results_multiagent")
    ap.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    ap.add_argument("--api_key", type=str, default="vllm-key")
    args = ap.parse_args()

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")

    dataset = load_sharegpt_dataset(args.dataset_path)
    if not dataset:
        raise ValueError(f"Loaded dataset is empty after filtering: {args.dataset_path}")

    random.shuffle(dataset)
    os.makedirs(args.output_dir, exist_ok=True)

    runner = get_runner(args.framework)
    print(f"[info] Using framework: {runner.name}")
    print(f"[info] Available frameworks: {', '.join(available_frameworks())}")

    llm = OpenAIStreamingClient(base_url=args.base_url, api_key=args.api_key)
    llm_semaphore = asyncio.Semaphore(args.concurrency)

    warmup_prompt = get_prompt_from_dataset(dataset, 0)

    async with httpx.AsyncClient() as http_client:
        ctx = RunContext(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            http_client=http_client,
            llm_semaphore=asyncio.Semaphore(1),  # warmup: serial
            streaming_client=llm,
            executors=args.executors,
        )

        print(f"[warmup] running 1 task (dataset-only)")
        _ = await runner.run_task(task_id=-1, prompt=warmup_prompt, context=ctx)
        await asyncio.sleep(1)

        # Real run
        ctx = RunContext(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            http_client=http_client,
            llm_semaphore=llm_semaphore,
            streaming_client=llm,
            executors=args.executors,
        )

        print(f"[run] framework={runner.name}  tasks={args.tasks}  "
              f"llm_concurrency={args.concurrency}  executors={args.executors}")
        t0 = time.time()

        task_futs = []
        for i in range(args.tasks):
            prompt = get_prompt_from_dataset(dataset, i)
            task_futs.append(
                asyncio.create_task(
                    runner.run_task(task_id=i, prompt=prompt, context=ctx)
                )
            )

        records: List[TaskRecord] = await asyncio.gather(*task_futs)
        t1 = time.time()

    total_s = t1 - t0
    print_summary(records, total_s)

    # Write trace JSONL
    suffix = f"_{runner.name}"
    out_path = write_trace_jsonl(records, args.output_dir, suffix=suffix)
    print(f"\nSaved task traces to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
