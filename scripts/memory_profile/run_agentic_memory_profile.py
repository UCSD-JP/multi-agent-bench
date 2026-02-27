#!/usr/bin/env python3
"""Agentic multi-turn memory profile experiment.

Simulates multi-turn agent sessions where context grows each turn
(instruction → model response → tool output → model response → ...),
while monitoring vLLM KV cache usage via /metrics.

Measures:
- Per-turn prompt_tokens / completion_tokens from API
- vLLM gpu_cache_usage_perc over time per concurrency level
- When preemptions start (cache pressure)
- Context growth pattern

Concurrency levels are run sequentially; within each level,
N agent sessions run in parallel (each session is multi-turn sequential).
"""

import argparse
import asyncio
import csv
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from vllm_metrics_sampler import VllmMetricsSampler

# ---------------------------------------------------------------------------
# Padding generation (English words, ~8.2 chars/token for Qwen3-Next)
# ---------------------------------------------------------------------------
_PADDING_WORDS = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
    "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
    "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
    "victor", "whiskey", "xray", "yankee", "zulu", "class", "function",
    "return", "import", "module", "config", "server", "client", "request",
    "response", "handler", "parser", "builder", "factory", "adapter",
    "bridge", "proxy", "logger", "cache", "queue",
]

# Realistic code snippets to simulate file content reads
_CODE_TEMPLATES = [
    "def {fn}(self, data):\n    result = self.process(data)\n    if not result:\n        raise ValueError('Invalid input')\n    return result\n",
    "class {cls}Handler:\n    def __init__(self, config):\n        self.config = config\n        self.logger = logging.getLogger(__name__)\n\n    def execute(self, request):\n        self.logger.info(f'Processing {{request.id}}')\n        return self._handle(request)\n",
    "async def fetch_{fn}(session, url, params=None):\n    async with session.get(url, params=params) as resp:\n        if resp.status != 200:\n            raise HTTPError(f'Status {{resp.status}}')\n        return await resp.json()\n",
    "# Configuration for {cls}\n{cls}_CONFIG = {{\n    'max_retries': 3,\n    'timeout_s': 30,\n    'batch_size': 64,\n    'enable_cache': True,\n}}\n",
]

_FN_NAMES = ["process_data", "validate_input", "transform_output", "handle_error",
             "parse_config", "build_query", "execute_task", "run_pipeline"]
_CLS_NAMES = ["Request", "Response", "Pipeline", "Worker", "Manager", "Scheduler",
              "Dispatcher", "Collector", "Analyzer", "Reporter"]


def _generate_code_block(target_tokens: int) -> str:
    """Generate realistic-looking code content of ~target_tokens tokens."""
    chars_needed = int(target_tokens * 7.0)  # Conservative estimate
    parts = []
    total_chars = 0
    file_idx = 0

    while total_chars < chars_needed:
        file_idx += 1
        fn = random.choice(_FN_NAMES)
        cls = random.choice(_CLS_NAMES)
        template = random.choice(_CODE_TEMPLATES)
        block = f"# File: src/module/file_{file_idx}.py\n"
        block += template.format(fn=fn, cls=cls)
        block += "\n"
        # Pad with word sequences to fill space
        words_needed = max(0, (chars_needed - total_chars - len(block)) // 6)
        if words_needed > 0:
            word_line = " ".join(random.choices(_PADDING_WORDS, k=min(words_needed, 200)))
            block += f"# Output: {word_line}\n\n"

        parts.append(block)
        total_chars += len(block)

    return "".join(parts)[:chars_needed]


def _generate_tool_output(turn: int, base_tokens: int = 2000, growth: int = 1500) -> str:
    """Generate simulated tool output for a given turn.

    Context grows each turn: base_tokens + turn * growth tokens.
    Simulates reading files, running commands, seeing test output.
    """
    target = base_tokens + turn * growth
    header = f"[Tool Result - Turn {turn + 1}]\n"
    if turn % 3 == 0:
        header += "$ cat src/module/handler.py\n"
    elif turn % 3 == 1:
        header += "$ python -m pytest tests/ -v\n"
    else:
        header += "$ grep -r 'TODO' src/\n"

    body = _generate_code_block(target)
    return header + body


# ---------------------------------------------------------------------------
# Benchmark instruction templates
# ---------------------------------------------------------------------------
_SWEBENCH_TEMPLATE = """You are a software engineer. Fix the following bug.

## Issue
The `{cls}` class in `src/module/handler.py` raises a `TypeError` when processing
requests with empty payload. The error occurs on line 42 where `self.data.items()`
is called but `self.data` is None.

## Repository Structure
```
src/
  module/
    handler.py
    utils.py
    tests/
      test_handler.py
```

## Expected Behavior
When payload is empty, the handler should return an empty response with status 204.

## Steps to Reproduce
1. Send POST /api/process with empty body
2. Observe TypeError in logs

Please fix the bug and ensure all tests pass.
"""

_TERMINALBENCH_TEMPLATE = """Your task is to set up a {service} service on this machine.

Requirements:
- Install {service} if not already available
- Configure it to listen on port {port}
- Set up authentication with a secure password
- Create a systemd service file
- Verify the service starts correctly
- Write a health check script at /app/healthcheck.sh
"""

_LIVECODEBENCH_TEMPLATE = """You are an expert Python programmer. Solve the following problem.

### Problem
Given an array of integers `nums` and a target sum `k`, find all unique pairs
of elements that sum to `k`. Return the pairs sorted in ascending order.

### Constraints
- 1 <= len(nums) <= 10^5
- -10^9 <= nums[i] <= 10^9
- -10^9 <= k <= 10^9

### Examples
Input: nums = [1, 2, 3, 4, 5], k = 6
Output: [[1, 5], [2, 4]]

### Solution
```python
"""


def generate_instruction(benchmark: str = "swebench") -> str:
    """Generate a realistic benchmark-style instruction."""
    if benchmark == "swebench":
        cls = random.choice(_CLS_NAMES)
        return _SWEBENCH_TEMPLATE.format(cls=cls)
    elif benchmark == "terminalbench":
        services = ["nginx", "redis", "postgresql", "elasticsearch", "grafana"]
        svc = random.choice(services)
        port = random.choice([5432, 6379, 8080, 9200, 3000])
        return _TERMINALBENCH_TEMPLATE.format(service=svc, port=port)
    else:  # livecodebench
        return _LIVECODEBENCH_TEMPLATE


# Benchmark-specific context growth profiles (realistic)
# Based on actual benchmark agent behavior patterns:
#   SWE-bench: reads large source files (3-10K per file), runs test suites (2-5K output)
#   Terminal Bench: shell commands with shorter output (0.5-2K per command)
#   LiveCodeBench: compile/test output, mostly short (300-800 per turn)
BENCHMARK_PROFILES = {
    "swebench": {
        "tool_base_tokens": 4000,   # First file read ~4K
        "tool_growth_tokens": 2500, # Each subsequent read adds more (larger files, test output)
        "max_turns": 7,             # Typical: read → plan → edit → test → fix → test → done
        "max_tokens_per_turn": 512, # Agent writes patches, longer output
    },
    "terminalbench": {
        "tool_base_tokens": 800,    # First command output ~800 tokens
        "tool_growth_tokens": 600,  # Incremental growth (config files, logs)
        "max_turns": 10,            # More turns but shorter context each
        "max_tokens_per_turn": 256, # Short commands
    },
    "livecodebench": {
        "tool_base_tokens": 400,    # Compile output / test result ~400 tokens
        "tool_growth_tokens": 300,  # Small growth (error messages, test output)
        "max_turns": 5,             # Few turns: write code → test → fix → test → done
        "max_tokens_per_turn": 512, # Code generation can be longer
    },
}


def get_benchmark_profile(benchmark: str, args) -> dict:
    """Get benchmark-specific profile, with CLI args as override."""
    profile = BENCHMARK_PROFILES.get(benchmark, BENCHMARK_PROFILES["swebench"]).copy()
    # CLI args override if explicitly set (non-default)
    if args.tool_base_tokens != 2000:
        profile["tool_base_tokens"] = args.tool_base_tokens
    if args.tool_growth_tokens != 1500:
        profile["tool_growth_tokens"] = args.tool_growth_tokens
    if args.max_turns != 7:
        profile["max_turns"] = args.max_turns
    if args.max_tokens_per_turn != 256:
        profile["max_tokens_per_turn"] = args.max_tokens_per_turn
    return profile


# ---------------------------------------------------------------------------
# Agent session runner
# ---------------------------------------------------------------------------
async def run_agent_session(
    session_id: int,
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    max_turns: int = 7,
    max_tokens_per_turn: int = 256,
    max_context: int = 30000,
    benchmark: str = "swebench",
    tool_base_tokens: int = 2000,
    tool_growth_tokens: int = 1500,
) -> list[dict]:
    """Run one multi-turn agent session. Returns per-turn records."""

    messages = [
        {"role": "system", "content": "You are an expert coding assistant. Execute the task step by step."},
    ]
    instruction = generate_instruction(benchmark)
    messages.append({"role": "user", "content": instruction})

    records = []
    for turn in range(max_turns):
        start_ts = time.time()
        try:
            resp = await client.post(
                f"{base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens_per_turn,
                    "temperature": 0.7,
                },
                timeout=300,
            )
            end_ts = time.time()

            if resp.status_code != 200:
                records.append({
                    "session_id": session_id,
                    "turn": turn,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "latency_s": end_ts - start_ts,
                    "status": f"error:http_{resp.status_code}",
                    "cumulative_context_tokens": 0,
                })
                break

            data = resp.json()
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            assistant_msg = data["choices"][0]["message"]["content"]
            messages.append({"role": "assistant", "content": assistant_msg})

            records.append({
                "session_id": session_id,
                "turn": turn,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_s": round(end_ts - start_ts, 3),
                "status": "ok",
                "cumulative_context_tokens": prompt_tokens + completion_tokens,
            })

            # Check if approaching context limit
            if prompt_tokens + completion_tokens + tool_base_tokens > max_context:
                break

            # Simulate tool output (growing each turn)
            tool_output = _generate_tool_output(turn, tool_base_tokens, tool_growth_tokens)
            messages.append({
                "role": "user",
                "content": f"{tool_output}\n\nAnalyze the output and continue working on the task.",
            })

        except Exception as e:
            end_ts = time.time()
            records.append({
                "session_id": session_id,
                "turn": turn,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_s": round(end_ts - start_ts, 3),
                "status": f"error:{type(e).__name__}",
                "cumulative_context_tokens": 0,
            })
            break

    return records


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
async def run_experiment(args):
    """Run agentic memory profile experiment across concurrency levels."""

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = args.base_url.rstrip("/")

    concurrencies = [int(x) for x in args.concurrencies.split(",")]
    benchmarks = args.benchmarks.split(",")

    print(f"Agentic Memory Profile Experiment (benchmark-specific profiles)")
    print(f"  Server:        {base_url}")
    print(f"  Model:         {args.model}")
    print(f"  Concurrencies: {concurrencies}")
    print(f"  Benchmarks:    {benchmarks}")
    print(f"  Max context:   {args.max_context}")
    print(f"  Output:        {out_dir}/")
    for bm in benchmarks:
        p = get_benchmark_profile(bm, args)
        print(f"  [{bm}] tool_base={p['tool_base_tokens']}, growth={p['tool_growth_tokens']}, "
              f"max_turns={p['max_turns']}, max_tok/turn={p['max_tokens_per_turn']}")
    print()

    all_turn_records = []
    concurrency_summaries = []

    for benchmark in benchmarks:
        print(f"{'='*60}")
        print(f"Benchmark: {benchmark}")
        print(f"{'='*60}")

        profile = get_benchmark_profile(benchmark, args)

        for conc in concurrencies:
            print(f"\n--- Concurrency={conc} (tool_base={profile['tool_base_tokens']}, "
                  f"growth={profile['tool_growth_tokens']}, turns={profile['max_turns']}) ---")

            # Start metrics sampler
            metrics_path = out_dir / f"metrics_{benchmark}_c{conc}.csv"
            sampler = VllmMetricsSampler(
                base_url=base_url,
                output_path=metrics_path,
                interval_s=0.3,
            )
            sampler.start()

            # Brief warmup pause
            await asyncio.sleep(1.0)

            # Run concurrent agent sessions
            async with httpx.AsyncClient() as client:
                tasks = [
                    run_agent_session(
                        session_id=s,
                        client=client,
                        base_url=base_url,
                        model=args.model,
                        max_turns=profile["max_turns"],
                        max_tokens_per_turn=profile["max_tokens_per_turn"],
                        max_context=args.max_context,
                        benchmark=benchmark,
                        tool_base_tokens=profile["tool_base_tokens"],
                        tool_growth_tokens=profile["tool_growth_tokens"],
                    )
                    for s in range(conc)
                ]
                session_results = await asyncio.gather(*tasks)

            # Brief cooldown
            await asyncio.sleep(2.0)
            sampler.stop()

            # Collect turn records
            for session_records in session_results:
                for r in session_records:
                    r["benchmark"] = benchmark
                    r["concurrency"] = conc
                    all_turn_records.append(r)

            # Summary for this concurrency level
            all_turns = [r for sess in session_results for r in sess]
            ok_turns = [r for r in all_turns if r["status"] == "ok"]
            max_ctx = max((r["cumulative_context_tokens"] for r in ok_turns), default=0)
            total_turns = sum(len(sess) for sess in session_results)
            peak_cache = sampler.peak_cache_usage()

            summary = {
                "benchmark": benchmark,
                "concurrency": conc,
                "sessions": conc,
                "tool_base_tokens": profile["tool_base_tokens"],
                "tool_growth_tokens": profile["tool_growth_tokens"],
                "total_turns": total_turns,
                "ok_turns": len(ok_turns),
                "max_context_tokens": max_ctx,
                "peak_cache_usage_pct": round(peak_cache * 100, 2),
                "avg_prompt_tokens": round(
                    sum(r["prompt_tokens"] for r in ok_turns) / max(len(ok_turns), 1)
                ),
                "avg_completion_tokens": round(
                    sum(r["completion_tokens"] for r in ok_turns) / max(len(ok_turns), 1)
                ),
                "avg_latency_s": round(
                    sum(r["latency_s"] for r in ok_turns) / max(len(ok_turns), 1), 3
                ),
            }
            concurrency_summaries.append(summary)

            print(f"  Sessions: {conc}, Turns: {total_turns} ({len(ok_turns)} ok)")
            print(f"  Max context: {max_ctx:,} tokens")
            print(f"  Peak cache:  {peak_cache:.1%}")
            print(f"  Avg latency: {summary['avg_latency_s']:.3f}s/turn")

            # Stabilization pause between concurrency levels
            await asyncio.sleep(3.0)

    # Write turn trace CSV
    turn_csv = out_dir / "turn_trace.csv"
    if all_turn_records:
        fieldnames = [
            "benchmark", "concurrency", "session_id", "turn",
            "prompt_tokens", "completion_tokens", "cumulative_context_tokens",
            "latency_s", "status",
        ]
        with open(turn_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_turn_records)
        print(f"\nTurn trace: {turn_csv} ({len(all_turn_records)} rows)")

    # Write summary JSON
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(concurrency_summaries, f, indent=2)
    print(f"Summary: {summary_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Benchmark':<16} {'Conc':>5} {'Turns':>6} {'MaxCtx':>8} {'PeakCache':>10} {'AvgLat':>8}")
    print(f"{'-'*80}")
    for s in concurrency_summaries:
        print(
            f"{s['benchmark']:<16} {s['concurrency']:>5} "
            f"{s['total_turns']:>6} {s['max_context_tokens']:>8,} "
            f"{s['peak_cache_usage_pct']:>9.1f}% {s['avg_latency_s']:>7.3f}s"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Agentic multi-turn memory profile experiment"
    )
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000/v1",
        help="vLLM API base URL",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-Next-80B-A3B-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--concurrencies", default="1,2,4,8,16",
        help="Comma-separated concurrency levels",
    )
    parser.add_argument(
        "--benchmarks", default="swebench,terminalbench,livecodebench",
        help="Comma-separated benchmark types to simulate",
    )
    parser.add_argument(
        "--max-turns", type=int, default=7,
        help="Max turns per agent session (default: 7)",
    )
    parser.add_argument(
        "--max-tokens-per-turn", type=int, default=256,
        help="Max output tokens per turn (default: 256)",
    )
    parser.add_argument(
        "--max-context", type=int, default=30000,
        help="Max context tokens before stopping session (default: 30000)",
    )
    parser.add_argument(
        "--tool-base-tokens", type=int, default=2000,
        help="Base tool output tokens at turn 0 (default: 2000)",
    )
    parser.add_argument(
        "--tool-growth-tokens", type=int, default=1500,
        help="Additional tokens per subsequent turn (default: 1500)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/memory_profile/agentic",
        help="Output directory",
    )
    args = parser.parse_args()
    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()
