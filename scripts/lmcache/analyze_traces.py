#!/usr/bin/env python3
"""
Analyze LMCache agent traces for workload characterization.

Reads JSONL traces from lmcache-agent-trace repo and extracts:
1. Token distribution per step (input/output)
2. Context growth rate across turns
3. Prefix overlap ratio between consecutive steps
4. Session length distribution
5. System prompt size estimation

This data is used to calibrate our synthetic benchmark generator
(benchmark_prefix_workload.py) with realistic parameters.

Usage:
  # Clone traces first:
  git clone https://github.com/LMCache/lmcache-agent-trace.git /tmp/lmcache-traces

  # Analyze all agents:
  python analyze_traces.py --trace_dir /tmp/lmcache-traces

  # Specific agent:
  python analyze_traces.py --trace_dir /tmp/lmcache-traces --agent miniswe
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class StepInfo:
    step_idx: int
    input_tokens: int
    output_tokens: int
    input_len_chars: int = 0
    timestamp: str = ""


def load_jsonl(fpath: str) -> List[dict]:
    """Load a JSONL file."""
    records = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def analyze_trace_file(fpath: str) -> Dict:
    """Analyze a single trace file."""
    records = load_jsonl(fpath)
    if not records:
        return {}

    steps = []
    for i, rec in enumerate(records):
        input_tokens = rec.get("input_tokens", 0)
        output_tokens = rec.get("output_tokens", 0)

        # Some formats store tokens differently
        if input_tokens == 0 and "usage" in rec:
            usage = rec["usage"] if isinstance(rec["usage"], dict) else {}
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

        input_text = rec.get("input", "")
        if isinstance(input_text, list):
            # OpenAI messages format
            input_text = json.dumps(input_text)

        steps.append(StepInfo(
            step_idx=i,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_len_chars=len(str(input_text)),
            timestamp=rec.get("timestamp", ""),
        ))

    if not steps:
        return {}

    input_tokens = [s.input_tokens for s in steps]
    output_tokens = [s.output_tokens for s in steps]

    # Context growth: how much input grows per step
    growth = []
    for i in range(1, len(steps)):
        if steps[i].input_tokens > 0 and steps[i - 1].input_tokens > 0:
            growth.append(steps[i].input_tokens - steps[i - 1].input_tokens)

    # Prefix overlap ratio (approximate via char-level comparison)
    prefix_overlaps = []
    for i in range(1, len(steps)):
        prev_chars = steps[i - 1].input_len_chars
        curr_chars = steps[i].input_len_chars
        if prev_chars > 0 and curr_chars > 0:
            # Approximate: if context grows monotonically, overlap = prev_len / curr_len
            overlap = min(prev_chars, curr_chars) / max(prev_chars, curr_chars)
            prefix_overlaps.append(overlap)

    return {
        "file": os.path.basename(fpath),
        "num_steps": len(steps),
        "total_input_tokens": sum(input_tokens),
        "total_output_tokens": sum(output_tokens),
        "input_tokens_per_step": {
            "mean": sum(input_tokens) / len(input_tokens) if input_tokens else 0,
            "min": min(input_tokens) if input_tokens else 0,
            "max": max(input_tokens) if input_tokens else 0,
            "first": input_tokens[0] if input_tokens else 0,
            "last": input_tokens[-1] if input_tokens else 0,
        },
        "output_tokens_per_step": {
            "mean": sum(output_tokens) / len(output_tokens) if output_tokens else 0,
            "min": min(output_tokens) if output_tokens else 0,
            "max": max(output_tokens) if output_tokens else 0,
        },
        "context_growth_per_step": {
            "mean": sum(growth) / len(growth) if growth else 0,
            "min": min(growth) if growth else 0,
            "max": max(growth) if growth else 0,
        },
        "prefix_overlap_ratio": {
            "mean": sum(prefix_overlaps) / len(prefix_overlaps) if prefix_overlaps else 0,
        },
        "io_ratio": (
            sum(input_tokens) / sum(output_tokens)
            if sum(output_tokens) > 0 else float("inf")
        ),
    }


def analyze_agent(trace_dir: str, agent_name: str) -> List[Dict]:
    """Analyze all traces for a given agent."""
    agent_dir = os.path.join(trace_dir, agent_name)
    if not os.path.isdir(agent_dir):
        return []

    results = []
    for fpath in sorted(glob.glob(os.path.join(agent_dir, "*.jsonl"))):
        analysis = analyze_trace_file(fpath)
        if analysis:
            results.append(analysis)
    return results


def aggregate_agent_stats(agent_results: List[Dict]) -> Dict:
    """Aggregate stats across all traces for an agent."""
    if not agent_results:
        return {}

    num_traces = len(agent_results)
    steps_per_trace = [r["num_steps"] for r in agent_results]
    total_input = [r["total_input_tokens"] for r in agent_results]
    total_output = [r["total_output_tokens"] for r in agent_results]
    growth_means = [r["context_growth_per_step"]["mean"] for r in agent_results if r["context_growth_per_step"]["mean"] != 0]
    overlap_means = [r["prefix_overlap_ratio"]["mean"] for r in agent_results if r["prefix_overlap_ratio"]["mean"] > 0]
    io_ratios = [r["io_ratio"] for r in agent_results if r["io_ratio"] != float("inf")]

    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0

    return {
        "num_traces": num_traces,
        "steps_per_trace": {
            "mean": safe_mean(steps_per_trace),
            "min": min(steps_per_trace),
            "max": max(steps_per_trace),
        },
        "tokens_per_trace": {
            "input_mean": safe_mean(total_input),
            "output_mean": safe_mean(total_output),
        },
        "context_growth_tokens_per_step": safe_mean(growth_means),
        "prefix_overlap_ratio_mean": safe_mean(overlap_means),
        "io_ratio_mean": safe_mean(io_ratios),
    }


def print_agent_report(agent_name: str, agent_results: List[Dict], agg: Dict):
    """Print analysis report for an agent."""
    print(f"\n{'='*60}")
    print(f"  Agent: {agent_name} ({agg.get('num_traces', 0)} traces)")
    print(f"{'='*60}")

    if not agg:
        print("  No data.")
        return

    sp = agg["steps_per_trace"]
    print(f"  Steps/trace:    mean={sp['mean']:.1f}  min={sp['min']}  max={sp['max']}")

    tp = agg["tokens_per_trace"]
    print(f"  Tokens/trace:   input={tp['input_mean']:.0f}  output={tp['output_mean']:.0f}")

    growth = agg["context_growth_tokens_per_step"]
    print(f"  Context growth: {growth:.0f} tokens/step")

    overlap = agg["prefix_overlap_ratio_mean"]
    print(f"  Prefix overlap: {overlap:.3f}")

    io = agg["io_ratio_mean"]
    print(f"  I/O ratio:      {io:.1f}:1")

    # Calibration recommendations
    print(f"\n  Benchmark calibration recommendations:")
    print(f"    --num_turns {int(sp['mean'])}")
    print(f"    --context_growth_tokens {max(500, int(growth))}")
    print(f"    --system_prompt_tokens ~{int(agent_results[0]['input_tokens_per_step']['first']) if agent_results else 2000}")


def main():
    parser = argparse.ArgumentParser(description="Analyze LMCache agent traces")
    parser.add_argument("--trace_dir", required=True, help="Path to lmcache-agent-trace repo")
    parser.add_argument("--agent", default=None, help="Specific agent to analyze (e.g., miniswe, claudecode)")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    args = parser.parse_args()

    if not os.path.isdir(args.trace_dir):
        print(f"ERROR: Directory not found: {args.trace_dir}")
        sys.exit(1)

    # Discover agents
    agents = []
    if args.agent:
        agents = [args.agent]
    else:
        for d in sorted(os.listdir(args.trace_dir)):
            agent_dir = os.path.join(args.trace_dir, d)
            if os.path.isdir(agent_dir) and not d.startswith(".") and d not in ("visualizer",):
                # Check if it has JSONL files
                if glob.glob(os.path.join(agent_dir, "*.jsonl")):
                    agents.append(d)

    if not agents:
        print(f"No agent trace directories found in {args.trace_dir}")
        sys.exit(1)

    all_reports = {}
    for agent in agents:
        agent_results = analyze_agent(args.trace_dir, agent)
        agg = aggregate_agent_stats(agent_results)
        all_reports[agent] = {
            "aggregate": agg,
            "traces": agent_results,
        }
        if not args.json:
            print_agent_report(agent, agent_results, agg)

    if args.json:
        print(json.dumps(all_reports, indent=2, default=str))

    # Summary comparison
    if not args.json and len(all_reports) > 1:
        print(f"\n{'='*80}")
        print("Cross-Agent Comparison")
        print(f"{'='*80}")
        print(f"{'Agent':<15} {'Traces':>7} {'Steps':>7} {'Growth':>8} {'Overlap':>8} {'I/O':>8}")
        print("-" * 55)
        for agent, report in sorted(all_reports.items()):
            agg = report["aggregate"]
            if not agg:
                continue
            print(f"{agent:<15} {agg['num_traces']:>7} "
                  f"{agg['steps_per_trace']['mean']:>6.1f} "
                  f"{agg['context_growth_tokens_per_step']:>7.0f} "
                  f"{agg['prefix_overlap_ratio_mean']:>7.3f} "
                  f"{agg['io_ratio_mean']:>7.1f}")


if __name__ == "__main__":
    main()
