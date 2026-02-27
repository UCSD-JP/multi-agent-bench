#!/usr/bin/env python3
"""Analyze concurrency sweep results from trace JSONL files.

Usage:
    python scripts/analyze_conc_sweep.py [--sweep_dir results_multiagent/conc_sweep]
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    d = k - f
    return s[f] + d * (s[c] - s[f])


def load_trace(path: str) -> List[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def analyze_level(records: List[dict]) -> dict:
    """Compute summary stats for one concurrency level."""
    all_tpot = []
    all_ttft = []
    all_latency = []
    role_latency: Dict[str, List[float]] = {}
    role_prompt_tokens: Dict[str, List[int]] = {}
    role_completion_tokens: Dict[str, List[int]] = {}

    total_tokens = 0
    makespans = []
    crit_paths = []

    for r in records:
        makespans.append(r["makespan_ms"])
        crit_paths.append(r["critical_path_ms"])
        total_tokens += r.get("tokens_exchanged", 0)

        for sid, step in r.get("steps", {}).items():
            role = step.get("agent_role", "unknown")
            lat = step.get("latency_ms", 0)
            all_latency.append(lat)
            role_latency.setdefault(role, []).append(lat)

            pt = step.get("prompt_tokens", 0)
            ct = step.get("completion_tokens", 0)
            role_prompt_tokens.setdefault(role, []).append(pt)
            role_completion_tokens.setdefault(role, []).append(ct)

            if step.get("tpot_ms") is not None:
                all_tpot.append(step["tpot_ms"])
            if step.get("ttft_ms") is not None:
                all_ttft.append(step["ttft_ms"])

    result = {
        "tasks": len(records),
        "total_tokens": total_tokens,
    }

    for name, vals in [("tpot", all_tpot), ("ttft", all_ttft),
                       ("makespan", makespans), ("crit_path", crit_paths)]:
        if vals:
            result[f"{name}_mean"] = statistics.mean(vals)
            result[f"{name}_p50"] = percentile(vals, 50)
            result[f"{name}_p95"] = percentile(vals, 95)
            result[f"{name}_p99"] = percentile(vals, 99)

    result["roles"] = {}
    for role in sorted(role_latency.keys()):
        rd = {
            "latency_mean": statistics.mean(role_latency[role]),
            "latency_p50": percentile(role_latency[role], 50),
            "latency_p95": percentile(role_latency[role], 95),
        }
        pt = role_prompt_tokens.get(role, [])
        ct = role_completion_tokens.get(role, [])
        if pt:
            rd["prompt_tokens_mean"] = statistics.mean(pt)
        if ct:
            rd["completion_tokens_mean"] = statistics.mean(ct)
        if pt and ct:
            mean_pt = statistics.mean(pt)
            mean_ct = statistics.mean(ct)
            rd["pd_ratio"] = mean_pt / mean_ct if mean_ct > 0 else float("inf")
        result["roles"][role] = rd

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", default="results_multiagent/conc_sweep",
                    help="Directory containing per-concurrency subdirs")
    ap.add_argument("--json_out", default=None,
                    help="Optional: save structured results to JSON")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"ERROR: {sweep_dir} does not exist")
        sys.exit(1)

    # Find all concurrency level dirs
    levels: List[Tuple[int, str]] = []
    for d in sorted(sweep_dir.iterdir()):
        if d.is_dir() and "_c" in d.name:
            try:
                conc = int(d.name.split("_c")[-1])
            except ValueError:
                continue
            # Find trace file
            traces = list(d.glob("trace_*.jsonl"))
            if traces:
                levels.append((conc, str(traces[0])))

    if not levels:
        print(f"No sweep data found in {sweep_dir}")
        sys.exit(1)

    levels.sort(key=lambda x: x[0])

    print(f"\n{'='*80}")
    print(f"Concurrency Sweep Analysis")
    print(f"{'='*80}")
    print(f"Levels found: {[c for c, _ in levels]}")
    print()

    all_results = {}
    for conc, path in levels:
        records = load_trace(path)
        result = analyze_level(records)
        all_results[conc] = result

    # Main comparison table
    header_concs = [f"c={c}" for c, _ in levels]
    print(f"| Metric | " + " | ".join(header_concs) + " |")
    print(f"|--------|" + "|".join(["-------" for _ in levels]) + "|")

    def row(name, key, fmt=".1f"):
        vals = []
        for c, _ in levels:
            v = all_results[c].get(key)
            vals.append(f"{v:{fmt}}" if v is not None else "—")
        print(f"| {name} | " + " | ".join(vals) + " |")

    row("TPOT mean (ms)", "tpot_mean")
    row("TPOT p50 (ms)", "tpot_p50")
    row("TPOT p95 (ms)", "tpot_p95")
    row("TTFT mean (ms)", "ttft_mean")
    row("TTFT p50 (ms)", "ttft_p50")
    row("TTFT p95 (ms)", "ttft_p95")
    row("Makespan mean (ms)", "makespan_mean", ".0f")
    row("Makespan p50 (ms)", "makespan_p50", ".0f")
    row("Total tokens", "total_tokens", ".0f")

    # Degradation table
    base_conc = levels[0][0]
    print(f"\n| Degradation (vs c={base_conc}) |", end="")
    print(" | ".join(header_concs[1:]) + " |")
    print(f"|--------|" + "|".join(["-------" for _ in levels[1:]]) + "|")

    for name, key in [("TPOT mean", "tpot_mean"), ("TPOT p95", "tpot_p95"),
                      ("TTFT mean", "ttft_mean"), ("Makespan mean", "makespan_mean")]:
        base_val = all_results[base_conc].get(key, 0)
        vals = []
        for c, _ in levels[1:]:
            v = all_results[c].get(key, 0)
            ratio = v / base_val if base_val > 0 else 0
            vals.append(f"{ratio:.2f}x")
        print(f"| {name} | " + " | ".join(vals) + " |")

    # Per-role table
    all_roles = set()
    for c, _ in levels:
        all_roles.update(all_results[c].get("roles", {}).keys())

    print(f"\n| Role Latency (ms) | " + " | ".join(header_concs) + " |")
    print(f"|--------|" + "|".join(["-------" for _ in levels]) + "|")
    for role in sorted(all_roles):
        vals = []
        for c, _ in levels:
            rd = all_results[c].get("roles", {}).get(role, {})
            v = rd.get("latency_mean")
            vals.append(f"{v:.0f}" if v is not None else "—")
        print(f"| {role} | " + " | ".join(vals) + " |")

    # P:D ratio
    print(f"\n| P:D Ratio | " + " | ".join(header_concs) + " |")
    print(f"|--------|" + "|".join(["-------" for _ in levels]) + "|")
    for role in sorted(all_roles):
        vals = []
        for c, _ in levels:
            rd = all_results[c].get("roles", {}).get(role, {})
            v = rd.get("pd_ratio")
            vals.append(f"{v:.2f}" if v is not None else "—")
        print(f"| {role} | " + " | ".join(vals) + " |")

    # Optional JSON output
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved structured results to: {args.json_out}")


if __name__ == "__main__":
    main()
