#!/usr/bin/env python3
"""
Analyze KV cache / LMCache experiment results.

Reads result JSON files from run_experiment.sh and produces:
1. Per-mode, per-concurrency TTFT/TPOT/TPS comparison
2. Per-turn TTFT breakdown (shows prefix cache warming effect)
3. Server metrics comparison (prefix hit rate, preemptions)
4. CSV export for plotting

Usage:
  python analyze_results.py --result_dir /path/to/results
  python analyze_results.py --result_dir /path/to/results --csv
  python analyze_results.py --result_dir /path/to/results --summary_only
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict


def load_results(result_dir: str) -> dict:
    """Load all result JSON files from experiment directory."""
    results = {}
    pattern = os.path.join(result_dir, "**/results_*.json")
    for fpath in sorted(glob.glob(pattern, recursive=True)):
        try:
            with open(fpath) as f:
                data = json.load(f)
            summary = data.get("summary", {})
            config = summary.get("config", {})
            mode = config.get("mode", "unknown")
            conc = config.get("concurrency", 0)
            key = (mode, conc)
            results[key] = {
                "file": fpath,
                "summary": summary,
                "requests": data.get("requests", []),
            }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [WARN] Skipping {fpath}: {e}", file=sys.stderr)
    return results


def print_summary_table(results: dict):
    """Print comparison table across modes and concurrencies."""
    if not results:
        print("No results found.")
        return

    # Group by mode
    modes = sorted(set(k[0] for k in results.keys()))
    concs = sorted(set(k[1] for k in results.keys()))

    # Header
    print(f"\n{'Mode':<20} {'Conc':>5} {'TTFT_mean':>10} {'TTFT_p50':>10} {'TTFT_p99':>10} "
          f"{'TPOT_mean':>10} {'TPS':>8} {'PfxHit':>8} {'Preempt':>8}")
    print("-" * 100)

    for mode in modes:
        for conc in concs:
            key = (mode, conc)
            if key not in results:
                continue
            s = results[key]["summary"]

            ttft = s.get("ttft_ms", {})
            tpot = s.get("tpot_ms", {})
            tps = s.get("throughput_tps", 0)
            sm = s.get("server_metrics", {})

            ttft_mean = f"{ttft.get('mean', 0):.1f}" if ttft else "-"
            ttft_p50 = f"{ttft.get('p50', 0):.1f}" if ttft else "-"
            ttft_p99 = f"{ttft.get('p99', 0):.1f}" if ttft else "-"
            tpot_mean = f"{tpot.get('mean', 0):.1f}" if tpot else "-"
            tps_str = f"{tps:.1f}" if tps else "-"
            pfx_hit = f"{sm.get('prefix_cache_hit_rate', -1):.3f}" if sm.get("prefix_cache_hit_rate", -1) >= 0 else "-"
            preempt = str(sm.get("preemptions_delta", -1)) if sm.get("preemptions_delta", -1) >= 0 else "-"

            print(f"{mode:<20} {conc:>5} {ttft_mean:>10} {ttft_p50:>10} {ttft_p99:>10} "
                  f"{tpot_mean:>10} {tps_str:>8} {pfx_hit:>8} {preempt:>8}")


def print_turn_breakdown(results: dict):
    """Print per-turn TTFT breakdown for each mode/concurrency."""
    print(f"\n{'='*80}")
    print("Per-Turn TTFT Breakdown (prefix cache warming effect)")
    print(f"{'='*80}")

    modes = sorted(set(k[0] for k in results.keys()))
    concs = sorted(set(k[1] for k in results.keys()))

    for conc in concs:
        print(f"\n--- Concurrency = {conc} ---")
        header = f"{'Turn':<8}"
        for mode in modes:
            if (mode, conc) in results:
                header += f" {mode:>15}"
        print(header)
        print("-" * (8 + 16 * len(modes)))

        # Find max turns
        max_turns = 0
        for mode in modes:
            key = (mode, conc)
            if key in results:
                per_turn = results[key]["summary"].get("per_turn", {})
                max_turns = max(max_turns, len(per_turn))

        for t in range(max_turns):
            row = f"turn_{t:<3}"
            for mode in modes:
                key = (mode, conc)
                if key in results:
                    per_turn = results[key]["summary"].get("per_turn", {})
                    tv = per_turn.get(f"turn_{t}", {})
                    ttft = tv.get("ttft_mean_ms", 0)
                    row += f" {ttft:>14.1f}ms" if ttft > 0 else f" {'---':>15}"
                else:
                    row += f" {'n/a':>15}"
            print(row)


def print_speedup(results: dict):
    """Print TTFT speedup of prefix modes vs baseline."""
    concs = sorted(set(k[1] for k in results.keys()))
    modes = sorted(set(k[0] for k in results.keys()))

    if "baseline" not in modes:
        return

    print(f"\n{'='*60}")
    print("TTFT Speedup vs Baseline")
    print(f"{'='*60}")
    print(f"{'Mode':<20} {'Conc':>5} {'Baseline':>10} {'Mode':>10} {'Speedup':>10}")
    print("-" * 55)

    for mode in modes:
        if mode == "baseline":
            continue
        for conc in concs:
            base_key = ("baseline", conc)
            mode_key = (mode, conc)
            if base_key not in results or mode_key not in results:
                continue
            base_ttft = results[base_key]["summary"].get("ttft_ms", {}).get("mean", 0)
            mode_ttft = results[mode_key]["summary"].get("ttft_ms", {}).get("mean", 0)
            if base_ttft > 0 and mode_ttft > 0:
                speedup = base_ttft / mode_ttft
                print(f"{mode:<20} {conc:>5} {base_ttft:>9.1f}ms {mode_ttft:>9.1f}ms {speedup:>9.2f}x")


def export_csv(results: dict, output_path: str):
    """Export results to CSV for plotting."""
    rows = []
    for (mode, conc), data in sorted(results.items()):
        s = data["summary"]
        ttft = s.get("ttft_ms", {})
        tpot = s.get("tpot_ms", {})
        sm = s.get("server_metrics", {})
        rows.append({
            "mode": mode,
            "concurrency": conc,
            "ttft_mean_ms": ttft.get("mean", ""),
            "ttft_p50_ms": ttft.get("p50", ""),
            "ttft_p99_ms": ttft.get("p99", ""),
            "tpot_mean_ms": tpot.get("mean", ""),
            "tpot_p50_ms": tpot.get("p50", ""),
            "throughput_tps": s.get("throughput_tps", ""),
            "prefix_cache_hit_rate": sm.get("prefix_cache_hit_rate", ""),
            "kv_cache_usage_pct": sm.get("kv_cache_usage_pct", ""),
            "preemptions": sm.get("preemptions_delta", ""),
            "total_requests": s.get("total_requests", ""),
            "errors": s.get("errors", ""),
            "wall_time_s": s.get("wall_time_s", ""),
        })

    if not rows:
        return

    headers = list(rows[0].keys())
    with open(output_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in headers) + "\n")
    print(f"\nCSV exported: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze KV cache experiment results")
    parser.add_argument("--result_dir", required=True, help="Path to experiment results directory")
    parser.add_argument("--csv", action="store_true", help="Export to CSV")
    parser.add_argument("--summary_only", action="store_true", help="Print summary table only")
    args = parser.parse_args()

    results = load_results(args.result_dir)
    if not results:
        print(f"No results found in {args.result_dir}")
        return

    print(f"\nLoaded {len(results)} result files from {args.result_dir}")

    print_summary_table(results)

    if not args.summary_only:
        print_turn_breakdown(results)
        print_speedup(results)

    if args.csv:
        csv_path = os.path.join(args.result_dir, "comparison.csv")
        export_csv(results, csv_path)


if __name__ == "__main__":
    main()
