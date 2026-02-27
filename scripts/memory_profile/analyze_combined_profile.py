#!/usr/bin/env python3
"""Combined analysis of agentic memory profile v2 + v3 (high-concurrency).

Merges c=1..16 (v2) and c=32,64 (v3) data into unified figures for paper.
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_turn_trace(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "benchmark": r["benchmark"],
                "concurrency": int(r["concurrency"]),
                "session_id": int(r["session_id"]),
                "turn": int(r["turn"]),
                "prompt_tokens": int(r["prompt_tokens"]),
                "completion_tokens": int(r["completion_tokens"]),
                "cumulative_context_tokens": int(r["cumulative_context_tokens"]),
                "latency_s": float(r["latency_s"]),
                "status": r["status"],
            })
    return rows


def load_metrics_csv(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v) if v else 0.0
                except ValueError:
                    row[k] = v
            rows.append(row)
    return rows


def load_summary(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure A: Peak KV Cache vs Concurrency (full range c=1..64)
# ---------------------------------------------------------------------------
def fig_peak_cache_full(summaries: list[dict], out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    benchmarks = ["swebench", "terminalbench", "livecodebench"]
    labels = ["SWE-bench", "Terminal Bench", "LiveCodeBench"]
    markers = ["s", "^", "o"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for bm, label, marker, color in zip(benchmarks, labels, markers, colors):
        concs = sorted(set(s["concurrency"] for s in summaries if s["benchmark"] == bm))
        vals = []
        for c in concs:
            match = [s for s in summaries if s["benchmark"] == bm and s["concurrency"] == c]
            vals.append(match[0]["peak_cache_usage_pct"] if match else 0)
        ax.plot(concs, vals, marker=marker, label=label, color=color,
                linewidth=2.5, markersize=8)

    ax.axhline(y=100, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.annotate("100% KV Capacity", xy=(40, 102), fontsize=10, color="red", alpha=0.8)

    # Shade danger zone
    ax.axhspan(90, 110, alpha=0.08, color="red")

    ax.set_xlabel("Concurrent Agent Sessions", fontsize=13)
    ax.set_ylabel("Peak KV Cache Usage (%)", fontsize=13)
    ax.set_title("KV Cache Saturation vs Agent Concurrency\n(Qwen3-Next-80B, TP2 FP16, 2×H100 96GB, max_model_len=32K)",
                 fontsize=13)
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 110)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.tight_layout()
    fig.savefig(out_dir / "paper_F1_peak_cache_vs_concurrency.png", dpi=200)
    plt.close(fig)
    print(f"  paper_F1: peak cache vs concurrency")


# ---------------------------------------------------------------------------
# Figure B: Latency vs Concurrency (full range)
# ---------------------------------------------------------------------------
def fig_latency_vs_concurrency(summaries: list[dict], out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    benchmarks = ["swebench", "terminalbench", "livecodebench"]
    labels = ["SWE-bench", "Terminal Bench", "LiveCodeBench"]
    markers = ["s", "^", "o"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for bm, label, marker, color in zip(benchmarks, labels, markers, colors):
        concs = sorted(set(s["concurrency"] for s in summaries if s["benchmark"] == bm))
        vals = [next(s["avg_latency_s"] for s in summaries
                     if s["benchmark"] == bm and s["concurrency"] == c)
                for c in concs]
        ax.plot(concs, vals, marker=marker, label=label, color=color,
                linewidth=2.5, markersize=8)

    ax.set_xlabel("Concurrent Agent Sessions", fontsize=13)
    ax.set_ylabel("Avg Per-Turn Latency (s)", fontsize=13)
    ax.set_title("Per-Turn Latency vs Agent Concurrency", fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.tight_layout()
    fig.savefig(out_dir / "paper_F2_latency_vs_concurrency.png", dpi=200)
    plt.close(fig)
    print(f"  paper_F2: latency vs concurrency")


# ---------------------------------------------------------------------------
# Figure C: Context growth comparison (3 benchmarks)
# ---------------------------------------------------------------------------
def fig_context_growth_comparison(turns: list[dict], out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    benchmarks = ["swebench", "terminalbench", "livecodebench"]
    labels = ["SWE-bench (4K+2.5K/turn)", "Terminal Bench (0.8K+0.6K/turn)",
              "LiveCodeBench (0.4K+0.3K/turn)"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    markers = ["s", "^", "o"]

    for bm, label, color, marker in zip(benchmarks, labels, colors, markers):
        # Use c=1 data for clean single-session growth
        bm_turns = [t for t in turns
                    if t["benchmark"] == bm and t["concurrency"] == 1 and t["status"] == "ok"]
        turn_nums = sorted(set(t["turn"] for t in bm_turns))
        avg_tokens = []
        for tn in turn_nums:
            vals = [t["prompt_tokens"] for t in bm_turns if t["turn"] == tn]
            avg_tokens.append(np.mean(vals) / 1000)
        ax.plot(turn_nums, avg_tokens, marker=marker, label=label, color=color,
                linewidth=2.5, markersize=8)

    ax.set_xlabel("Turn Number", fontsize=13)
    ax.set_ylabel("Prompt Tokens (K)", fontsize=13)
    ax.set_title("Context Growth Per Turn — Single Agent Session", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(out_dir / "paper_F3_context_growth.png", dpi=200)
    plt.close(fig)
    print(f"  paper_F3: context growth per turn")


# ---------------------------------------------------------------------------
# Figure D: Preemptions + Cache hit rate vs concurrency
# ---------------------------------------------------------------------------
def fig_preemption_and_cache(v2_dir: Path, v3_dir: Path, out_dir: Path):
    """Dual-axis chart: preemptions and prefix cache hit rate vs concurrency."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, bm, title in [(ax1, "swebench", "SWE-bench"), (ax2, "terminalbench", "Terminal Bench")]:
        concs = []
        preemptions = []
        hit_rates = []

        for conc in [1, 2, 4, 8, 16, 32, 64]:
            if conc <= 16:
                mpath = v2_dir / f"metrics_{bm}_c{conc}.csv"
            else:
                mpath = v3_dir / f"metrics_{bm}_c{conc}.csv"
            if not mpath.exists():
                continue

            metrics = load_metrics_csv(mpath)
            if not metrics:
                continue

            concs.append(conc)

            # Preemptions delta
            p_start = metrics[0].get("vllm:num_preemptions_total", 0)
            p_end = metrics[-1].get("vllm:num_preemptions_total", 0)
            preemptions.append(p_end - p_start)

            # Prefix cache hit rate
            q_start = metrics[0].get("vllm:prefix_cache_queries_total", 0)
            q_end = metrics[-1].get("vllm:prefix_cache_queries_total", 0)
            h_start = metrics[0].get("vllm:prefix_cache_hits_total", 0)
            h_end = metrics[-1].get("vllm:prefix_cache_hits_total", 0)
            dq = q_end - q_start
            dh = h_end - h_start
            hr = (dh / dq * 100) if dq > 0 else 0
            hit_rates.append(hr)

        color_bar = "#E53935"
        color_line = "#1565C0"

        ax_twin = ax.twinx()
        bars = ax.bar(range(len(concs)), preemptions, alpha=0.6, color=color_bar, label="Preemptions")
        line = ax_twin.plot(range(len(concs)), hit_rates, marker="o", color=color_line,
                           linewidth=2.5, markersize=8, label="Prefix Cache Hit %")

        ax.set_xticks(range(len(concs)))
        ax.set_xticklabels([str(c) for c in concs])
        ax.set_xlabel("Concurrency", fontsize=12)
        ax.set_ylabel("Preemptions (count)", fontsize=12, color=color_bar)
        ax_twin.set_ylabel("Prefix Cache Hit Rate (%)", fontsize=12, color=color_line)
        ax_twin.set_ylim(0, 80)
        ax.set_title(title, fontsize=13)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=10)

    fig.suptitle("KV Cache Saturation Effects: Preemptions & Prefix Cache Destruction", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "paper_F4_preemption_cache.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  paper_F4: preemption + cache hit rate")


# ---------------------------------------------------------------------------
# Figure E: KV Cache Timeline c=16 vs c=32 vs c=64 (swebench)
# ---------------------------------------------------------------------------
def fig_cache_timeline_high(v2_dir: Path, v3_dir: Path, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"16": "#FFC107", "32": "#FF5722", "64": "#B71C1C"}

    for ax, bm, title in [(axes[0], "swebench", "SWE-bench"),
                           (axes[1], "terminalbench", "Terminal Bench")]:
        for conc_str, conc in [("16", 16), ("32", 32), ("64", 64)]:
            if conc <= 16:
                mpath = v2_dir / f"metrics_{bm}_c{conc}.csv"
            else:
                mpath = v3_dir / f"metrics_{bm}_c{conc}.csv"
            if not mpath.exists():
                continue

            metrics = load_metrics_csv(mpath)
            elapsed = [m["elapsed_s"] for m in metrics]
            cache = [m.get("vllm:kv_cache_usage_perc", 0) * 100 for m in metrics]
            ax.plot(elapsed, cache, label=f"c={conc_str}", color=colors[conc_str],
                    linewidth=2, alpha=0.9)

        ax.axhline(y=100, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xlabel("Elapsed Time (s)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 110)
        ax.legend(fontsize=11)

    axes[0].set_ylabel("KV Cache Usage (%)", fontsize=12)
    fig.suptitle("KV Cache Timeline: Saturation at High Concurrency", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "paper_F5_cache_timeline_high.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  paper_F5: cache timeline c=16/32/64")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    v2_dir = Path("results/memory_profile/agentic_v2")
    v3_dir = Path("results/memory_profile/agentic_v3_highconc")
    out_dir = Path("results/memory_profile/paper_figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and merge summaries
    s_v2 = load_summary(v2_dir / "summary.json")
    s_v3 = load_summary(v3_dir / "summary.json")
    summaries = s_v2 + s_v3

    # Load and merge turn traces
    turns_v2 = load_turn_trace(v2_dir / "turn_trace.csv")
    turns_v3 = load_turn_trace(v3_dir / "turn_trace.csv")
    turns = turns_v2 + turns_v3

    print(f"Merged: {len(summaries)} summary entries, {len(turns)} turn records\n")

    # Generate paper figures
    fig_peak_cache_full(summaries, out_dir)
    fig_latency_vs_concurrency(summaries, out_dir)
    fig_context_growth_comparison(turns, out_dir)
    fig_preemption_and_cache(v2_dir, v3_dir, out_dir)
    fig_cache_timeline_high(v2_dir, v3_dir, out_dir)

    # Print combined summary table
    print(f"\n{'='*110}")
    print(f"COMBINED SUMMARY — Agentic Memory Profile")
    print(f"{'='*110}")
    print(f"{'Benchmark':<16} {'Conc':>5} {'Turns':>6} {'MaxCtx':>8} {'PeakKV%':>8} "
          f"{'AvgLat':>8} {'AvgPrompt':>10} {'AvgCompl':>10}")
    print("-" * 110)
    for s in sorted(summaries, key=lambda x: (x["benchmark"], x["concurrency"])):
        print(f"{s['benchmark']:<16} {s['concurrency']:>5} {s['total_turns']:>6} "
              f"{s['max_context_tokens']:>8,} {s['peak_cache_usage_pct']:>7.1f}% "
              f"{s['avg_latency_s']:>7.3f}s {s['avg_prompt_tokens']:>10,} "
              f"{s['avg_completion_tokens']:>10}")

    # Benchmark comparison
    print(f"\n{'='*90}")
    print("BENCHMARK COMPARISON (c=1 baseline)")
    print(f"{'='*90}")
    print(f"{'Metric':<30} {'SWE-bench':>15} {'Terminal Bench':>15} {'LiveCodeBench':>15}")
    print("-" * 90)

    for bm_data in [("swebench", "SWE-bench"), ("terminalbench", "Terminal Bench"), ("livecodebench", "LiveCodeBench")]:
        pass

    bms = {"swebench": {}, "terminalbench": {}, "livecodebench": {}}
    for s in summaries:
        bms.setdefault(s["benchmark"], {})[s["concurrency"]] = s

    metrics_list = [
        ("Peak context (tokens)", lambda s: f"{s.get('max_context_tokens', 0):,}"),
        ("Turns per session", lambda s: f"{s.get('total_turns', 0) // max(s.get('sessions', 1), 1)}"),
        ("Avg prompt tokens/turn", lambda s: f"{s.get('avg_prompt_tokens', 0):,}"),
        ("Avg completion tokens/turn", lambda s: f"{s.get('avg_completion_tokens', 0)}"),
        ("Avg latency/turn (c=1)", lambda s: f"{s.get('avg_latency_s', 0):.3f}s"),
    ]
    for label, fn in metrics_list:
        vals = []
        for bm in ["swebench", "terminalbench", "livecodebench"]:
            s = bms.get(bm, {}).get(1, {})
            vals.append(fn(s) if s else "—")
        print(f"{label:<30} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15}")

    # KV cache per session
    print(f"\n{'Concurrency':<15} ", end="")
    for bm in ["SWE-bench", "Terminal", "LiveCode"]:
        print(f" {bm:>12}", end="")
    print()
    print("-" * 55)
    for conc in [1, 2, 4, 8, 16, 32, 64]:
        print(f"  c={conc:<11}", end="")
        for bm in ["swebench", "terminalbench", "livecodebench"]:
            s = bms.get(bm, {}).get(conc, {})
            pct = s.get("peak_cache_usage_pct", None)
            if pct is not None:
                print(f" {pct:>11.1f}%", end="")
            else:
                print(f" {'—':>12}", end="")
        print()

    # Write merged summary
    merged_path = out_dir / "merged_summary.json"
    with open(merged_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nMerged summary: {merged_path}")
    print(f"Figures: {out_dir}/")


if __name__ == "__main__":
    main()
