#!/usr/bin/env python3
"""Analyze agentic multi-turn memory profile results.

Generates:
1. KV cache usage over time per concurrency (line chart)
2. Context growth per turn per session (line chart)
3. Peak cache vs concurrency (bar chart)
4. Latency vs turn (showing prefill overhead growth)
5. Summary table
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_turn_trace(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
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
        reader = csv.DictReader(f)
        for r in reader:
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
# Figure 1: KV Cache Usage Over Time (per concurrency, one benchmark)
# ---------------------------------------------------------------------------
def fig_cache_over_time(data_dir: Path, out_dir: Path, benchmark: str = "swebench"):
    """Time-series of KV cache usage for each concurrency level."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 5))

    for i, conc in enumerate([1, 2, 4, 8, 16]):
        mpath = data_dir / f"metrics_{benchmark}_c{conc}.csv"
        if not mpath.exists():
            continue
        metrics = load_metrics_csv(mpath)
        elapsed = [m["elapsed_s"] for m in metrics]
        cache = [m.get("vllm:kv_cache_usage_perc", 0) * 100 for m in metrics]
        ax.plot(elapsed, cache, label=f"c={conc}", color=colors[i], linewidth=2)

    ax.set_xlabel("Elapsed Time (s)", fontsize=12)
    ax.set_ylabel("KV Cache Usage (%)", fontsize=12)
    ax.set_title(f"KV Cache Usage Over Time — {benchmark} (Qwen3-Next-80B, TP2 FP16)", fontsize=13)
    ax.legend(fontsize=11, title="Concurrency")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_dir / f"F1_cache_timeline_{benchmark}.png", dpi=150)
    plt.close(fig)
    print(f"  F1: {out_dir}/F1_cache_timeline_{benchmark}.png")


# ---------------------------------------------------------------------------
# Figure 2: Context Growth Per Turn (averaged across sessions)
# ---------------------------------------------------------------------------
def fig_context_growth(turns: list[dict], out_dir: Path):
    """Context tokens vs turn number, grouped by concurrency."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    benchmarks = ["swebench", "terminalbench", "livecodebench"]
    titles = ["SWE-bench", "Terminal Bench", "LiveCodeBench"]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 5))

    for ax, bm, title in zip(axes, benchmarks, titles):
        bm_turns = [t for t in turns if t["benchmark"] == bm and t["status"] == "ok"]
        for i, conc in enumerate([1, 2, 4, 8, 16]):
            ct = [t for t in bm_turns if t["concurrency"] == conc]
            if not ct:
                continue
            # Average across sessions per turn
            turn_data = {}
            for t in ct:
                turn_data.setdefault(t["turn"], []).append(t["prompt_tokens"])
            turn_nums = sorted(turn_data.keys())
            avg_tokens = [np.mean(turn_data[tn]) for tn in turn_nums]
            ax.plot(turn_nums, [t / 1000 for t in avg_tokens],
                    marker="o", label=f"c={conc}", color=colors[i], linewidth=2)

        ax.set_xlabel("Turn", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axes[0].set_ylabel("Prompt Tokens (K)", fontsize=11)
    axes[2].legend(fontsize=10, title="Concurrency", loc="upper left")
    fig.suptitle("Context Growth Per Turn — Multi-Turn Agentic Sessions", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "F2_context_growth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  F2: {out_dir}/F2_context_growth.png")


# ---------------------------------------------------------------------------
# Figure 3: Peak KV Cache Usage vs Concurrency
# ---------------------------------------------------------------------------
def fig_peak_cache(summary: list[dict], out_dir: Path):
    """Bar chart: peak cache usage per concurrency, grouped by benchmark."""
    fig, ax = plt.subplots(figsize=(10, 6))
    benchmarks = ["swebench", "terminalbench", "livecodebench"]
    labels = ["SWE-bench", "Terminal Bench", "LiveCodeBench"]
    concurrencies = [1, 2, 4, 8, 16]
    x = np.arange(len(concurrencies))
    width = 0.25

    for i, (bm, label) in enumerate(zip(benchmarks, labels)):
        vals = []
        for c in concurrencies:
            match = [s for s in summary if s["benchmark"] == bm and s["concurrency"] == c]
            vals.append(match[0]["peak_cache_usage_pct"] if match else 0)
        ax.bar(x + i * width, vals, width, label=label, alpha=0.85)

    ax.set_xlabel("Concurrent Agent Sessions", fontsize=12)
    ax.set_ylabel("Peak KV Cache Usage (%)", fontsize=12)
    ax.set_title("Peak KV Cache vs Concurrency — ~27K Context per Session", fontsize=13)
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(c) for c in concurrencies])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Add capacity line
    ax.axhline(y=100, color="red", linestyle="--", alpha=0.7, label="100% capacity")
    # Extrapolate to show when we'd hit 100%
    # At c=16, ~64%. Linear: 100% at c=25
    ax.annotate(f"~100% at c≈25\n(extrapolated)", xy=(4.3, 70),
                fontsize=10, color="red", alpha=0.8)

    fig.tight_layout()
    fig.savefig(out_dir / "F3_peak_cache_vs_concurrency.png", dpi=150)
    plt.close(fig)
    print(f"  F3: {out_dir}/F3_peak_cache_vs_concurrency.png")


# ---------------------------------------------------------------------------
# Figure 4: Latency vs Turn (showing prefill cost growth)
# ---------------------------------------------------------------------------
def fig_latency_growth(turns: list[dict], out_dir: Path):
    """Latency per turn — shows how prefill cost grows with context."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    benchmarks = ["swebench", "terminalbench", "livecodebench"]
    titles = ["SWE-bench", "Terminal Bench", "LiveCodeBench"]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 5))

    for ax, bm, title in zip(axes, benchmarks, titles):
        bm_turns = [t for t in turns if t["benchmark"] == bm and t["status"] == "ok"]
        for i, conc in enumerate([1, 2, 4, 8, 16]):
            ct = [t for t in bm_turns if t["concurrency"] == conc]
            if not ct:
                continue
            turn_data = {}
            for t in ct:
                turn_data.setdefault(t["turn"], []).append(t["latency_s"])
            turn_nums = sorted(turn_data.keys())
            avg_lat = [np.mean(turn_data[tn]) for tn in turn_nums]
            ax.plot(turn_nums, avg_lat, marker="s", label=f"c={conc}",
                    color=colors[i], linewidth=2)

        ax.set_xlabel("Turn", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axes[0].set_ylabel("Latency (s)", fontsize=11)
    axes[2].legend(fontsize=10, title="Concurrency", loc="upper left")
    fig.suptitle("Per-Turn Latency Growth — Prefill Overhead + Decode", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "F4_latency_per_turn.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  F4: {out_dir}/F4_latency_per_turn.png")


# ---------------------------------------------------------------------------
# Figure 5: Combined Dashboard — cache timeline for all 3 benchmarks
# ---------------------------------------------------------------------------
def fig_cache_dashboard(data_dir: Path, out_dir: Path):
    """3-panel cache timeline, all benchmarks."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    benchmarks = ["swebench", "terminalbench", "livecodebench"]
    titles = ["SWE-bench", "Terminal Bench", "LiveCodeBench"]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 5))

    for ax, bm, title in zip(axes, benchmarks, titles):
        for i, conc in enumerate([1, 2, 4, 8, 16]):
            mpath = data_dir / f"metrics_{bm}_c{conc}.csv"
            if not mpath.exists():
                continue
            metrics = load_metrics_csv(mpath)
            elapsed = [m["elapsed_s"] for m in metrics]
            cache = [m.get("vllm:kv_cache_usage_perc", 0) * 100 for m in metrics]
            ax.plot(elapsed, cache, label=f"c={conc}", color=colors[i], linewidth=1.5)

        ax.set_xlabel("Elapsed (s)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("KV Cache Usage (%)", fontsize=11)
    axes[2].legend(fontsize=9, title="Concurrency")
    axes[0].set_ylim(0, 80)
    fig.suptitle("KV Cache Timeline — All Benchmarks (Qwen3-Next-80B, TP2 FP16, max_model_len=32K)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "F5_cache_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  F5: {out_dir}/F5_cache_dashboard.png")


# ---------------------------------------------------------------------------
# Summary Table
# ---------------------------------------------------------------------------
def print_summary_table(summary: list[dict], turns: list[dict]):
    print("\n" + "=" * 100)
    print("AGENTIC MEMORY PROFILE — SUMMARY")
    print("=" * 100)
    print(f"\n{'Benchmark':<16} {'Conc':>5} {'Turns':>6} {'MaxCtx':>8} {'PeakKV%':>8} "
          f"{'AvgLat':>8} {'AvgPrompt':>10} {'AvgCompl':>10}")
    print("-" * 100)
    for s in summary:
        print(f"{s['benchmark']:<16} {s['concurrency']:>5} {s['total_turns']:>6} "
              f"{s['max_context_tokens']:>8,} {s['peak_cache_usage_pct']:>7.1f}% "
              f"{s['avg_latency_s']:>7.3f}s {s['avg_prompt_tokens']:>10,} "
              f"{s['avg_completion_tokens']:>10}")

    # Per-turn context growth
    print("\n\nPER-TURN CONTEXT (prompt_tokens, averaged across all benchmarks):")
    print(f"{'Turn':>5}  {'c=1':>8}  {'c=2':>8}  {'c=4':>8}  {'c=8':>8}  {'c=16':>8}")
    print("-" * 55)
    ok_turns = [t for t in turns if t["status"] == "ok"]
    for turn in range(7):
        vals = []
        for conc in [1, 2, 4, 8, 16]:
            ct = [t["prompt_tokens"] for t in ok_turns
                  if t["concurrency"] == conc and t["turn"] == turn]
            vals.append(int(np.mean(ct)) if ct else 0)
        if any(v > 0 for v in vals):
            print(f"{turn:>5}  {'':>0}" +
                  "  ".join(f"{v:>8,}" if v > 0 else f"{'—':>8}" for v in vals))

    # KV capacity extrapolation
    print("\n\nKV CACHE CAPACITY EXTRAPOLATION:")
    print("  (assuming ~27K context per session, linear scaling)")
    # Average peak cache per concurrency across benchmarks
    for conc in [1, 2, 4, 8, 16]:
        vals = [s["peak_cache_usage_pct"] for s in summary if s["concurrency"] == conc]
        avg_pct = np.mean(vals)
        # Extrapolate: at what concurrency would we hit 100%?
        est_max_conc = conc * (100.0 / avg_pct) if avg_pct > 0 else float("inf")
        print(f"  c={conc:>2}: peak KV = {avg_pct:.1f}% → est. max concurrent sessions ≈ {est_max_conc:.0f}")

    # Latency scaling
    print("\n\nLATENCY SCALING (avg per-turn latency):")
    for conc in [1, 2, 4, 8, 16]:
        vals = [s["avg_latency_s"] for s in summary if s["concurrency"] == conc]
        avg = np.mean(vals)
        baseline = np.mean([s["avg_latency_s"] for s in summary if s["concurrency"] == 1])
        print(f"  c={conc:>2}: {avg:.3f}s  ({avg / baseline:.1f}x vs c=1)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/memory_profile/agentic_full")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else data_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data: {data_dir}/")
    print(f"Output: {out_dir}/")

    # Load data
    turns = load_turn_trace(data_dir / "turn_trace.csv")
    summary = load_summary(data_dir / "summary.json")

    print(f"\nLoaded {len(turns)} turn records, {len(summary)} summary entries\n")

    # Generate figures
    fig_cache_over_time(data_dir, out_dir, "swebench")
    fig_context_growth(turns, out_dir)
    fig_peak_cache(summary, out_dir)
    fig_latency_growth(turns, out_dir)
    fig_cache_dashboard(data_dir, out_dir)

    # Print analysis
    print_summary_table(summary, turns)


if __name__ == "__main__":
    main()
