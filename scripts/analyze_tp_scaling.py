#!/usr/bin/env python3
"""
TP Scaling & Interconnect Comparison Analysis

Compares batch sweep and agentic results across:
  - H100 PCIe (TP2 Paladin, TP4 Vast.ai)
  - H100 NVLink (TP2, TP4 from 4×H100 SXM)
  - H200 NVLink (TP2, TP4, TP8 from 4×/8× H200)

Generates comparison tables and plots.

Usage:
    python scripts/analyze_tp_scaling.py \
      --h100-pcie-tp2 /path/to/paladin_csv \
      --h100-pcie-tp4 /path/to/tp4_pcie_agentic \
      --h100-nvlink /path/to/h100_sxm_results \
      --h200-nvlink /path/to/h200_results \
      --h200-tp8 /path/to/tp8_results \
      --output-dir figures/
"""

import argparse
import glob
import json
import os
import csv
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not available, text-only output")


def load_batch_sweep_json(directory):
    """Load batch sweep JSON files (H200/H100 NVLink format)."""
    results = []
    for f in sorted(glob.glob(os.path.join(directory, "batch_i*_b*.json"))):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def load_batch_sweep_csv(filepath):
    """Load Paladin CSV format batch sweep."""
    results = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "input_len": int(row["input_len"]),
                "batch_size": int(row["batch_size"]),
                "tpot_mean_ms": float(row["mean_tpot_ms"]),
                "tpot_p50_ms": float(row["p50_tpot_ms"]),
                "ttft_single_ms": float(row["mean_ttft_ms"]),
                "gen_tps": float(row.get("out_tok_s", 0)),
            })
    return results


def load_agentic_summary(directory):
    """Load agentic sweep server_metrics JSONs."""
    results = []
    for f in sorted(glob.glob(os.path.join(directory, "**/server_metrics_*.json"),
                               recursive=True)):
        with open(f) as fh:
            d = json.load(fh)
            # Extract concurrency from filename
            basename = os.path.basename(f)
            parts = basename.replace(".json", "").split("_c")
            conc = int(parts[-1]) if len(parts) > 1 else 0
            d["concurrency"] = conc
            results.append(d)
    # Average over repeats
    by_conc = defaultdict(list)
    for r in results:
        by_conc[r["concurrency"]].append(r)
    averaged = []
    for conc, runs in sorted(by_conc.items()):
        avg = {"concurrency": conc}
        for key in ["tpot_mean_ms", "gen_tps", "ttft_mean_ms"]:
            vals = [r[key] for r in runs if key in r]
            avg[key] = sum(vals) / len(vals) if vals else 0
        avg["n_repeats"] = len(runs)
        averaged.append(avg)
    return averaged


def print_batch_table(label, data):
    """Print batch sweep results as table."""
    if not data:
        print(f"  {label}: no data")
        return
    print(f"\n  {label}:")
    print(f"    {'input':>8} {'batch':>6} {'TPOT':>8} {'TPS':>8} {'TTFT':>8}")
    print(f"    {'-'*42}")
    for r in sorted(data, key=lambda x: (x["input_len"], x["batch_size"])):
        tpot = r.get("tpot_mean_ms", 0)
        tps = r.get("gen_tps", 0)
        ttft = r.get("ttft_single_ms", 0)
        print(f"    {r['input_len']:>8} {r['batch_size']:>6} "
              f"{tpot:>7.2f}ms {tps:>7.1f} {ttft:>7.1f}ms")


def print_agentic_table(label, data):
    """Print agentic sweep results."""
    if not data:
        print(f"  {label}: no data")
        return
    print(f"\n  {label}:")
    print(f"    {'conc':>6} {'TPOT':>8} {'TPS':>8} {'TTFT':>8} {'reps':>5}")
    print(f"    {'-'*40}")
    for r in data:
        print(f"    {r['concurrency']:>6} "
              f"{r.get('tpot_mean_ms',0):>7.2f}ms "
              f"{r.get('gen_tps',0):>7.1f} "
              f"{r.get('ttft_mean_ms',0):>7.1f}ms "
              f"{r.get('n_repeats',0):>5}")


def plot_batch_comparison(configs, output_dir, metric="tpot_mean_ms"):
    """Plot TPOT vs batch_size for each input_len across configs."""
    if not HAS_MPL:
        return

    input_lens = sorted(set(
        r["input_len"] for cfg in configs.values() for r in cfg.get("batch", [])
    ))
    if not input_lens:
        return

    fig, axes = plt.subplots(1, len(input_lens), figsize=(5 * len(input_lens), 4),
                             sharey=True)
    if len(input_lens) == 1:
        axes = [axes]

    colors = {"H100 PCIe TP2": "C0", "H100 PCIe TP4": "C1",
              "H100 NVLink TP2": "C2", "H100 NVLink TP4": "C3",
              "H200 NVLink TP2": "C4", "H200 NVLink TP4": "C5",
              "H200 NVLink TP8": "C6"}
    markers = {"TP2": "o", "TP4": "s", "TP8": "^"}

    for ax, il in zip(axes, input_lens):
        for label, cfg in configs.items():
            data = [r for r in cfg.get("batch", []) if r["input_len"] == il]
            if not data:
                continue
            data.sort(key=lambda x: x["batch_size"])
            bs = [r["batch_size"] for r in data]
            vals = [r[metric] for r in data]
            tp = "TP8" if "TP8" in label else ("TP4" if "TP4" in label else "TP2")
            ax.plot(bs, vals, marker=markers.get(tp, "o"),
                    color=colors.get(label, "gray"), label=label, linewidth=1.5)
        ax.set_title(f"input_len={il}")
        ax.set_xlabel("batch_size")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("TPOT (ms)")
    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.suptitle("Batch Sweep: TPOT vs Batch Size", fontsize=13)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "batch_tpot_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"\n  [PLOT] batch_tpot_comparison.png saved")


def plot_agentic_comparison(configs, output_dir):
    """Plot TPOT vs concurrency across configs."""
    if not HAS_MPL:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for label, cfg in configs.items():
        data = cfg.get("agentic", [])
        if not data:
            continue
        concs = [r["concurrency"] for r in data]
        tpots = [r.get("tpot_mean_ms", 0) for r in data]
        tps = [r.get("gen_tps", 0) for r in data]
        ax1.plot(concs, tpots, marker="o", label=label, linewidth=1.5)
        ax2.plot(concs, tps, marker="s", label=label, linewidth=1.5)

    ax1.set_xlabel("Concurrency")
    ax1.set_ylabel("TPOT (ms)")
    ax1.set_title("Agentic: TPOT vs Concurrency")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.set_xlabel("Concurrency")
    ax2.set_ylabel("Generation TPS")
    ax2.set_title("Agentic: Throughput vs Concurrency")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "agentic_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  [PLOT] agentic_comparison.png saved")


def main():
    parser = argparse.ArgumentParser(description="TP Scaling Analysis")

    # H100 PCIe (existing)
    parser.add_argument("--h100-pcie-tp2-csv", help="Paladin TP2 CSV file")
    parser.add_argument("--h100-pcie-tp4-agentic", help="TP4 PCIe agentic dir")

    # H100 NVLink (new)
    parser.add_argument("--h100-nvlink-dir", help="H100 SXM results dir (has tp2-fp16/, tp4-fp16/)")

    # H200 NVLink (new + existing)
    parser.add_argument("--h200-nvlink-dir", help="H200 results dir (has tp2-fp16/, tp4-fp16/)")
    parser.add_argument("--h200-tp8-batch", help="H200 TP8 batch_sweep_v4 dir")
    parser.add_argument("--h200-tp8-agentic", help="H200 TP8 agentic sweep dir")

    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    configs = {}

    # --- Load H100 PCIe ---
    if args.h100_pcie_tp2_csv and os.path.exists(args.h100_pcie_tp2_csv):
        configs["H100 PCIe TP2"] = {"batch": load_batch_sweep_csv(args.h100_pcie_tp2_csv)}
    if args.h100_pcie_tp4_agentic and os.path.isdir(args.h100_pcie_tp4_agentic):
        configs.setdefault("H100 PCIe TP4", {})["agentic"] = load_agentic_summary(args.h100_pcie_tp4_agentic)

    # --- Load H100 NVLink ---
    if args.h100_nvlink_dir:
        for preset in ["tp2-fp16", "tp4-fp16"]:
            tp_label = "TP2" if "tp2" in preset else "TP4"
            key = f"H100 NVLink {tp_label}"
            batch_dir = os.path.join(args.h100_nvlink_dir, preset, "batch_sweep_v4")
            if os.path.isdir(batch_dir):
                configs.setdefault(key, {})["batch"] = load_batch_sweep_json(batch_dir)
            agentic_dir = os.path.join(args.h100_nvlink_dir, preset)
            if os.path.isdir(agentic_dir):
                agentic = load_agentic_summary(agentic_dir)
                if agentic:
                    configs.setdefault(key, {})["agentic"] = agentic

    # --- Load H200 NVLink ---
    if args.h200_nvlink_dir:
        for preset in ["tp2-fp16", "tp4-fp16"]:
            tp_label = "TP2" if "tp2" in preset else "TP4"
            key = f"H200 NVLink {tp_label}"
            batch_dir = os.path.join(args.h200_nvlink_dir, preset, "batch_sweep_v4")
            if os.path.isdir(batch_dir):
                configs.setdefault(key, {})["batch"] = load_batch_sweep_json(batch_dir)
            agentic_dir = os.path.join(args.h200_nvlink_dir, preset)
            if os.path.isdir(agentic_dir):
                agentic = load_agentic_summary(agentic_dir)
                if agentic:
                    configs.setdefault(key, {})["agentic"] = agentic

    if args.h200_tp8_batch and os.path.isdir(args.h200_tp8_batch):
        configs.setdefault("H200 NVLink TP8", {})["batch"] = load_batch_sweep_json(args.h200_tp8_batch)
    if args.h200_tp8_agentic and os.path.isdir(args.h200_tp8_agentic):
        configs.setdefault("H200 NVLink TP8", {})["agentic"] = load_agentic_summary(args.h200_tp8_agentic)

    if not configs:
        print("[ERROR] No data loaded. Provide at least one --*-dir argument.")
        return

    # --- Print tables ---
    print("=" * 60)
    print(" TP Scaling & Interconnect Comparison")
    print("=" * 60)

    print("\n=== Batch Sweep ===")
    for label, cfg in sorted(configs.items()):
        print_batch_table(label, cfg.get("batch", []))

    print("\n=== Agentic Sweep ===")
    for label, cfg in sorted(configs.items()):
        print_agentic_table(label, cfg.get("agentic", []))

    # --- TP scaling efficiency ---
    print("\n=== TP Scaling Efficiency (batch=1, input=128) ===")
    b1_tpot = {}
    for label, cfg in configs.items():
        batch = cfg.get("batch", [])
        b1 = [r for r in batch if r["batch_size"] == 1 and r["input_len"] == 128]
        if b1:
            b1_tpot[label] = b1[0]["tpot_mean_ms"]
    if b1_tpot:
        print(f"  {'Config':<25} {'TPOT':>8} {'vs TP2':>8}")
        print(f"  {'-'*43}")
        for label in sorted(b1_tpot.keys()):
            tpot = b1_tpot[label]
            # Find matching TP2 for comparison
            gpu_type = label.split(" TP")[0]
            tp2_key = f"{gpu_type} TP2"
            ratio = ""
            if tp2_key in b1_tpot and tp2_key != label:
                ratio = f"{tpot / b1_tpot[tp2_key]:.2f}x"
            print(f"  {label:<25} {tpot:>7.2f}ms {ratio:>8}")

    # --- Interconnect comparison (same TP, different interconnect) ---
    print("\n=== NVLink vs PCIe (batch=1, input=128) ===")
    for tp in ["TP2", "TP4"]:
        pcie = b1_tpot.get(f"H100 PCIe {tp}")
        nvlink = b1_tpot.get(f"H100 NVLink {tp}")
        if pcie and nvlink:
            speedup = pcie / nvlink
            delta = pcie - nvlink
            print(f"  {tp}: PCIe={pcie:.2f}ms → NVLink={nvlink:.2f}ms "
                  f"({speedup:.2f}x, -{delta:.2f}ms)")

    # --- Plots ---
    if HAS_MPL:
        plot_batch_comparison(configs, args.output_dir)
        plot_agentic_comparison(configs, args.output_dir)


if __name__ == "__main__":
    main()
