#!/usr/bin/env python3
"""Generate 4GPU/8GPU comparison figures and summary tables from real-result dumps."""

import csv
import importlib
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_FIG = ROOT / "results_multiagent" / "figures"
OUT_DATA = ROOT / "results_multiagent" / "analysis_4gpu8gpu"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_DATA.mkdir(parents=True, exist_ok=True)
VENDOR_DIR = Path("/tmp/mab_dateutil_vendor")

REAL_BASE = Path(
    os.environ.get(
        "GPUSIM_REAL_RESULTS_BASE",
        "/home/jp/paper_resource/gpusim/results_from_real_H100",
    )
)


def ensure_compatible_dateutil():
    """Matplotlib>=3.8 requires dateutil>=2.7; sandbox env has 2.6.1."""
    if VENDOR_DIR.exists():
        sys.path.insert(0, str(VENDOR_DIR))
    try:
        import dateutil  # type: ignore

        ver = tuple(int(x) for x in dateutil.__version__.split(".")[:2])
        if ver >= (2, 7):
            return

        src_dir = Path(dateutil.__file__).resolve().parent
        dst_dir = VENDOR_DIR / "dateutil"
        if not dst_dir.exists():
            VENDOR_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_dir, dst_dir)
            vfile = dst_dir / "_version.py"
            txt = vfile.read_text()
            txt = txt.replace("VERSION_MINOR = 6", "VERSION_MINOR = 8")
            txt = txt.replace("VERSION_PATCH = 1", "VERSION_PATCH = 2")
            vfile.write_text(txt)

        if str(VENDOR_DIR) not in sys.path:
            sys.path.insert(0, str(VENDOR_DIR))
        # Reload dateutil from vendor path.
        for key in [k for k in list(sys.modules) if k == "dateutil" or k.startswith("dateutil.")]:
            sys.modules.pop(key, None)
        importlib.import_module("dateutil")
    except Exception as e:
        raise RuntimeError(f"Failed to prepare dateutil compatibility shim: {e}") from e


ensure_compatible_dateutil()
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def parse_conc_from_name(path: Path) -> int:
    m = re.search(r"_c(\d+)\.json$", path.name)
    if not m:
        raise ValueError(f"Cannot parse concurrency from: {path}")
    return int(m.group(1))


def average_rows(rows):
    metrics = [
        "tpot_mean_ms",
        "gen_tps",
        "ttft_mean_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
    ]
    out = {}
    for k in metrics:
        vals = [r.get(k) for r in rows if r.get(k) is not None]
        if vals:
            out[k] = float(mean(vals))
    out["n"] = len(rows)
    return out


def load_agentic_sources():
    """Load concurrency-based metrics from summary/sweep directories."""
    data = {}

    # 4GPU: pre-aggregated summary with n_repeats.
    tp4_summary = load_json(
        REAL_BASE / "tp4_fp16_4xH100_PCIe" / "sweep_tp4_fp16_autogen_summary.json"
    )
    tp4_rows = {}
    for r in tp4_summary:
        c = int(r["concurrency"])
        tp4_rows[c] = {
            "tpot_mean_ms": float(r["tpot_mean_ms"]),
            "gen_tps": float(r["gen_tps"]),
            "ttft_mean_ms": float(r["ttft_mean_ms"]),
            "n": int(r.get("n_repeats", 1)),
        }
    data["TP4 FP16 (4xH100 PCIe)"] = tp4_rows

    # 8GPU: repeat directories with server_metrics.
    sweep_specs = {
        "TP8 FP16 (8xH200 NVLink)": REAL_BASE
        / "tp8_fp16_8xH200_NVLink"
        / "sweep_tp8-fp16_autogen",
        "TP8EP BF16 (8xH200 NVLink)": REAL_BASE
        / "tp8ep_bf16_8xH200_NVLink"
        / "agentic_sweep"
        / "sweep_tp8ep-bf16_autogen",
        "TP2DP4EP BF16 (8xH200 NVLink)": REAL_BASE
        / "tp2dp4ep_bf16_8xH200_NVLink"
        / "agentic_sweep"
        / "sweep_tp2dp4ep-bf16_autogen",
        "TP4DP2EP BF16 (8xH200 NVLink)": REAL_BASE
        / "tp4dp2ep_bf16_8xH200_NVLink"
        / "agentic_sweep"
        / "sweep_tp4dp2ep-bf16_autogen",
    }
    for name, base in sweep_specs.items():
        files = sorted(base.glob("**/server_metrics_autogen_c*.json"))
        by_c = defaultdict(list)
        for p in files:
            c = parse_conc_from_name(p)
            row = load_json(p)
            # Skip failed/empty runs that report no token/latency samples.
            if row.get("tpot_count", 1) == 0 and row.get("ttft_count", 1) == 0:
                continue
            if (
                row.get("tpot_mean_ms") is None
                and row.get("ttft_mean_ms") is None
                and float(row.get("gen_tps", 0.0)) == 0.0
            ):
                continue
            by_c[c].append(row)
        data[name] = {c: average_rows(rows) for c, rows in sorted(by_c.items()) if rows}

    return data


def load_batch_summaries():
    """Load batch_sweep_v4 summaries for 8GPU topologies."""
    specs = {
        "TP8 FP16": REAL_BASE
        / "tp8_fp16_8xH200_NVLink"
        / "batch_sweep_v4"
        / "summary.json",
        "TP8EP BF16": REAL_BASE
        / "tp8ep_bf16_8xH200_NVLink"
        / "batch_sweep_v4"
        / "summary.json",
        "TP4DP2 FP16": REAL_BASE
        / "tp4dp2_fp16_8xH200_NVLink"
        / "batch_sweep_v4"
        / "summary.json",
        "TP4DP2EP BF16": REAL_BASE
        / "tp4dp2ep_bf16_8xH200_NVLink"
        / "batch_sweep_v4"
        / "summary.json",
        "TP2DP4EP BF16": REAL_BASE
        / "tp2dp4ep_bf16_8xH200_NVLink"
        / "batch_sweep_v4"
        / "summary.json",
    }
    rows = []
    for topo, path in specs.items():
        data = load_json(path)
        for r in data:
            rows.append(
                {
                    "topology": topo,
                    "input_len": int(r["input_len"]),
                    "batch_size": int(r["batch_size"]),
                    "tpot_mean_ms": float(r["tpot_mean_ms"]),
                    "tpot_p95_ms": float(r["tpot_p95_ms"]),
                    "ttft_single_ms": float(r["ttft_single_ms"]),
                    "gen_tps": float(r["gen_tps"]),
                }
            )
    return rows


def save_flat_tables(agentic, batch_rows):
    agentic_csv = OUT_DATA / "agentic_4gpu8gpu_summary.csv"
    with agentic_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "topology",
                "concurrency",
                "n",
                "tpot_mean_ms",
                "gen_tps",
                "ttft_mean_ms",
                "ttft_p95_ms",
                "ttft_p99_ms",
            ]
        )
        for topo, rows in agentic.items():
            for c, m in sorted(rows.items()):
                w.writerow(
                    [
                        topo,
                        c,
                        m.get("n"),
                        m.get("tpot_mean_ms"),
                        m.get("gen_tps"),
                        m.get("ttft_mean_ms"),
                        m.get("ttft_p95_ms"),
                        m.get("ttft_p99_ms"),
                    ]
                )

    batch_csv = OUT_DATA / "batch_8gpu_summary.csv"
    with batch_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "topology",
                "input_len",
                "batch_size",
                "tpot_mean_ms",
                "tpot_p95_ms",
                "ttft_single_ms",
                "gen_tps",
            ],
        )
        w.writeheader()
        for r in batch_rows:
            w.writerow(r)


def build_figures(agentic, batch_rows):
    # Styling
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 150,
        }
    )
    colors = {
        "TP4 FP16 (4xH100 PCIe)": "#8E44AD",
        "TP8 FP16 (8xH200 NVLink)": "#E74C3C",
        "TP8EP BF16 (8xH200 NVLink)": "#2ECC71",
        "TP2DP4EP BF16 (8xH200 NVLink)": "#3498DB",
        "TP4DP2EP BF16 (8xH200 NVLink)": "#F39C12",
        "TP8 FP16": "#E74C3C",
        "TP8EP BF16": "#2ECC71",
        "TP4DP2 FP16": "#9B59B6",
        "TP4DP2EP BF16": "#F39C12",
        "TP2DP4EP BF16": "#3498DB",
    }

    # Figure 18: Agentic scaling (4GPU vs 8GPU)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    metrics = [
        ("tpot_mean_ms", "TPOT (ms)", "(a) TPOT"),
        ("gen_tps", "Generation TPS", "(b) Throughput"),
        ("ttft_mean_ms", "TTFT Mean (ms)", "(c) TTFT"),
    ]
    for ax, (metric, ylabel, title) in zip(axes, metrics):
        for topo, rows in agentic.items():
            xs = sorted(rows.keys())
            ys = [rows[c].get(metric, np.nan) for c in xs]
            valid = [(x, y) for x, y in zip(xs, ys) if np.isfinite(y)]
            if not valid:
                continue
            vx, vy = zip(*valid)
            if len(vx) >= 2:
                ax.plot(
                    vx,
                    vy,
                    marker="o",
                    linewidth=2,
                    label=topo,
                    color=colors.get(topo, None),
                )
            else:
                ax.scatter(
                    vx,
                    vy,
                    marker="x",
                    s=80,
                    label=topo,
                    color=colors.get(topo, None),
                )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Concurrency")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 8, 32, 64, 128], labels=["1", "8", "32", "64", "128"])
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Figure 18: 4GPU vs 8GPU Agentic Scaling (AutoGen)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_FIG / "fig18_4gpu8gpu_agentic_scaling.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 19: Tail comparison for tp8 vs tp8ep (where repeat data exists)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in [
        (axes[0], "ttft_p95_ms", "(a) TTFT p95"),
        (axes[1], "ttft_p99_ms", "(b) TTFT p99"),
    ]:
        for topo in ["TP8 FP16 (8xH200 NVLink)", "TP8EP BF16 (8xH200 NVLink)"]:
            rows = agentic.get(topo, {})
            xs = sorted(c for c in rows if metric in rows[c])
            ys = [rows[c][metric] for c in xs]
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=2, label=topo, color=colors[topo])
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 8, 32, 64, 128], labels=["1", "8", "32", "64", "128"])
        ax.legend(fontsize=8)
    fig.suptitle("Figure 19: 8GPU Tail Latency — TP8 vs TP8EP (AutoGen)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_FIG / "fig19_8gpu_tail_tp8_vs_tp8ep.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 20: Batch throughput by input length
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, input_len in zip(axes, [128, 512, 2048]):
        for topo in sorted(set(r["topology"] for r in batch_rows)):
            rows = [r for r in batch_rows if r["topology"] == topo and r["input_len"] == input_len]
            rows = sorted(rows, key=lambda r: r["batch_size"])
            if not rows:
                continue
            ax.plot(
                [r["batch_size"] for r in rows],
                [r["gen_tps"] for r in rows],
                marker="o",
                linewidth=2,
                label=topo,
                color=colors.get(topo, None),
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Generation TPS")
        ax.set_title(f"Input Length = {input_len}")
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 8, 16, 32, 64], labels=["1", "8", "16", "32", "64"])
    axes[0].legend(fontsize=8)
    fig.suptitle("Figure 20: 8GPU Batch Throughput by Topology", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_FIG / "fig20_8gpu_batch_throughput_by_input.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 21: Batch TPOT by input length
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, input_len in zip(axes, [128, 512, 2048]):
        for topo in sorted(set(r["topology"] for r in batch_rows)):
            rows = [r for r in batch_rows if r["topology"] == topo and r["input_len"] == input_len]
            rows = sorted(rows, key=lambda r: r["batch_size"])
            if not rows:
                continue
            ax.plot(
                [r["batch_size"] for r in rows],
                [r["tpot_mean_ms"] for r in rows],
                marker="o",
                linewidth=2,
                label=topo,
                color=colors.get(topo, None),
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("TPOT (ms)")
        ax.set_title(f"Input Length = {input_len}")
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 8, 16, 32, 64], labels=["1", "8", "16", "32", "64"])
    axes[0].legend(fontsize=8)
    fig.suptitle("Figure 21: 8GPU Batch TPOT by Topology", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_FIG / "fig21_8gpu_batch_tpot_by_input.png", bbox_inches="tight")
    plt.close(fig)


def compute_key_stats(agentic):
    out = {}
    tp4 = agentic.get("TP4 FP16 (4xH100 PCIe)", {})
    tp8 = agentic.get("TP8 FP16 (8xH200 NVLink)", {})
    tp8ep = agentic.get("TP8EP BF16 (8xH200 NVLink)", {})
    for c in [1, 8, 32, 64]:
        if c in tp4 and c in tp8:
            out[f"tp8_vs_tp4_tps_gain_c{c}_pct"] = (
                (tp8[c]["gen_tps"] - tp4[c]["gen_tps"]) / tp4[c]["gen_tps"] * 100.0
            )
    for c in [8, 32, 64, 128]:
        if c in tp8 and c in tp8ep:
            out[f"tp8ep_vs_tp8_ttft_gain_c{c}_pct"] = (
                (tp8[c]["ttft_mean_ms"] - tp8ep[c]["ttft_mean_ms"])
                / tp8[c]["ttft_mean_ms"]
                * 100.0
            )
            out[f"tp8ep_vs_tp8_tps_gain_c{c}_pct"] = (
                (tp8ep[c]["gen_tps"] - tp8[c]["gen_tps"]) / tp8[c]["gen_tps"] * 100.0
            )
    return out


def main():
    agentic = load_agentic_sources()
    batch_rows = load_batch_summaries()
    save_flat_tables(agentic, batch_rows)
    build_figures(agentic, batch_rows)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "real_results_base": str(REAL_BASE),
        "agentic": agentic,
        "batch_rows": batch_rows,
        "key_stats": compute_key_stats(agentic),
        "generated_figures": [
            "fig18_4gpu8gpu_agentic_scaling.png",
            "fig19_8gpu_tail_tp8_vs_tp8ep.png",
            "fig20_8gpu_batch_throughput_by_input.png",
            "fig21_8gpu_batch_tpot_by_input.png",
        ],
    }
    with (OUT_DATA / "summary_4gpu8gpu.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved figures to: {OUT_FIG}")
    for name in summary["generated_figures"]:
        print(f"  - {name}")
    print(f"Saved data summary: {OUT_DATA / 'summary_4gpu8gpu.json'}")
    print(f"Saved CSV: {OUT_DATA / 'agentic_4gpu8gpu_summary.csv'}")
    print(f"Saved CSV: {OUT_DATA / 'batch_8gpu_summary.csv'}")


if __name__ == "__main__":
    main()
