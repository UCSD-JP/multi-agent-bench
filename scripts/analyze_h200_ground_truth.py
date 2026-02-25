#!/usr/bin/env python3
"""H200 Ground Truth Analysis — Batch Sweep + Agentic Sweep across TP configs + H100.

Figures: 10 comparison-focused PNGs with error bars, annotations, captions.
Tables: T1-T4 printed to stdout.
CSVs: 3 files for simulator input.

Usage:
    python scripts/analyze_h200_ground_truth.py [--output-dir results/h200_analysis/]
"""

import argparse
import csv
import importlib
import json
import math
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# dateutil compat shim
# ---------------------------------------------------------------------------
VENDOR_DIR = Path("/tmp/mab_dateutil_vendor")


def ensure_compatible_dateutil():
    if VENDOR_DIR.exists():
        sys.path.insert(0, str(VENDOR_DIR))
    try:
        import dateutil
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
        for key in [k for k in list(sys.modules) if k == "dateutil" or k.startswith("dateutil.")]:
            sys.modules.pop(key, None)
        importlib.import_module("dateutil")
    except Exception:
        pass


ensure_compatible_dateutil()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MAB_ROOT = Path(__file__).resolve().parents[1]
GPUSIM_ROOT = MAB_ROOT.parent / "gpusim"

H200_CONFIGS = {
    "TP2 H200 NVLink": {
        "batch_dir": MAB_ROOT / "results" / "h200_4gpu" / "tp2-fp16" / "batch_sweep_v4",
        "agentic_dir": MAB_ROOT / "results" / "h200_4gpu" / "tp2-fp16" / "agentic_sweep",
        "conc_levels": [1, 8, 32, 64], "n_tasks": 64, "n_repeats": 3,
        "color": "#2196F3", "marker": "o", "tp": 2, "gpu": "H200", "ic": "NVLink",
    },
    "TP4 H200 NVLink": {
        "batch_dir": MAB_ROOT / "results" / "h200_4gpu" / "tp4-fp16" / "batch_sweep_v4",
        "agentic_dir": MAB_ROOT / "results" / "h200_4gpu" / "tp4-fp16" / "agentic_sweep",
        "conc_levels": [1, 8, 32, 64], "n_tasks": 64, "n_repeats": 3,
        "color": "#FF9800", "marker": "s", "tp": 4, "gpu": "H200", "ic": "NVLink",
    },
    "TP8 H200 NVLink": {
        "batch_dir": GPUSIM_ROOT / "results_from_real_H100" / "tp8_fp16_8xH200_NVLink" / "batch_sweep_v4",
        "agentic_dir": GPUSIM_ROOT / "results_from_real_H100" / "tp8_fp16_8xH200_NVLink" / "sweep_tp8-fp16_autogen",
        "conc_levels": [1, 8, 32, 64, 128], "n_tasks": 48, "n_repeats": 3,
        "color": "#E74C3C", "marker": "^", "tp": 8, "gpu": "H200", "ic": "NVLink",
    },
    "TP8+EP H200 NVLink": {
        "batch_dir": GPUSIM_ROOT / "results_from_real_H100" / "tp8ep_bf16_8xH200_NVLink" / "batch_sweep_v4",
        "agentic_dir": GPUSIM_ROOT / "results_from_real_H100" / "tp8ep_bf16_8xH200_NVLink" / "agentic_sweep" / "sweep_tp8ep-bf16_autogen",
        "conc_levels": [1, 8, 32, 64, 128], "n_tasks": 48, "n_repeats": 3,
        "color": "#4CAF50", "marker": "D", "tp": 8, "gpu": "H200", "ic": "NVLink",
    },
    # --- DP configs (batch only, agentic unusable due to vLLM 0.15.1 DP crash) ---
    "TP4-DP2 H200 NVLink": {
        "batch_dir": GPUSIM_ROOT / "results_from_real_H100" / "tp4dp2_fp16_8xH200_NVLink" / "batch_sweep_v4",
        "agentic_dir": None,
        "conc_levels": [], "n_tasks": 0, "n_repeats": 0,
        "color": "#795548", "marker": "v", "tp": 4, "dp": 2, "gpu": "H200", "ic": "NVLink",
    },
    "TP4-DP2-EP H200 NVLink": {
        "batch_dir": GPUSIM_ROOT / "results_from_real_H100" / "tp4dp2ep_bf16_8xH200_NVLink" / "batch_sweep_v4",
        "agentic_dir": None,
        "conc_levels": [], "n_tasks": 0, "n_repeats": 0,
        "color": "#607D8B", "marker": "h", "tp": 4, "dp": 2, "gpu": "H200", "ic": "NVLink",
    },
    "TP2-DP4-EP H200 NVLink": {
        "batch_dir": GPUSIM_ROOT / "results_from_real_H100" / "tp2dp4ep_bf16_8xH200_NVLink" / "batch_sweep_v4",
        "agentic_dir": None,
        "conc_levels": [], "n_tasks": 0, "n_repeats": 0,
        "color": "#009688", "marker": "p", "tp": 2, "dp": 4, "gpu": "H200", "ic": "NVLink",
    },
}

H100_CONFIG = {
    "TP4 H100 PCIe": {
        "summary_path": GPUSIM_ROOT / "results_from_real_H100" / "tp4_fp16_4xH100_PCIe" / "sweep_tp4_fp16_autogen_summary.json",
        "color": "#9C27B0", "marker": "P", "tp": 4, "gpu": "H100", "ic": "PCIe",
    },
}

ALL_CONFIGS = list(H200_CONFIGS.keys()) + list(H100_CONFIG.keys())

PLT_RC = {
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
}


# ============================================================================
# Helpers
# ============================================================================
def pct(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


def ci95(vals):
    """95% CI half-width. Returns 0 if n<2."""
    n = len(vals)
    if n < 2:
        return 0.0
    return 1.96 * statistics.stdev(vals) / math.sqrt(n)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def ccol(name):
    for d in (H200_CONFIGS, H100_CONFIG):
        if name in d:
            return d[name]["color"]
    return "#757575"


# ============================================================================
# Data Loading
# ============================================================================
# Anomalous batch points to exclude: (config_name, input_len, batch_size)
BATCH_POINT_ANOMALIES = {
    ("TP2-DP4-EP H200 NVLink", 128, 8),  # TPOT=406ms, cloud measurement anomaly
}


def load_batch_sweep(config_name):
    cfg = H200_CONFIGS[config_name]
    sp = cfg["batch_dir"] / "summary.json"
    if not sp.exists():
        print(f"  [SKIP] batch: {sp}")
        return []
    data = load_json(sp)
    rows = []
    for r in data:
        il, bs = int(r["input_len"]), int(r["batch_size"])
        if (config_name, il, bs) in BATCH_POINT_ANOMALIES:
            print(f"  [PRUNE] {config_name} i={il} b={bs} tpot={r['tpot_mean_ms']:.2f}ms (anomaly)")
            continue
        rows.append({
            "config": config_name,
            "input_len": int(r["input_len"]),
            "batch_size": int(r["batch_size"]),
            "output_len": int(r.get("output_len", 128)),
            "tpot_mean_ms": float(r["tpot_mean_ms"]),
            "tpot_p50_ms": float(r.get("tpot_p50_ms", 0)),
            "ttft_single_ms": float(r.get("ttft_single_ms", 0)),
            "ttft_mean_ms": float(r.get("ttft_mean_ms", 0)),
            "gen_tps": float(r.get("gen_tps", 0)),
            "sys_tpot_ms": float(r.get("sys_tpot_ms", 0)),
            "e2e_mean_ms": float(r.get("e2e_mean_ms", 0)),
        })
    print(f"  {len(rows)} batch pts — {config_name}")
    return rows


def load_agentic_traces(config_name):
    cfg = H200_CONFIGS[config_name]
    d = cfg["agentic_dir"]
    if d is None or not d.exists():
        print(f"  [SKIP] agentic: {d}")
        return []
    rows = []
    for rep in range(1, cfg["n_repeats"] + 1):
        for conc in cfg["conc_levels"]:
            cd = d / f"r{rep}" / f"autogen_c{conc}"
            if not cd.exists():
                continue
            traces = list(cd.glob("trace_*.jsonl"))
            if not traces:
                continue
            for task in load_jsonl(traces[0]):
                for sid, step in task.get("steps", {}).items():
                    if not step.get("ok", False):
                        continue
                    tpot = step.get("tpot_ms")
                    if tpot is None or tpot <= 0:
                        continue
                    rows.append({
                        "config": config_name, "concurrency": conc, "repeat": rep,
                        "task_id": task.get("task_id", -1), "step_id": sid,
                        "agent_role": step.get("agent_role", "unknown"),
                        "tpot_ms": float(tpot),
                        "ttft_ms": float(step.get("ttft_ms", 0)),
                        "latency_ms": float(step.get("latency_ms", 0)),
                        "prompt_tokens": int(step.get("prompt_tokens", 0)),
                        "completion_tokens": int(step.get("completion_tokens", 0)),
                    })
    print(f"  {len(rows)} agentic steps — {config_name}")
    return rows


def load_agentic_server_metrics(config_name):
    """Per (config,conc) → {gen_tps, tpot_mean_ms, ttft_mean_ms} + per-repeat for CI."""
    cfg = H200_CONFIGS[config_name]
    d = cfg["agentic_dir"]
    if d is None or not d.exists():
        return {}, {}
    by_conc = defaultdict(list)
    for rep in range(1, cfg["n_repeats"] + 1):
        for conc in cfg["conc_levels"]:
            cd = d / f"r{rep}" / f"autogen_c{conc}"
            mfs = list(cd.glob("server_metrics_*.json")) if cd.exists() else []
            for mf in mfs:
                m = load_json(mf)
                by_conc[conc].append({
                    "gen_tps": float(m.get("gen_tps", 0)),
                    "tpot_mean_ms": float(m.get("tpot_mean_ms", 0)),
                    "ttft_mean_ms": float(m.get("ttft_mean_ms", 0)),
                })
    means, per_rep = {}, {}
    for conc, items in by_conc.items():
        means[conc] = {k: statistics.mean([i[k] for i in items]) for k in items[0]}
        per_rep[conc] = items  # list of dicts, one per repeat
    return means, per_rep


def load_h100_agentic():
    cfg = H100_CONFIG["TP4 H100 PCIe"]
    path = cfg["summary_path"]
    if not path.exists():
        print(f"  [SKIP] H100: {path}")
        return [], {}, {}
    data = load_json(path)
    agg, smeans, srep = [], {}, {}
    for r in data:
        c = int(r["concurrency"])
        agg.append({
            "config": "TP4 H100 PCIe", "concurrency": c,
            "n_repeats": int(r.get("n_repeats", 3)),
            "tpot_mean": float(r["tpot_mean_ms"]),
            "tpot_p50": float(r.get("tpot_p50_ms", 0)),
            "tpot_p95": float("nan"),
            "tpot_ci": 0.0,
            "ttft_mean": float(r.get("ttft_mean_ms", 0)),
            "ttft_p50": float("nan"), "ttft_p95": float("nan"), "ttft_ci": 0.0,
            "n_steps_total": 0,
        })
        smeans[c] = {"gen_tps": float(r.get("gen_tps", 0)),
                      "tpot_mean_ms": float(r["tpot_mean_ms"]),
                      "ttft_mean_ms": float(r.get("ttft_mean_ms", 0))}
        srep[c] = []  # no per-repeat breakdown available
    print(f"  {len(agg)} agentic pts — TP4 H100 PCIe")
    return agg, smeans, srep


def load_all_data():
    print("Loading data...")
    all_batch, all_agentic = [], []
    server_means, server_reps = {}, {}
    for name in H200_CONFIGS:
        all_batch.extend(load_batch_sweep(name))
        all_agentic.extend(load_agentic_traces(name))
        sm, sr = load_agentic_server_metrics(name)
        server_means[name] = sm
        server_reps[name] = sr
    h100_agg, h100_sm, h100_sr = load_h100_agentic()
    server_means["TP4 H100 PCIe"] = h100_sm
    server_reps["TP4 H100 PCIe"] = h100_sr
    return all_batch, all_agentic, h100_agg, server_means, server_reps


# ============================================================================
# Aggregation (with CI)
# ============================================================================
def aggregate_agentic(agentic_steps):
    """Returns list of dicts with tpot_mean, tpot_ci, ttft_mean, ttft_ci, ..."""
    grouped = defaultdict(list)
    for s in agentic_steps:
        grouped[(s["config"], s["concurrency"], s["repeat"])].append(s)

    repeat_stats = defaultdict(list)
    for (config, conc, rep), steps in grouped.items():
        tpots = [s["tpot_ms"] for s in steps]
        ttfts = [s["ttft_ms"] for s in steps if s["ttft_ms"] > 0]
        repeat_stats[(config, conc)].append({
            "tpot_mean": statistics.mean(tpots),
            "tpot_p50": pct(tpots, 50), "tpot_p95": pct(tpots, 95),
            "ttft_mean": statistics.mean(ttfts) if ttfts else float("nan"),
            "ttft_p50": pct(ttfts, 50) if ttfts else float("nan"),
            "ttft_p95": pct(ttfts, 95) if ttfts else float("nan"),
            "n_steps": len(tpots),
        })

    agg = []
    for (config, conc), stats_list in sorted(repeat_stats.items()):
        row = {"config": config, "concurrency": conc, "n_repeats": len(stats_list)}
        for key in ["tpot_mean", "tpot_p50", "tpot_p95", "ttft_mean", "ttft_p50", "ttft_p95"]:
            vals = [s[key] for s in stats_list if np.isfinite(s[key])]
            row[key] = statistics.mean(vals) if vals else float("nan")
            ci_key = key.split("_")[0] + "_ci"  # tpot_ci or ttft_ci
            if ci_key not in row:
                row[ci_key] = ci95(vals) if vals else 0.0
        row["n_steps_total"] = sum(s["n_steps"] for s in stats_list)
        agg.append(row)
    return agg


def aggregate_agentic_by_role(agentic_steps):
    grouped = defaultdict(list)
    for s in agentic_steps:
        grouped[(s["config"], s["concurrency"], s["agent_role"])].append(s["tpot_ms"])
    rows = []
    for (config, conc, role), tpots in sorted(grouped.items()):
        rows.append({"config": config, "concurrency": conc, "role": role,
                      "tpot_mean": statistics.mean(tpots), "n": len(tpots)})
    return rows


# ============================================================================
# Tables
# ============================================================================
def print_tables(batch_rows, agentic_agg, server_means):
    """Enhanced tables with winner (*), delta columns, and descriptions."""
    BATCH_EXCLUDE = {("TP4 H200 NVLink", 128)}

    batch_sizes = sorted(set(r["batch_size"] for r in batch_rows)) if batch_rows else []
    input_lens = sorted(set(r["input_len"] for r in batch_rows)) if batch_rows else []
    batch_cfgs = sorted(set(r["config"] for r in batch_rows))
    bl = {}
    for r in batch_rows:
        bl[(r["config"], r["input_len"], r["batch_size"])] = r
    al = {(r["config"], r["concurrency"]): r for r in agentic_agg}
    all_conc = sorted(set(r["concurrency"] for r in agentic_agg))
    agentic_cfgs = sorted(set(r["config"] for r in agentic_agg),
                          key=lambda c: ALL_CONFIGS.index(c) if c in ALL_CONFIGS else 99)
    ttft_batch_cfgs = [c for c in batch_cfgs if "TP8" in c]
    ttft_agen_cfgs = [c for c in agentic_cfgs if "TP8" in c or "H100" in c]

    def sh(name):
        return name.replace(" H200 NVLink", "").replace(" H100 PCIe", "(PCIe)")

    W = 110
    SEP = "=" * W

    # ------------------------------------------------------------------
    # Batch metric table helper
    # ------------------------------------------------------------------
    def batch_table(tag, key, fmt_s, lower, cfgs_filter=None, extra_note=""):
        cfgs_all = cfgs_filter if cfgs_filter else batch_cfgs
        better = "Lower" if lower else "Higher"
        print(f"\n{SEP}")
        print(f"{tag} — {better} is better   (*=winner)")
        if extra_note:
            print(f"  {extra_note}")
        print(SEP)

        win_tot = defaultdict(int)
        n_cols = 0
        worst_delta = ("", 0.0)  # (label, ratio)
        descs = []

        for il in input_lens:
            cfgs = [c for c in cfgs_all if (c, il) not in BATCH_EXCLUDE]
            if not cfgs:
                continue
            excl = [sh(c) for c in cfgs_all if (c, il) in BATCH_EXCLUDE]
            excl_s = f"  [{', '.join(excl)} excl.]" if excl else ""
            print(f"\n  [i={il}]{excl_s}")

            hdr = f"  {'Config':<24}"
            for b in batch_sizes:
                hdr += f"  b={b:<5}"
            hdr += "  Δ(1→64)"
            print(hdr)
            print(f"  {'─'*(len(hdr)-2)}")

            # per-column winner
            col_w = {}
            for b in batch_sizes:
                cands = [(c, bl.get((c, il, b), {}).get(key, 0)) for c in cfgs]
                cands = [(c, v) for c, v in cands if v > 0]
                if cands:
                    col_w[b] = (min if lower else max)(cands, key=lambda x: x[1])[0]

            for cfg in cfgs:
                row = f"  {cfg:<24}"
                for b in batch_sizes:
                    v = bl.get((cfg, il, b), {}).get(key, 0)
                    is_win = col_w.get(b) == cfg
                    if is_win:
                        win_tot[cfg] += 1
                        n_cols += 1
                    if v > 0:
                        row += f" {v:{fmt_s}}{'*' if is_win else ' '}"
                    else:
                        row += "      —  "
                # degradation / scale-up
                v1 = bl.get((cfg, il, batch_sizes[0]), {}).get(key, 0)
                vn = bl.get((cfg, il, batch_sizes[-1]), {}).get(key, 0)
                if v1 > 0 and vn > 0:
                    ratio = vn / v1  # always show b64/b1 ratio
                    row += f"  {ratio:>5.1f}x"
                    if lower and ratio > worst_delta[1]:
                        worst_delta = (f"{sh(cfg)} i={il}", ratio)
                else:
                    row += "     —"
                print(row)

            # winner row
            wrow = f"  {'→ Winner':<24}"
            for b in batch_sizes:
                w = col_w.get(b, "")
                wrow += f"  {sh(w):<6}" if w else "     —  "
            print(wrow)

            # per-input_len delta description
            if len(cfgs) >= 2:
                for b in batch_sizes:
                    cands = [(c, bl.get((c, il, b), {}).get(key, 0)) for c in cfgs]
                    cands = [(c, v) for c, v in cands if v > 0]
                    if len(cands) >= 2:
                        s = sorted(cands, key=lambda x: x[1])
                        best_v, worst_v = s[0][1], s[-1][1]
                        gap = worst_v - best_v if lower else best_v - worst_v
                        gap_pct = gap / best_v * 100 if best_v > 0 else 0
                        if gap_pct > 50:
                            descs.append(f"i={il} b={b}: {sh(s[-1 if lower else 0][0])} "
                                         f"is {gap_pct:.0f}% worse ({s[0][1]:.1f} vs {s[-1][1]:.1f})")

        # summary
        print(f"\n  {'─'*80}")
        if win_tot:
            top = max(win_tot.items(), key=lambda x: x[1])
            print(f"  → {sh(top[0])} wins {top[1]}/{n_cols} columns.")
        if worst_delta[0] and lower:
            print(f"  → Steepest degradation: {worst_delta[0]} ({worst_delta[1]:.1f}x from b=1→b=64).")
        # top 3 notable deltas
        for d in descs[:3]:
            print(f"  → {d}")
        # TP4 cold-start note
        if key == "gen_tps":
            v = bl.get(("TP4 H200 NVLink", 512, 1), {}).get("gen_tps", 0)
            if v and v < 20:
                print(f"  → Note: TP4 i=512 b=1 TPS={v:.0f} (TTFT=12.9s, cold-start artifact).")

    # ------------------------------------------------------------------
    # Batch TTFT special table (8GPU — constant across batch sizes)
    # ------------------------------------------------------------------
    def batch_ttft_table():
        if not ttft_batch_cfgs:
            return
        print(f"\n{SEP}")
        print("T3: Batch TTFT (ms) — 8GPU only   (*=winner per column)")
        print("  4GPU TTFT excluded (scheduling artifacts: 400-12000ms)")
        print("  TTFT is constant across batch sizes (single-request prefill latency)")
        print(SEP)
        hdr = f"  {'Config':<24}"
        for il in input_lens:
            hdr += f"  i={il:<5}"
        hdr += "  EP overhead"
        print(hdr)
        print(f"  {'─'*(len(hdr)-2)}")
        col_w = {}
        for il in input_lens:
            cands = [(c, bl.get((c, il, 1), {}).get("ttft_single_ms", 0)) for c in ttft_batch_cfgs]
            cands = [(c, v) for c, v in cands if v > 0]
            if cands:
                col_w[il] = min(cands, key=lambda x: x[1])[0]
        for cfg in ttft_batch_cfgs:
            row = f"  {cfg:<24}"
            for il in input_lens:
                v = bl.get((cfg, il, 1), {}).get("ttft_single_ms", 0)
                is_win = col_w.get(il) == cfg
                row += f" {v:>7.1f}{'*' if is_win else ' '}" if v > 0 else "      —  "
            # EP overhead column
            if "EP" in cfg:
                base_cfg = cfg.replace("+EP ", " ")
                vals = []
                for il in input_lens:
                    ep_v = bl.get((cfg, il, 1), {}).get("ttft_single_ms", 0)
                    base_v = bl.get((base_cfg, il, 1), {}).get("ttft_single_ms", 0)
                    if ep_v > 0 and base_v > 0:
                        vals.append(ep_v - base_v)
                if vals:
                    row += f"  +{statistics.mean(vals):.1f}ms avg"
                else:
                    row += "     —"
            else:
                row += "  (baseline)"
            print(row)
        wrow = f"  {'→ Winner':<24}"
        for il in input_lens:
            w = col_w.get(il, "")
            wrow += f"  {sh(w):<6}" if w else "     —  "
        print(wrow)
        print(f"\n  {'─'*80}")
        print(f"  → TP8 has lower TTFT at all input lengths (no EP routing overhead).")
        for il in input_lens:
            tp8 = bl.get(("TP8 H200 NVLink", il, 1), {}).get("ttft_single_ms", 0)
            ep = bl.get(("TP8+EP H200 NVLink", il, 1), {}).get("ttft_single_ms", 0)
            if tp8 > 0 and ep > 0:
                print(f"  → i={il}: EP adds +{ep-tp8:.1f}ms ({(ep-tp8)/tp8*100:.0f}% overhead).")

    # ------------------------------------------------------------------
    # Agentic metric table helper
    # ------------------------------------------------------------------
    def agentic_table(tag, key, fmt_s, lower, cfgs_filter=None, extra_note=""):
        cfgs_use = cfgs_filter if cfgs_filter else agentic_cfgs
        concs = sorted(set(r["concurrency"] for r in agentic_agg if r["config"] in cfgs_use))
        if not concs:
            return
        better = "Lower" if lower else "Higher"
        print(f"\n{SEP}")
        print(f"{tag} — {better} is better   (*=winner)")
        if extra_note:
            print(f"  {extra_note}")
        print(SEP)

        hdr = f"  {'Config':<24}"
        for c in concs:
            hdr += f"  c={c:<5}"
        hdr += "  Δ(min→max)"
        print(hdr)
        print(f"  {'─'*(len(hdr)-2)}")

        col_w = {}
        for c in concs:
            cands = [(cfg, al.get((cfg, c), {}).get(key, 0)) for cfg in cfgs_use]
            cands = [(cfg, v) for cfg, v in cands if v and v > 0 and np.isfinite(v)]
            if cands:
                col_w[c] = (min if lower else max)(cands, key=lambda x: x[1])[0]

        worst_delta = ("", 0.0)
        for cfg in cfgs_use:
            row = f"  {cfg:<24}"
            for c in concs:
                v = al.get((cfg, c), {}).get(key, 0)
                is_win = col_w.get(c) == cfg
                if v and v > 0 and np.isfinite(v):
                    row += f" {v:{fmt_s}}{'*' if is_win else ' '}"
                else:
                    row += "      —  "
            # degradation
            vals_cfg = [al.get((cfg, c), {}).get(key, 0) for c in concs]
            vals_cfg = [v for v in vals_cfg if v and v > 0 and np.isfinite(v)]
            if len(vals_cfg) >= 2:
                ratio = max(vals_cfg) / min(vals_cfg) if min(vals_cfg) > 0 else 0
                row += f"  {ratio:>5.1f}x"
                if ratio > worst_delta[1]:
                    worst_delta = (sh(cfg), ratio)
            else:
                row += "     —"
            print(row)

        wrow = f"  {'→ Winner':<24}"
        for c in concs:
            w = col_w.get(c, "")
            wrow += f"  {sh(w):<6}" if w else "     —  "
        print(wrow)

        # descriptions
        print(f"\n  {'─'*80}")
        if col_w:
            counts = defaultdict(int)
            for w in col_w.values():
                counts[w] += 1
            top = max(counts.items(), key=lambda x: x[1])
            print(f"  → {sh(top[0])} wins {top[1]}/{len(col_w)} columns.")
        if worst_delta[0]:
            lbl = "degradation" if lower else "range"
            print(f"  → Max {lbl}: {worst_delta[0]} ({worst_delta[1]:.1f}x).")
        # crossover / transition
        for i in range(1, len(concs)):
            w_prev = col_w.get(concs[i - 1])
            w_curr = col_w.get(concs[i])
            if w_prev and w_curr and w_prev != w_curr:
                print(f"  → Winner transition at c={concs[i]}: {sh(w_prev)} → {sh(w_curr)}.")

    # ------------------------------------------------------------------
    # Agentic TPS table (from server_means)
    # ------------------------------------------------------------------
    def agentic_tps_table():
        concs = all_conc
        cfgs_use = agentic_cfgs
        print(f"\n{SEP}")
        print("T5: Agentic TPS — Higher is better   (*=winner)")
        print(SEP)

        hdr = f"  {'Config':<24}"
        for c in concs:
            hdr += f"  c={c:<5}"
        hdr += "   Peak"
        print(hdr)
        print(f"  {'─'*(len(hdr)-2)}")

        col_w = {}
        for c in concs:
            cands = [(cfg, server_means.get(cfg, {}).get(c, {}).get("gen_tps", 0)) for cfg in cfgs_use]
            cands = [(cfg, v) for cfg, v in cands if v and v > 0]
            if cands:
                col_w[c] = max(cands, key=lambda x: x[1])[0]

        for cfg in cfgs_use:
            sm = server_means.get(cfg, {})
            row = f"  {cfg:<24}"
            vals = []
            for c in concs:
                v = sm.get(c, {}).get("gen_tps", 0)
                is_win = col_w.get(c) == cfg
                if v and v > 0:
                    row += f" {v:>6.0f}{'*' if is_win else ' '}"
                    vals.append((c, v))
                else:
                    row += "      —  "
            if vals:
                peak_c, peak_v = max(vals, key=lambda x: x[1])
                row += f"  {peak_v:>6.0f}"
            print(row)

        wrow = f"  {'→ Winner':<24}"
        for c in concs:
            w = col_w.get(c, "")
            wrow += f"  {sh(w):<6}" if w else "     —  "
        print(wrow)

        # descriptions
        print(f"\n  {'─'*80}")
        if col_w:
            counts = defaultdict(int)
            for w in col_w.values():
                counts[w] += 1
            top = max(counts.items(), key=lambda x: x[1])
            print(f"  → {sh(top[0])} wins {top[1]}/{len(col_w)} columns.")
        for cfg in cfgs_use:
            sm = server_means.get(cfg, {})
            vals = [(c, sm.get(c, {}).get("gen_tps", 0)) for c in concs]
            vals = [(c, v) for c, v in vals if v > 0]
            if len(vals) >= 2:
                peak_c, peak_v = max(vals, key=lambda x: x[1])
                last_c, last_v = vals[-1]
                if last_v < peak_v * 0.85 and last_c != peak_c:
                    print(f"  → {sh(cfg)}: peaks c={peak_c} ({peak_v:.0f}), drops {(1-last_v/peak_v)*100:.0f}% at c={last_c}.")

    # ------------------------------------------------------------------
    # Execute all tables
    # ------------------------------------------------------------------
    tp_only_cfgs = [c for c in batch_cfgs if "DP" not in c]
    batch_table("T1: Batch TPOT (ms)", "tpot_mean_ms", ">7.2f", True, cfgs_filter=tp_only_cfgs)
    batch_table("T2: Batch TPS", "gen_tps", ">7.0f", False, cfgs_filter=tp_only_cfgs)
    batch_ttft_table()
    agentic_table("T4: Agentic TPOT (ms)", "tpot_mean", ">7.2f", True)
    agentic_tps_table()
    agentic_table("T6: Agentic TTFT (ms)", "ttft_mean", ">7.1f", True,
                  cfgs_filter=ttft_agen_cfgs,
                  extra_note="4GPU H200 TTFT excluded (scheduling artifacts)")

    # T7: Platform comparison
    print(f"\n{SEP}")
    print("T7: Platform — TP4 H200 NVLink vs TP4 H100 PCIe")
    print(SEP)
    ic_pair = ("TP4 H200 NVLink", "TP4 H100 PCIe")
    common = sorted(
        set(r["concurrency"] for r in agentic_agg if r["config"] == ic_pair[0])
        & set(r["concurrency"] for r in agentic_agg if r["config"] == ic_pair[1]))
    if common:
        hdr = f"  {'Metric':<24}"
        for c in common:
            hdr += f"  c={c:<5}"
        print(hdr)
        print(f"  {'─'*(len(hdr)-2)}")
        for lbl, cfg, key in [("H200 NVLink TPOT(ms)", ic_pair[0], "tpot_mean"),
                               ("H100 PCIe TPOT(ms)", ic_pair[1], "tpot_mean")]:
            row = f"  {lbl:<24}"
            for c in common:
                r = al.get((cfg, c))
                row += f" {r[key]:>7.1f}" if r and np.isfinite(r[key]) else "     —  "
            print(row)
        # delta row
        row = f"  {'Δ H100−H200 (ms)':<24}"
        for c in common:
            h2 = al.get((ic_pair[0], c))
            h1 = al.get((ic_pair[1], c))
            if h2 and h1 and np.isfinite(h2["tpot_mean"]) and np.isfinite(h1["tpot_mean"]):
                d = h1["tpot_mean"] - h2["tpot_mean"]
                row += f" {d:>+7.1f}"
            else:
                row += "     —  "
        print(row)
        print()
        for lbl, cfg, key in [("H200 NVLink TPS", ic_pair[0], "gen_tps"),
                               ("H100 PCIe TPS", ic_pair[1], "gen_tps")]:
            sm = server_means.get(cfg, {})
            row = f"  {lbl:<24}"
            for c in common:
                v = sm.get(c, {}).get("gen_tps", 0)
                row += f" {v:>7.0f}" if v > 0 else "     —  "
            print(row)
        row = f"  {'Δ H200−H100 TPS':<24}"
        for c in common:
            v200 = server_means.get(ic_pair[0], {}).get(c, {}).get("gen_tps", 0)
            v100 = server_means.get(ic_pair[1], {}).get(c, {}).get("gen_tps", 0)
            if v200 > 0 and v100 > 0:
                row += f" {v200-v100:>+7.0f}"
            else:
                row += "     —  "
        print(row)

        print(f"\n  {'─'*80}")
        # description
        for c in common:
            h2_t = al.get((ic_pair[0], c), {}).get("tpot_mean", 0)
            h1_t = al.get((ic_pair[1], c), {}).get("tpot_mean", 0)
            if h2_t > 0 and h1_t > 0:
                speedup = h1_t / h2_t
                v200 = server_means.get(ic_pair[0], {}).get(c, {}).get("gen_tps", 0)
                v100 = server_means.get(ic_pair[1], {}).get(c, {}).get("gen_tps", 0)
                tps_gain = ((v200 - v100) / v100 * 100) if v100 > 0 else 0
                print(f"  → c={c}: H200 is {speedup:.1f}x faster TPOT"
                      f"{f', +{tps_gain:.0f}% TPS' if tps_gain else ''}.")

    # ------------------------------------------------------------------
    # T8: DP Batch TPOT (same format as T1 but DP configs only)
    # ------------------------------------------------------------------
    dp_cfgs = [c for c in batch_cfgs if "DP" in c]
    if dp_cfgs:
        batch_table("T8: DP Batch TPOT (ms)", "tpot_mean_ms", ">7.2f", True,
                     cfgs_filter=dp_cfgs,
                     extra_note="DP configs only. TP2-DP4-EP i=128 b=8 excluded (anomaly).")

    # ------------------------------------------------------------------
    # T9: TP vs DP Comparison (ratio table)
    # ------------------------------------------------------------------
    dp_compare = [("TP8 H200 NVLink", "TP4-DP2 H200 NVLink", "TP4-DP2/TP8"),
                  ("TP8 H200 NVLink", "TP2-DP4-EP H200 NVLink", "TP2-DP4-EP/TP8")]
    dp_compare = [(base, dp, lbl) for base, dp, lbl in dp_compare
                  if base in batch_cfgs and dp in batch_cfgs]
    if dp_compare:
        print(f"\n{SEP}")
        print("T9: TP vs DP — TPOT Ratio (DP / TP8, >1 = DP slower)")
        print(SEP)
        for il in input_lens:
            print(f"\n  [i={il}]")
            hdr = f"  {'Comparison':<24}"
            for b in batch_sizes:
                hdr += f"  b={b:<5}"
            print(hdr)
            print(f"  {'─'*(len(hdr)-2)}")
            for base_cfg, dp_cfg, lbl in dp_compare:
                row = f"  {lbl:<24}"
                for b in batch_sizes:
                    base_v = bl.get((base_cfg, il, b), {}).get("tpot_mean_ms", 0)
                    dp_v = bl.get((dp_cfg, il, b), {}).get("tpot_mean_ms", 0)
                    if base_v > 0 and dp_v > 0:
                        ratio = dp_v / base_v
                        row += f" {ratio:>6.2f}x"
                    else:
                        row += "      —  "
                print(row)
        print(f"\n  {'─'*80}")
        print(f"  → Ratio >1 = DP is slower than pure TP8.")
        print(f"  → Ratio ~1 = equivalent per-request TPOT (DP gains throughput via parallelism).")


# ============================================================================
# Figures (10 comparison-focused)
# ============================================================================
def _add_value_labels(ax, bars, fmt=".1f", fontsize=7, rotation=0, threshold=100):
    """Disabled — exact values shown in tables instead."""
    pass


def _annotate_saturation(ax, batch_sizes, vals_by_cfg, cfg_colors):
    """Mark saturation: where TPS gain < 15% from previous point."""
    for cfg, vals in vals_by_cfg.items():
        for i in range(1, len(vals)):
            if vals[i - 1] > 0 and vals[i] > 0:
                gain = (vals[i] - vals[i - 1]) / vals[i - 1]
                if gain < 0.15 and i >= 2:  # saturating after at least 2 increases
                    ax.annotate("sat.", xy=(i, vals[i]), fontsize=6,
                                color=cfg_colors.get(cfg, "gray"), alpha=0.7,
                                ha="center", va="bottom",
                                xytext=(0, 8), textcoords="offset points")
                    break


def _annotate_knee(ax, x_positions, vals_by_cfg, cfg_colors):
    """Mark knee: where TPOT increases > 80% from previous step."""
    for cfg, vals in vals_by_cfg.items():
        for i in range(1, len(vals)):
            if vals[i - 1] > 0 and vals[i] > 0:
                ratio = vals[i] / vals[i - 1]
                if ratio > 1.8:
                    ax.annotate(f"{ratio:.1f}x", xy=(x_positions[i], vals[i]),
                                fontsize=6, color=cfg_colors.get(cfg, "gray"),
                                ha="center", va="bottom",
                                xytext=(0, 6), textcoords="offset points")
                    break


def make_figures(batch_rows, agentic_agg, server_means, server_reps, out_dir):
    plt.rcParams.update(PLT_RC)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = sorted(set(r["batch_size"] for r in batch_rows)) if batch_rows else []
    input_lens = sorted(set(r["input_len"] for r in batch_rows)) if batch_rows else []
    batch_cfgs = sorted(set(r["config"] for r in batch_rows))
    # TP-only configs for F1-F10 (excludes DP configs)
    tp_only_batch_cfgs = [c for c in batch_cfgs if "DP" not in c]

    bl = defaultdict(dict)
    for r in batch_rows:
        bl[(r["config"], r["input_len"])][r["batch_size"]] = r
    al = {(r["config"], r["concurrency"]): r for r in agentic_agg}

    all_conc = sorted(set(r["concurrency"] for r in agentic_agg))
    agentic_cfgs = sorted(set(r["config"] for r in agentic_agg),
                          key=lambda c: ALL_CONFIGS.index(c) if c in ALL_CONFIGS else 99)

    # --- Pruning rules ---
    # TP4 H200 i=128 is anomalous (CUDA graph issue); exclude from batch plots
    BATCH_EXCLUDE = {("TP4 H200 NVLink", 128)}
    # 4GPU TTFT dominated by scheduling artifacts (400-12000ms); 8GPU is clean
    TTFT_CONFIGS_8GPU = [c for c in batch_cfgs if "TP8" in c]
    AGENTIC_TTFT_CFGS = [c for c in agentic_cfgs if "TP8" in c]
    # H200-only configs for F4/F5/F6 (H100 compared separately in F9)
    h200_agentic_cfgs = [c for c in agentic_cfgs if "H100" not in c]
    h200_conc = sorted(set(r["concurrency"] for r in agentic_agg if "H100" not in r["config"]))
    # Filtered input_lens for batch plots (per config)
    def batch_input_lens(cfg):
        return [il for il in input_lens if (cfg, il) not in BATCH_EXCLUDE]

    # ================================================================
    # F1 — Batch TPOT (TP-only configs, all input lengths)
    # ================================================================
    fig, axes = plt.subplots(1, len(input_lens), figsize=(5.5 * len(input_lens), 5.2), squeeze=False)
    axes = axes[0]
    nc = len(tp_only_batch_cfgs)
    for ax, il in zip(axes, input_lens):
        x = np.arange(len(batch_sizes))
        w = 0.8 / max(nc, 1)
        for ci, cfg in enumerate(tp_only_batch_cfgs):
            vals = [bl.get((cfg, il), {}).get(b, {}).get("tpot_mean_ms", 0) for b in batch_sizes]
            bars = ax.bar(x + ci * w, vals, w, label=cfg, color=ccol(cfg), alpha=0.85)
        ax.set_xticks(x + w * nc / 2)
        ax.set_xticklabels([str(b) for b in batch_sizes])
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("TPOT (ms)")
        ax.set_title(f"Input Length = {il}")
        ax.set_ylim(0, 30)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("F1: Batch TPOT", fontweight="bold", y=1.02)
    fig.text(0.5, -0.02, "TP4 i=128 shows CUDA graph anomaly (b=32 spike). All configs shown for consistency.",
             ha="center", fontsize=9, style="italic", color="gray")
    fig.tight_layout()
    fig.savefig(fig_dir / "F1_batch_tpot.png")
    plt.close(fig)
    print("  F1")

    # ================================================================
    # F2 — Batch TPS (TP-only configs, + saturation annotation)
    # ================================================================
    fig, axes = plt.subplots(1, len(input_lens), figsize=(5.5 * len(input_lens), 5.2), squeeze=False)
    axes = axes[0]
    nc = len(tp_only_batch_cfgs)
    for ax, il in zip(axes, input_lens):
        x = np.arange(len(batch_sizes))
        w = 0.8 / max(nc, 1)
        cfg_vals = {}
        for ci, cfg in enumerate(tp_only_batch_cfgs):
            vals = [bl.get((cfg, il), {}).get(b, {}).get("gen_tps", 0) for b in batch_sizes]
            cfg_vals[cfg] = vals
            bars = ax.bar(x + ci * w, vals, w, label=cfg, color=ccol(cfg), alpha=0.85)
        _annotate_saturation(ax, batch_sizes, cfg_vals, {c: ccol(c) for c in tp_only_batch_cfgs})
        ax.set_xticks(x + w * nc / 2)
        ax.set_xticklabels([str(b) for b in batch_sizes])
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Generation TPS")
        ax.set_title(f"Input Length = {il}")
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("F2: Batch TPS", fontweight="bold", y=1.02)
    fig.text(0.5, -0.02, "TP4 i=128 shows anomalous TPS pattern. 'sat.' = <15% gain from previous batch.",
             ha="center", fontsize=9, style="italic", color="gray")
    fig.tight_layout()
    fig.savefig(fig_dir / "F2_batch_tps.png")
    plt.close(fig)
    print("  F2")

    # ================================================================
    # F3 — Batch TTFT (8GPU, b=64 fixed, input_length on x-axis)
    # ================================================================
    if TTFT_CONFIGS_8GPU:
        fix_b = 64
        fig, ax = plt.subplots(figsize=(8, 5.5))
        x = np.arange(len(input_lens))
        nc = len(TTFT_CONFIGS_8GPU)
        w = 0.8 / max(nc, 1)
        for ci, cfg in enumerate(TTFT_CONFIGS_8GPU):
            vals = [bl.get((cfg, il), {}).get(fix_b, {}).get("ttft_single_ms", 0) for il in input_lens]
            ax.bar(x + ci * w, vals, w, label=cfg, color=ccol(cfg), alpha=0.85)
        ax.set_xticks(x + w * nc / 2)
        ax.set_xticklabels([str(il) for il in input_lens])
        ax.set_xlabel("Input Length")
        ax.set_ylabel("TTFT (ms)")
        ax.set_title("F3: Batch TTFT — 8GPU (b=64 fixed)")
        ax.legend(fontsize=8)
        fig.text(0.5, -0.02, "TTFT independent of batch size. Fixed b=64. 4GPU excluded (scheduling artifacts). EP adds +2.6ms (i=128) to +57ms (i≥512).",
                 ha="center", fontsize=9, style="italic", color="gray")
        fig.tight_layout()
        fig.savefig(fig_dir / "F3_batch_ttft.png")
        plt.close(fig)
    print("  F3")

    # ================================================================
    # F4 — Agentic TPOT (H200 only, CI error bars + knee)
    # ================================================================
    fig, ax = plt.subplots(figsize=(max(10, len(h200_conc) * 2.2), 6))
    x = np.arange(len(h200_conc))
    nc = len(h200_agentic_cfgs)
    w = 0.8 / nc
    cfg_vals = {}
    for ci, cfg in enumerate(h200_agentic_cfgs):
        vals, errs = [], []
        for c in h200_conc:
            r = al.get((cfg, c))
            if r and np.isfinite(r["tpot_mean"]):
                vals.append(r["tpot_mean"])
                errs.append(r.get("tpot_ci", 0))
            else:
                vals.append(0)
                errs.append(0)
        cfg_vals[cfg] = vals
        bars = ax.bar(x + ci * w, vals, w, yerr=errs, capsize=2,
                       label=cfg, color=ccol(cfg), alpha=0.85,
                       error_kw={"linewidth": 0.8})
    _annotate_knee(ax, x + w * nc / 2, cfg_vals, {c: ccol(c) for c in h200_agentic_cfgs})
    ax.set_xticks(x + w * nc / 2)
    ax.set_xticklabels([f"c={c}" for c in h200_conc])
    ax.set_ylabel("TPOT Mean (ms)")
    ax.set_title("F4: Agentic TPOT (H200)")
    ax.legend(fontsize=8)
    fig.text(0.5, -0.02, "H200 configs only. H100 compared in F9. Error bars = 95% CI (n=3 repeats).",
             ha="center", fontsize=9, style="italic", color="gray")
    fig.tight_layout()
    fig.savefig(fig_dir / "F4_agentic_tpot.png")
    plt.close(fig)
    print("  F4")

    # ================================================================
    # F5 — Agentic TPS (H200 only, CI + peak annotation)
    # ================================================================
    fig, ax = plt.subplots(figsize=(max(10, len(h200_conc) * 2.2), 6))
    x = np.arange(len(h200_conc))
    nc = len(h200_agentic_cfgs)
    w = 0.8 / nc
    for ci, cfg in enumerate(h200_agentic_cfgs):
        sm = server_means.get(cfg, {})
        sr = server_reps.get(cfg, {})
        vals, errs = [], []
        for c in h200_conc:
            v = sm.get(c, {}).get("gen_tps", 0)
            vals.append(v if v else 0)
            reps = [rr["gen_tps"] for rr in sr.get(c, [])] if sr.get(c) else []
            errs.append(ci95(reps) if len(reps) >= 2 else 0)
        bars = ax.bar(x + ci * w, vals, w, yerr=errs, capsize=2,
                       label=cfg, color=ccol(cfg), alpha=0.85,
                       error_kw={"linewidth": 0.8})
        # Annotate peak → flat/drop
        peak_i = np.argmax(vals)
        if peak_i < len(vals) - 1 and vals[peak_i] > 0:
            drop = vals[peak_i + 1] / vals[peak_i] if vals[peak_i] > 0 else 1
            if drop < 0.95:
                ax.annotate("peak", xy=(x[peak_i] + ci * w + w / 2, vals[peak_i]),
                            fontsize=6, color=ccol(cfg), ha="center", va="bottom",
                            xytext=(0, 10), textcoords="offset points")
    ax.set_xticks(x + w * nc / 2)
    ax.set_xticklabels([f"c={c}" for c in h200_conc])
    ax.set_ylabel("Generation TPS")
    ax.set_title("F5: Agentic TPS (H200)")
    ax.legend(fontsize=8)
    fig.text(0.5, -0.02, "H200 configs only. Error bars = 95% CI. 'peak' = TPS drop after this concurrency.",
             ha="center", fontsize=9, style="italic", color="gray")
    fig.tight_layout()
    fig.savefig(fig_dir / "F5_agentic_tps.png")
    plt.close(fig)
    print("  F5")

    # ================================================================
    # F6 — Agentic TTFT (all H200 configs, log scale)
    # ================================================================
    if h200_agentic_cfgs:
        ttft_concs = sorted(set(r["concurrency"] for r in agentic_agg if r["config"] in h200_agentic_cfgs))
        fig, ax = plt.subplots(figsize=(max(10, len(ttft_concs) * 2.2), 6))
        x = np.arange(len(ttft_concs))
        nc = len(h200_agentic_cfgs)
        w = 0.8 / max(nc, 1)
        for ci, cfg in enumerate(h200_agentic_cfgs):
            vals, errs = [], []
            for c in ttft_concs:
                r = al.get((cfg, c))
                if r and np.isfinite(r["ttft_mean"]):
                    vals.append(r["ttft_mean"])
                    errs.append(r.get("ttft_ci", 0))
                else:
                    vals.append(0)
                    errs.append(0)
            bars = ax.bar(x + ci * w, vals, w, yerr=errs, capsize=2,
                           label=cfg, color=ccol(cfg), alpha=0.85,
                           error_kw={"linewidth": 0.8})
        ax.set_xticks(x + w * nc / 2)
        ax.set_xticklabels([f"c={c}" for c in ttft_concs])
        ax.set_ylabel("TTFT Mean (ms)")
        ax.set_yscale("log")
        ax.set_title("F6: Agentic TTFT — All H200 Configs")
        ax.legend(fontsize=8)
        fig.text(0.5, -0.02, "Log scale — 4GPU TTFT (400-2000ms) has vLLM scheduling overhead. 8GPU TTFT (80-200ms) is clean. H100 in F9. CI=95%.",
                 ha="center", fontsize=9, style="italic", color="gray")
        fig.tight_layout()
        fig.savefig(fig_dir / "F6_agentic_ttft.png")
        plt.close(fig)
    print("  F6")

    # ================================================================
    # F7 — EP Batch: TPOT, TPS, TTFT (3 rows × 3 input_len cols)
    # ================================================================
    pair = ("TP8 H200 NVLink", "TP8+EP H200 NVLink")
    EP_BATCH_ANOMALY = {(128, 8)}  # i=128, b=8 is outlier for EP
    ep_metrics = [
        ("TPOT (ms)", "tpot_mean_ms"),
        ("Gen TPS", "gen_tps"),
        ("TTFT (ms)", "ttft_single_ms"),
    ]
    nrows = len(ep_metrics)
    ncols = len(input_lens)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)
    for ri, (ylabel, key) in enumerate(ep_metrics):
        for ci_ax, il in enumerate(input_lens):
            ax = axes[ri][ci_ax]
            bs_here = [b for b in batch_sizes if (il, b) not in EP_BATCH_ANOMALY]
            xp = np.arange(len(bs_here))
            w = 0.35
            for ci, cfg in enumerate(pair):
                vals = [bl.get((cfg, il), {}).get(b, {}).get(key, 0) for b in bs_here]
                bars = ax.bar(xp + ci * w, vals, w, label=cfg, color=ccol(cfg), alpha=0.85)
            ax.set_xticks(xp + w)
            ax.set_xticklabels([str(b) for b in bs_here])
            ax.set_xlabel("Batch Size")
            ax.set_ylabel(ylabel)
            title = f"i={il}"
            if any((il, b) in EP_BATCH_ANOMALY for b in batch_sizes):
                title += " (b=8 excl.)"
            ax.set_title(title)
            ax.legend(fontsize=7)
    fig.suptitle("F7: EP Effect — TP8 vs TP8+EP (Batch: TPOT, TPS, TTFT)", fontweight="bold", y=1.01)
    fig.text(0.5, -0.01, "i=128 b=8 excluded (anomalous EP spike). TTFT = single-request prefill latency.",
             ha="center", fontsize=9, style="italic", color="gray")
    fig.tight_layout()
    fig.savefig(fig_dir / "F7_ep_batch.png")
    plt.close(fig)
    print("  F7")

    # ================================================================
    # F7b — EP Penalty Ratio (3 rows: TPOT, TPS, TTFT)
    # ================================================================
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.0 * nrows), squeeze=False)
    for ri, (ylabel, key) in enumerate(ep_metrics):
        higher_better = "TPS" in ylabel
        for ci_ax, il in enumerate(input_lens):
            ax = axes[ri][ci_ax]
            bs_here = [b for b in batch_sizes if (il, b) not in EP_BATCH_ANOMALY]
            ratios = []
            for b in bs_here:
                tp8v = bl.get((pair[0], il), {}).get(b, {}).get(key, 0)
                epv = bl.get((pair[1], il), {}).get(b, {}).get(key, 0)
                ratios.append(epv / tp8v if tp8v > 0 else 0)
            # For TPOT/TTFT: >1 = EP penalty (red), <1 = EP benefit (green)
            # For TPS: >1 = EP benefit (green), <1 = EP penalty (red)
            if higher_better:
                colors_r = ["#4CAF50" if r > 1.05 else "#D32F2F" if r < 0.95 else "#757575" for r in ratios]
            else:
                colors_r = ["#D32F2F" if r > 1.05 else "#4CAF50" if r < 0.95 else "#757575" for r in ratios]
            xi = np.arange(len(bs_here))
            bars = ax.bar(xi, ratios, color=colors_r, alpha=0.85, edgecolor="white", linewidth=0.5)
            ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1)
            ax.set_xticks(xi)
            ax.set_xticklabels([str(b) for b in bs_here])
            ax.set_xlabel("Batch Size")
            ax.set_ylabel(f"EP/TP8 {ylabel}")
            title = f"i={il}"
            if any((il, b) in EP_BATCH_ANOMALY for b in batch_sizes):
                title += " (b=8 excl.)"
            ax.set_title(title)
            ax.set_ylim(0.5, max(max(ratios) * 1.15, 1.3) if ratios else 1.5)
            # Annotate values
            for j, (bar, r) in enumerate(zip(bars, ratios)):
                if higher_better:
                    color = "green" if r > 1.05 else "red" if r < 0.95 else "gray"
                else:
                    color = "red" if r > 1.05 else "green" if r < 0.95 else "gray"
                ax.text(bar.get_x() + bar.get_width() / 2, r + 0.01,
                        f"{r:.2f}", ha="center", va="bottom", fontsize=8,
                        color=color, fontweight="bold" if abs(r - 1) > 0.1 else "normal")
    fig.suptitle("F7b: EP Penalty Ratio — TP8+EP / TP8 (Batch: TPOT, TPS, TTFT)", fontweight="bold", y=1.01)
    fig.text(0.5, -0.01, "TPOT/TTFT: Red>1 = EP penalty. TPS: Red<1 = EP penalty. Dashed = parity.",
             ha="center", fontsize=9, style="italic", color="gray")
    fig.tight_layout()
    fig.savefig(fig_dir / "F7b_ep_penalty_ratio.png")
    plt.close(fig)
    print("  F7b")

    # ================================================================
    # F8 — EP Agentic (3-subplot: TPOT, TPS, TTFT with CI)
    # ================================================================
    ep_concs = sorted(
        set(r["concurrency"] for r in agentic_agg if r["config"] == pair[0])
        & set(r["concurrency"] for r in agentic_agg if r["config"] == pair[1]))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    specs = [
        ("TPOT (ms)", lambda cfg, c: (al.get((cfg, c), {}).get("tpot_mean", 0),
                                       al.get((cfg, c), {}).get("tpot_ci", 0))),
        ("Gen TPS", lambda cfg, c: _sm_with_ci(cfg, c, "gen_tps", server_means, server_reps)),
        ("TTFT (ms)", lambda cfg, c: (al.get((cfg, c), {}).get("ttft_mean", 0),
                                       al.get((cfg, c), {}).get("ttft_ci", 0))),
    ]
    for ax, (ylabel, get_v) in zip(axes, specs):
        xp = np.arange(len(ep_concs))
        w = 0.35
        for ci, cfg in enumerate(pair):
            vals, errs = zip(*[get_v(cfg, c) for c in ep_concs])
            vals = [v if np.isfinite(v) and v else 0 for v in vals]
            errs = [e if np.isfinite(e) else 0 for e in errs]
            bars = ax.bar(xp + ci * w, vals, w, yerr=errs, capsize=2,
                           label=cfg, color=ccol(cfg), alpha=0.85, error_kw={"linewidth": 0.8})
            _add_value_labels(ax, bars, fmt=".1f" if max(vals) < 100 else ".0f",
                              threshold=100, rotation=25)
        # Winner transition marker
        for i in range(1, len(ep_concs)):
            v0 = [get_v(cfg, ep_concs[i - 1])[0] for cfg in pair]
            v1 = [get_v(cfg, ep_concs[i])[0] for cfg in pair]
            if all(v > 0 for v in v0 + v1):
                prev_better = 0 if v0[0] < v0[1] else 1
                curr_better = 0 if v1[0] < v1[1] else 1
                # For TPS, higher is better
                if "TPS" in ylabel:
                    prev_better = 0 if v0[0] > v0[1] else 1
                    curr_better = 0 if v1[0] > v1[1] else 1
                if prev_better != curr_better:
                    ax.axvline(x=i - 0.5 + w, color="red", linestyle=":", alpha=0.5)
        ax.set_xticks(xp + w)
        ax.set_xticklabels([f"c={c}" for c in ep_concs])
        ax.set_ylabel(ylabel)
        if ax == axes[0]:
            ax.legend(fontsize=7)
    fig.suptitle("F8: EP Effect — Agentic (TPOT / TPS / TTFT)", fontweight="bold")
    fig.text(0.5, -0.02, "EP impact differs by metric and concurrency. Red dotted = winner transition. CI=95%.",
             ha="center", fontsize=9, style="italic", color="gray")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(fig_dir / "F8_ep_agentic.png")
    plt.close(fig)
    print("  F8")

    # ================================================================
    # F9 — Platform Comparison (TP4 H200 NVLink vs TP4 H100 PCIe)
    #      TTFT: min-max range bars instead of mean+CI
    # ================================================================
    ic_pair = ("TP4 H200 NVLink", "TP4 H100 PCIe")
    ic_concs = sorted(
        set(r["concurrency"] for r in agentic_agg if r["config"] == ic_pair[0])
        & set(r["concurrency"] for r in agentic_agg if r["config"] == ic_pair[1]))
    if ic_concs:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        # --- TPOT subplot ---
        ax = axes[0]
        xp = np.arange(len(ic_concs))
        w = 0.35
        for ci, cfg in enumerate(ic_pair):
            vals = [al.get((cfg, c), {}).get("tpot_mean", 0) for c in ic_concs]
            errs = [al.get((cfg, c), {}).get("tpot_ci", 0) for c in ic_concs]
            vals = [v if np.isfinite(v) and v else 0 for v in vals]
            errs = [e if np.isfinite(e) else 0 for e in errs]
            ax.bar(xp + ci * w, vals, w, yerr=errs, capsize=2,
                   label=cfg, color=ccol(cfg), alpha=0.85, error_kw={"linewidth": 0.8})
        ax.set_xticks(xp + w); ax.set_xticklabels([f"c={c}" for c in ic_concs])
        ax.set_ylabel("TPOT (ms)"); ax.legend(fontsize=7)

        # --- TPS subplot ---
        ax = axes[1]
        for ci, cfg in enumerate(ic_pair):
            vals_errs = [_sm_with_ci(cfg, c, "gen_tps", server_means, server_reps) for c in ic_concs]
            vals = [v for v, _ in vals_errs]
            errs = [e for _, e in vals_errs]
            ax.bar(xp + ci * w, vals, w, yerr=errs, capsize=2,
                   label=cfg, color=ccol(cfg), alpha=0.85, error_kw={"linewidth": 0.8})
        ax.set_xticks(xp + w); ax.set_xticklabels([f"c={c}" for c in ic_concs])
        ax.set_ylabel("Gen TPS"); ax.legend(fontsize=7)

        # --- TTFT subplot (min-max range) ---
        ax = axes[2]
        for ci, cfg in enumerate(ic_pair):
            means, err_lo, err_hi = [], [], []
            for c in ic_concs:
                r = al.get((cfg, c))
                mean_v = r.get("ttft_mean", 0) if r else 0
                # Get per-repeat TTFT for min-max from server_reps
                sr = server_reps.get(cfg, {})
                reps = [rr.get("ttft_mean_ms", 0) for rr in sr.get(c, [])] if sr.get(c) else []
                reps = [v for v in reps if v > 0]
                if reps and mean_v > 0:
                    mn, mx = min(reps), max(reps)
                    means.append(mean_v)
                    err_lo.append(max(mean_v - mn, 0))
                    err_hi.append(max(mx - mean_v, 0))
                elif mean_v > 0:
                    means.append(mean_v)
                    err_lo.append(0)
                    err_hi.append(0)
                else:
                    means.append(0)
                    err_lo.append(0)
                    err_hi.append(0)
            ax.bar(xp + ci * w, means, w,
                   yerr=[err_lo, err_hi], capsize=3,
                   label=cfg, color=ccol(cfg), alpha=0.85,
                   error_kw={"linewidth": 0.8})
        ax.set_xticks(xp + w); ax.set_xticklabels([f"c={c}" for c in ic_concs])
        ax.set_ylabel("TTFT (ms)")
        ax.legend(fontsize=7)
        ax.text(0.5, 0.97, "Error bars = min–max range",
                transform=ax.transAxes, fontsize=7, ha="center", va="top",
                color="gray", style="italic")

        fig.suptitle("F9: Platform Comparison — TP4 H200 NVLink vs TP4 H100 PCIe", fontweight="bold")
        fig.text(0.5, -0.02, "TPOT/TPS: mean±95%CI. TTFT: mean with min–max range. H200 4GPU TTFT includes scheduling overhead.",
                 ha="center", fontsize=9, style="italic", color="gray")
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        fig.savefig(fig_dir / "F9_platform_comparison.png")
        plt.close(fig)
        print("  F9")

    # ================================================================
    # F10 — Heatmap (TP-only, excludes TP4 i=128 anomaly)
    # ================================================================
    ncfg = len(tp_only_batch_cfgs)
    fig, axes = plt.subplots(1, ncfg, figsize=(4.5 * ncfg, 3.8), squeeze=False)
    axes = axes[0]
    # Global vmin/vmax — exclude anomalous cells, TP-only
    clean_tpot = [r["tpot_mean_ms"] for r in batch_rows
                  if (r["config"], r["input_len"]) not in BATCH_EXCLUDE and "DP" not in r["config"]]
    vmin, vmax = (min(clean_tpot), max(clean_tpot)) if clean_tpot else (0, 1)
    for ax, cfg in zip(axes, tp_only_batch_cfgs):
        cfg_ilens = [il for il in input_lens if (cfg, il) not in BATCH_EXCLUDE]
        matrix = np.full((len(cfg_ilens), len(batch_sizes)), np.nan)
        for i, il in enumerate(cfg_ilens):
            for j, b in enumerate(batch_sizes):
                r = bl.get((cfg, il), {}).get(b)
                if r:
                    matrix[i, j] = r["tpot_mean_ms"]
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", origin="upper",
                        vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(batch_sizes)))
        ax.set_xticklabels([str(b) for b in batch_sizes])
        ax.set_yticks(range(len(cfg_ilens)))
        ax.set_yticklabels([str(il) for il in cfg_ilens])
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Input Len")
        ax.set_title(cfg, fontsize=10)
        med = np.nanmedian(matrix)
        for i in range(len(cfg_ilens)):
            for j in range(len(batch_sizes)):
                v = matrix[i, j]
                if np.isfinite(v):
                    is_hot = v > vmin + 0.8 * (vmax - vmin)
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                            color="white" if v > med else "black",
                            fontweight="bold" if is_hot else "normal")
        plt.colorbar(im, ax=ax, shrink=0.8, label="ms")
    fig.suptitle("F10: Batch TPOT Heatmap", fontweight="bold", y=1.02)
    fig.text(0.5, -0.02, "TP4 i=128 excluded (CUDA graph anomaly). Shared color scale. Hotspots = top 20%.",
             ha="center", fontsize=9, style="italic", color="gray")
    fig.tight_layout()
    fig.savefig(fig_dir / "F10_heatmap.png")
    plt.close(fig)
    print("  F10")

    # ================================================================
    # F11 — TP vs DP Batch: TPOT, TPS, TTFT (3 rows × 3 input_len cols)
    # ================================================================
    dp_compare_cfgs = ["TP8 H200 NVLink", "TP4-DP2 H200 NVLink", "TP2-DP4-EP H200 NVLink"]
    dp_compare_cfgs = [c for c in dp_compare_cfgs if c in batch_cfgs]
    if len(dp_compare_cfgs) >= 2:
        f11_metrics = [
            ("TPOT (ms)", "tpot_mean_ms"),
            ("Gen TPS", "gen_tps"),
            ("TTFT (ms)", "ttft_single_ms"),
        ]
        nrows = len(f11_metrics)
        ncols = len(input_lens)
        nc = len(dp_compare_cfgs)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)
        for ri, (ylabel, key) in enumerate(f11_metrics):
            for ci_ax, il in enumerate(input_lens):
                ax = axes[ri][ci_ax]
                x = np.arange(len(batch_sizes))
                w = 0.8 / max(nc, 1)
                for ci, cfg in enumerate(dp_compare_cfgs):
                    vals = [bl.get((cfg, il), {}).get(b, {}).get(key, 0) for b in batch_sizes]
                    ax.bar(x + ci * w, vals, w, label=cfg, color=ccol(cfg), alpha=0.85)
                ax.set_xticks(x + w * nc / 2)
                ax.set_xticklabels([str(b) for b in batch_sizes])
                ax.set_xlabel("Batch Size")
                ax.set_ylabel(ylabel)
                ax.set_title(f"i={il}")
                ax.legend(fontsize=7, loc="upper left")
        fig.suptitle("F11: TP vs DP — Batch TPOT / TPS / TTFT (8 GPU)", fontweight="bold", y=1.01)
        fig.text(0.5, -0.01, "Same 8×H200 NVLink. TP8 = pure TP. TP4-DP2 = TP4×DP2. TP2-DP4-EP = TP2×DP4+EP. TTFT = single-request prefill latency.",
                 ha="center", fontsize=9, style="italic", color="gray")
        fig.tight_layout()
        fig.savefig(fig_dir / "F11_tp_vs_dp_tpot.png")
        plt.close(fig)
        print("  F11")

    # ================================================================
    # F12 — EP in DP Mode: TP4-DP2 vs TP4-DP2-EP (TPOT, TPS, TTFT)
    # ================================================================
    dp_ep_pair = ("TP4-DP2 H200 NVLink", "TP4-DP2-EP H200 NVLink")
    if all(c in batch_cfgs for c in dp_ep_pair):
        dp_ep_metrics = [
            ("TPOT (ms)", "tpot_mean_ms"),
            ("Gen TPS", "gen_tps"),
            ("TTFT (ms)", "ttft_single_ms"),
        ]
        nrows = len(dp_ep_metrics)
        ncols = len(input_lens)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)
        for ri, (ylabel, key) in enumerate(dp_ep_metrics):
            for ci_ax, il in enumerate(input_lens):
                ax = axes[ri][ci_ax]
                xp = np.arange(len(batch_sizes))
                w = 0.35
                for ci, cfg in enumerate(dp_ep_pair):
                    vals = [bl.get((cfg, il), {}).get(b, {}).get(key, 0) for b in batch_sizes]
                    ax.bar(xp + ci * w, vals, w, label=cfg, color=ccol(cfg), alpha=0.85)
                ax.set_xticks(xp + w)
                ax.set_xticklabels([str(b) for b in batch_sizes])
                ax.set_xlabel("Batch Size")
                ax.set_ylabel(ylabel)
                ax.set_title(f"i={il}")
                ax.legend(fontsize=7)
        fig.suptitle("F12: EP Effect in DP Mode — TP4-DP2 vs TP4-DP2-EP (Batch)", fontweight="bold", y=1.01)
        fig.text(0.5, -0.01, "Same F7 format but for DP configs. Both use 8×H200 NVLink with TP4×DP2.",
                 ha="center", fontsize=9, style="italic", color="gray")
        fig.tight_layout()
        fig.savefig(fig_dir / "F12_dp_ep_batch.png")
        plt.close(fig)
        print("  F12")

    return fig_dir


def _sm_with_ci(cfg, conc, key, server_means, server_reps):
    """Get server metric mean + CI."""
    sm = server_means.get(cfg, {})
    sr = server_reps.get(cfg, {})
    v = sm.get(conc, {}).get(key, 0)
    reps = [rr[key] for rr in sr.get(conc, [])] if sr.get(conc) else []
    return (v if v else 0, ci95(reps) if len(reps) >= 2 else 0)


# ============================================================================
# CSV Export
# ============================================================================
def export_csvs(batch_rows, agentic_agg, agentic_steps, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = out_dir / "h200_batch_ground_truth.csv"
    f1 = ["config", "input_len", "batch_size", "output_len",
          "tpot_mean_ms", "tpot_p50_ms", "ttft_single_ms", "ttft_mean_ms",
          "gen_tps", "sys_tpot_ms", "e2e_mean_ms"]
    with open(p1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=f1, extrasaction="ignore")
        w.writeheader()
        for r in sorted(batch_rows, key=lambda x: (x["config"], x["input_len"], x["batch_size"])):
            w.writerow(r)
    print(f"  C1: {p1} ({len(batch_rows)} rows)")

    p2 = out_dir / "h200_agentic_ground_truth.csv"
    f2 = ["config", "concurrency", "n_repeats", "n_steps_total",
          "tpot_mean", "tpot_p50", "tpot_p95", "tpot_ci",
          "ttft_mean", "ttft_p50", "ttft_p95", "ttft_ci"]
    with open(p2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=f2, extrasaction="ignore")
        w.writeheader()
        for r in agentic_agg:
            w.writerow(r)
    print(f"  C2: {p2} ({len(agentic_agg)} rows)")

    p3 = out_dir / "h200_agentic_per_step.csv"
    f3 = ["config", "concurrency", "repeat", "task_id", "step_id",
          "agent_role", "tpot_ms", "ttft_ms", "latency_ms",
          "prompt_tokens", "completion_tokens"]
    with open(p3, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=f3, extrasaction="ignore")
        w.writeheader()
        for r in sorted(agentic_steps, key=lambda x: (x["config"], x["concurrency"], x["repeat"], x["task_id"])):
            w.writerow(r)
    print(f"  C3: {p3} ({len(agentic_steps)} rows)")


# ============================================================================
# Markdown Export
# ============================================================================
def export_markdown(batch_rows, agentic_agg, server_means, out_dir):
    """Write all 7 tables as a single markdown file."""
    BATCH_EXCLUDE = {("TP4 H200 NVLink", 128)}

    batch_sizes = sorted(set(r["batch_size"] for r in batch_rows)) if batch_rows else []
    input_lens = sorted(set(r["input_len"] for r in batch_rows)) if batch_rows else []
    batch_cfgs = sorted(set(r["config"] for r in batch_rows))
    bl = {}
    for r in batch_rows:
        bl[(r["config"], r["input_len"], r["batch_size"])] = r
    al = {(r["config"], r["concurrency"]): r for r in agentic_agg}
    all_conc = sorted(set(r["concurrency"] for r in agentic_agg))
    agentic_cfgs = sorted(set(r["config"] for r in agentic_agg),
                          key=lambda c: ALL_CONFIGS.index(c) if c in ALL_CONFIGS else 99)
    ttft_batch_cfgs = [c for c in batch_cfgs if "TP8" in c]
    ttft_agen_cfgs = [c for c in agentic_cfgs if "TP8" in c]

    def sh(name):
        return name.replace(" H200 NVLink", "").replace(" H100 PCIe", "(PCIe)")

    def _w(v, is_win, fmt):
        """Format value, bold if winner."""
        s = f"{v:{fmt}}"
        return f"**{s}**" if is_win else s

    lines = []
    L = lines.append

    L("# H200 Ground Truth — Analysis Tables\n")
    L(f"Generated: {Path(__file__).name}\n")

    # ------------------------------------------------------------------
    # Helper: batch metric table
    # ------------------------------------------------------------------
    def md_batch_table(tag, key, fmt, lower, cfgs_filter=None, note=""):
        cfgs_all = cfgs_filter if cfgs_filter else batch_cfgs
        better = "Lower" if lower else "Higher"
        L(f"\n## {tag}\n")
        L(f"{better} is better. **Bold** = winner per column.\n")
        if note:
            L(f"> {note}\n")

        win_tot = defaultdict(int)
        n_cols = 0
        worst_delta = ("", 0.0)
        descs = []

        for il in input_lens:
            cfgs = [c for c in cfgs_all if (c, il) not in BATCH_EXCLUDE]
            if not cfgs:
                continue
            excl = [sh(c) for c in cfgs_all if (c, il) in BATCH_EXCLUDE]
            excl_s = f" ({', '.join(excl)} excluded)" if excl else ""
            L(f"\n### i={il}{excl_s}\n")

            # Header
            hdr = "| Config |"
            sep = "|--------|"
            for b in batch_sizes:
                hdr += f" b={b} |"
                sep += "------:|"
            hdr += " Δ(1→64) |"
            sep += "---------:|"
            L(hdr)
            L(sep)

            # Winners
            col_w = {}
            for b in batch_sizes:
                cands = [(c, bl.get((c, il, b), {}).get(key, 0)) for c in cfgs]
                cands = [(c, v) for c, v in cands if v > 0]
                if cands:
                    col_w[b] = (min if lower else max)(cands, key=lambda x: x[1])[0]

            for cfg in cfgs:
                row = f"| {cfg} |"
                for b in batch_sizes:
                    v = bl.get((cfg, il, b), {}).get(key, 0)
                    is_win = col_w.get(b) == cfg
                    if is_win:
                        win_tot[cfg] += 1
                        n_cols += 1
                    row += f" {_w(v, is_win, fmt)} |" if v > 0 else " — |"
                v1 = bl.get((cfg, il, batch_sizes[0]), {}).get(key, 0)
                vn = bl.get((cfg, il, batch_sizes[-1]), {}).get(key, 0)
                if v1 > 0 and vn > 0:
                    ratio = vn / v1
                    row += f" {ratio:.1f}x |"
                    if lower and ratio > worst_delta[1]:
                        worst_delta = (f"{sh(cfg)} i={il}", ratio)
                else:
                    row += " — |"
                L(row)

            # Winner row
            wrow = "| **Winner** |"
            for b in batch_sizes:
                w = col_w.get(b, "")
                wrow += f" {sh(w)} |" if w else " — |"
            wrow += " |"
            L(wrow)

            # Notable deltas
            if len(cfgs) >= 2:
                for b in batch_sizes:
                    cands = [(c, bl.get((c, il, b), {}).get(key, 0)) for c in cfgs]
                    cands = [(c, v) for c, v in cands if v > 0]
                    if len(cands) >= 2:
                        s = sorted(cands, key=lambda x: x[1])
                        gap_pct = (s[-1][1] - s[0][1]) / s[0][1] * 100 if s[0][1] > 0 else 0
                        if gap_pct > 50:
                            descs.append(f"i={il} b={b}: {sh(s[-1 if lower else 0][0])} "
                                         f"is {gap_pct:.0f}% worse ({s[0][1]:.1f} vs {s[-1][1]:.1f})")

        # Summary
        L("")
        if win_tot:
            top = max(win_tot.items(), key=lambda x: x[1])
            L(f"- **{sh(top[0])}** wins {top[1]}/{n_cols} columns.")
        if worst_delta[0] and lower:
            L(f"- Steepest degradation: {worst_delta[0]} ({worst_delta[1]:.1f}x from b=1→b=64).")
        for d in descs[:3]:
            L(f"- {d}")

    # ------------------------------------------------------------------
    # Helper: batch TTFT (8GPU, config × input_len)
    # ------------------------------------------------------------------
    def md_batch_ttft():
        if not ttft_batch_cfgs:
            return
        L("\n## T3: Batch TTFT (ms) — 8GPU Only\n")
        L("Lower is better. **Bold** = winner. TTFT constant across batch sizes.\n")
        L("> 4GPU TTFT excluded (scheduling artifacts: 400–12000ms)\n")

        hdr = "| Config |"
        sep = "|--------|"
        for il in input_lens:
            hdr += f" i={il} |"
            sep += "------:|"
        hdr += " EP overhead |"
        sep += "------------|"
        L(hdr)
        L(sep)

        col_w = {}
        for il in input_lens:
            cands = [(c, bl.get((c, il, 1), {}).get("ttft_single_ms", 0)) for c in ttft_batch_cfgs]
            cands = [(c, v) for c, v in cands if v > 0]
            if cands:
                col_w[il] = min(cands, key=lambda x: x[1])[0]

        for cfg in ttft_batch_cfgs:
            row = f"| {cfg} |"
            for il in input_lens:
                v = bl.get((cfg, il, 1), {}).get("ttft_single_ms", 0)
                row += f" {_w(v, col_w.get(il) == cfg, '.1f')} |" if v > 0 else " — |"
            if "EP" in cfg:
                base_cfg = cfg.replace("+EP ", " ")
                deltas = []
                for il in input_lens:
                    ep_v = bl.get((cfg, il, 1), {}).get("ttft_single_ms", 0)
                    base_v = bl.get((base_cfg, il, 1), {}).get("ttft_single_ms", 0)
                    if ep_v > 0 and base_v > 0:
                        deltas.append(ep_v - base_v)
                row += f" +{statistics.mean(deltas):.1f}ms avg |" if deltas else " — |"
            else:
                row += " (baseline) |"
            L(row)

        L("")
        L("- **TP8** wins all input lengths (no EP routing overhead).")
        for il in input_lens:
            tp8 = bl.get(("TP8 H200 NVLink", il, 1), {}).get("ttft_single_ms", 0)
            ep = bl.get(("TP8+EP H200 NVLink", il, 1), {}).get("ttft_single_ms", 0)
            if tp8 > 0 and ep > 0:
                L(f"- i={il}: EP adds +{ep-tp8:.1f}ms ({(ep-tp8)/tp8*100:.0f}% overhead).")

    # ------------------------------------------------------------------
    # Helper: agentic metric table
    # ------------------------------------------------------------------
    def md_agentic_table(tag, key, fmt, lower, cfgs_filter=None, note=""):
        cfgs_use = cfgs_filter if cfgs_filter else agentic_cfgs
        concs = sorted(set(r["concurrency"] for r in agentic_agg if r["config"] in cfgs_use))
        if not concs:
            return
        better = "Lower" if lower else "Higher"
        L(f"\n## {tag}\n")
        L(f"{better} is better. **Bold** = winner.\n")
        if note:
            L(f"> {note}\n")

        hdr = "| Config |"
        sep = "|--------|"
        for c in concs:
            hdr += f" c={c} |"
            sep += "------:|"
        hdr += " Δ(min→max) |"
        sep += "----------:|"
        L(hdr)
        L(sep)

        col_w = {}
        for c in concs:
            cands = [(cfg, al.get((cfg, c), {}).get(key, 0)) for cfg in cfgs_use]
            cands = [(cfg, v) for cfg, v in cands if v and v > 0 and np.isfinite(v)]
            if cands:
                col_w[c] = (min if lower else max)(cands, key=lambda x: x[1])[0]

        worst_delta = ("", 0.0)
        for cfg in cfgs_use:
            row = f"| {cfg} |"
            for c in concs:
                v = al.get((cfg, c), {}).get(key, 0)
                if v and v > 0 and np.isfinite(v):
                    row += f" {_w(v, col_w.get(c) == cfg, fmt)} |"
                else:
                    row += " — |"
            vals_cfg = [al.get((cfg, c), {}).get(key, 0) for c in concs]
            vals_cfg = [v for v in vals_cfg if v and v > 0 and np.isfinite(v)]
            if len(vals_cfg) >= 2:
                ratio = max(vals_cfg) / min(vals_cfg) if min(vals_cfg) > 0 else 0
                row += f" {ratio:.1f}x |"
                if ratio > worst_delta[1]:
                    worst_delta = (sh(cfg), ratio)
            else:
                row += " — |"
            L(row)

        wrow = "| **Winner** |"
        for c in concs:
            w = col_w.get(c, "")
            wrow += f" {sh(w)} |" if w else " — |"
        wrow += " |"
        L(wrow)

        L("")
        if col_w:
            counts = defaultdict(int)
            for w in col_w.values():
                counts[w] += 1
            top = max(counts.items(), key=lambda x: x[1])
            L(f"- **{sh(top[0])}** wins {top[1]}/{len(col_w)} columns.")
        if worst_delta[0]:
            lbl = "degradation" if lower else "range"
            L(f"- Max {lbl}: {worst_delta[0]} ({worst_delta[1]:.1f}x).")
        for i in range(1, len(concs)):
            w_prev = col_w.get(concs[i - 1])
            w_curr = col_w.get(concs[i])
            if w_prev and w_curr and w_prev != w_curr:
                L(f"- Winner transition at c={concs[i]}: {sh(w_prev)} → {sh(w_curr)}.")

    # ------------------------------------------------------------------
    # Helper: agentic TPS
    # ------------------------------------------------------------------
    def md_agentic_tps():
        cfgs_use = agentic_cfgs
        concs = all_conc
        L("\n## T5: Agentic TPS\n")
        L("Higher is better. **Bold** = winner.\n")

        hdr = "| Config |"
        sep = "|--------|"
        for c in concs:
            hdr += f" c={c} |"
            sep += "------:|"
        hdr += " Peak |"
        sep += "-----:|"
        L(hdr)
        L(sep)

        col_w = {}
        for c in concs:
            cands = [(cfg, server_means.get(cfg, {}).get(c, {}).get("gen_tps", 0)) for cfg in cfgs_use]
            cands = [(cfg, v) for cfg, v in cands if v and v > 0]
            if cands:
                col_w[c] = max(cands, key=lambda x: x[1])[0]

        for cfg in cfgs_use:
            sm = server_means.get(cfg, {})
            row = f"| {cfg} |"
            vals = []
            for c in concs:
                v = sm.get(c, {}).get("gen_tps", 0)
                if v and v > 0:
                    row += f" {_w(v, col_w.get(c) == cfg, '.0f')} |"
                    vals.append(v)
                else:
                    row += " — |"
            row += f" {max(vals):.0f} |" if vals else " — |"
            L(row)

        wrow = "| **Winner** |"
        for c in concs:
            w = col_w.get(c, "")
            wrow += f" {sh(w)} |" if w else " — |"
        wrow += " |"
        L(wrow)

        L("")
        if col_w:
            counts = defaultdict(int)
            for w in col_w.values():
                counts[w] += 1
            top = max(counts.items(), key=lambda x: x[1])
            L(f"- **{sh(top[0])}** wins {top[1]}/{len(col_w)} columns.")
        for cfg in cfgs_use:
            sm = server_means.get(cfg, {})
            vals = [(c, sm.get(c, {}).get("gen_tps", 0)) for c in concs]
            vals = [(c, v) for c, v in vals if v > 0]
            if len(vals) >= 2:
                peak_c, peak_v = max(vals, key=lambda x: x[1])
                last_c, last_v = vals[-1]
                if last_v < peak_v * 0.85 and last_c != peak_c:
                    L(f"- {sh(cfg)}: peaks c={peak_c} ({peak_v:.0f}), drops {(1-last_v/peak_v)*100:.0f}% at c={last_c}.")

    # ------------------------------------------------------------------
    # T7: Platform comparison
    # ------------------------------------------------------------------
    def md_platform():
        ic_pair = ("TP4 H200 NVLink", "TP4 H100 PCIe")
        common = sorted(
            set(r["concurrency"] for r in agentic_agg if r["config"] == ic_pair[0])
            & set(r["concurrency"] for r in agentic_agg if r["config"] == ic_pair[1]))
        if not common:
            return
        L("\n## T7: Platform — TP4 H200 NVLink vs TP4 H100 PCIe\n")

        hdr = "| Metric |"
        sep = "|--------|"
        for c in common:
            hdr += f" c={c} |"
            sep += "------:|"
        L(hdr)
        L(sep)

        for lbl, cfg, key in [("H200 NVLink TPOT (ms)", ic_pair[0], "tpot_mean"),
                               ("H100 PCIe TPOT (ms)", ic_pair[1], "tpot_mean")]:
            row = f"| {lbl} |"
            for c in common:
                r = al.get((cfg, c))
                row += f" {r[key]:.1f} |" if r and np.isfinite(r[key]) else " — |"
            L(row)
        row = "| **Δ H100−H200 (ms)** |"
        for c in common:
            h2 = al.get((ic_pair[0], c))
            h1 = al.get((ic_pair[1], c))
            if h2 and h1:
                d = h1["tpot_mean"] - h2["tpot_mean"]
                row += f" +{d:.1f} |"
            else:
                row += " — |"
        L(row)
        L("")

        hdr2 = "| Metric |"
        for c in common:
            hdr2 += f" c={c} |"
        L(hdr2)
        L(sep)
        for lbl, cfg in [("H200 NVLink TPS", ic_pair[0]), ("H100 PCIe TPS", ic_pair[1])]:
            sm = server_means.get(cfg, {})
            row = f"| {lbl} |"
            for c in common:
                v = sm.get(c, {}).get("gen_tps", 0)
                row += f" {v:.0f} |" if v > 0 else " — |"
            L(row)
        row = "| **Δ H200−H100** |"
        for c in common:
            v200 = server_means.get(ic_pair[0], {}).get(c, {}).get("gen_tps", 0)
            v100 = server_means.get(ic_pair[1], {}).get(c, {}).get("gen_tps", 0)
            row += f" +{v200-v100:.0f} |" if v200 > 0 and v100 > 0 else " — |"
        L(row)

        L("")
        for c in common:
            h2_t = al.get((ic_pair[0], c), {}).get("tpot_mean", 0)
            h1_t = al.get((ic_pair[1], c), {}).get("tpot_mean", 0)
            v200 = server_means.get(ic_pair[0], {}).get(c, {}).get("gen_tps", 0)
            v100 = server_means.get(ic_pair[1], {}).get(c, {}).get("gen_tps", 0)
            if h2_t > 0 and h1_t > 0:
                speedup = h1_t / h2_t
                tps_gain = ((v200 - v100) / v100 * 100) if v100 > 0 else 0
                L(f"- c={c}: H200 is **{speedup:.1f}x** faster TPOT"
                  f"{f', +{tps_gain:.0f}% TPS' if tps_gain else ''}.")

    # ------------------------------------------------------------------
    # Generate all tables
    # ------------------------------------------------------------------
    tp_only_cfgs = [c for c in batch_cfgs if "DP" not in c]
    md_batch_table("T1: Batch TPOT (ms)", "tpot_mean_ms", ".2f", True, cfgs_filter=tp_only_cfgs)
    md_batch_table("T2: Batch TPS", "gen_tps", ".0f", False, cfgs_filter=tp_only_cfgs)
    md_batch_ttft()
    md_agentic_table("T4: Agentic TPOT (ms)", "tpot_mean", ".2f", True)
    md_agentic_tps()
    md_agentic_table("T6: Agentic TTFT (ms)", "ttft_mean", ".1f", True,
                     cfgs_filter=ttft_agen_cfgs,
                     note="4GPU H200 TTFT excluded (scheduling artifacts)")
    md_platform()

    # ------------------------------------------------------------------
    # T8: DP Batch TPOT
    # ------------------------------------------------------------------
    dp_cfgs = [c for c in batch_cfgs if "DP" in c]
    if dp_cfgs:
        md_batch_table("T8: DP Batch TPOT (ms)", "tpot_mean_ms", ".2f", True,
                       cfgs_filter=dp_cfgs,
                       note="DP configs only. TP2-DP4-EP i=128 b=8 excluded (anomaly).")

    # ------------------------------------------------------------------
    # T9: TP vs DP Comparison (ratio)
    # ------------------------------------------------------------------
    dp_compare = [("TP8 H200 NVLink", "TP4-DP2 H200 NVLink", "TP4-DP2/TP8"),
                  ("TP8 H200 NVLink", "TP2-DP4-EP H200 NVLink", "TP2-DP4-EP/TP8")]
    dp_compare = [(base, dp, lbl) for base, dp, lbl in dp_compare
                  if base in batch_cfgs and dp in batch_cfgs]
    if dp_compare:
        L(f"\n## T9: TP vs DP — TPOT Ratio\n")
        L("Ratio = DP TPOT / TP8 TPOT. >1 means DP is slower per-request.\n")
        for il in input_lens:
            avail = [(base, dp, lbl) for base, dp, lbl in dp_compare
                     if bl.get((dp, il, 1), {}).get("tpot_mean_ms", 0) > 0]
            if not avail:
                continue
            L(f"\n### i={il}\n")
            hdr = "| Comparison |"
            sep = "|------------|"
            for b in batch_sizes:
                hdr += f" b={b} |"
                sep += "------:|"
            L(hdr)
            L(sep)
            for base_cfg, dp_cfg, lbl in avail:
                row = f"| {lbl} |"
                for b in batch_sizes:
                    base_v = bl.get((base_cfg, il, b), {}).get("tpot_mean_ms", 0)
                    dp_v = bl.get((dp_cfg, il, b), {}).get("tpot_mean_ms", 0)
                    if base_v > 0 and dp_v > 0:
                        ratio = dp_v / base_v
                        row += f" {ratio:.2f}x |"
                    else:
                        row += " — |"
                L(row)
        L("")
        L("- Ratio >1: DP is slower per-request (but total throughput scales with DP factor).")
        L("- Ratio ~1: DP matches TP8 per-request TPOT.")

    md_path = out_dir / "h200_ground_truth_tables.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"  MD: {md_path} ({len(lines)} lines)")


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="H200 Ground Truth Analysis")
    ap.add_argument("--output-dir", default="results/h200_analysis")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    batch_rows, agentic_steps, h100_agg, server_means, server_reps = load_all_data()
    if not batch_rows and not agentic_steps:
        print("ERROR: No data.")
        sys.exit(1)

    agentic_agg = aggregate_agentic(agentic_steps) if agentic_steps else []
    agentic_agg.extend(h100_agg)
    agentic_agg.sort(key=lambda r: (ALL_CONFIGS.index(r["config"]) if r["config"] in ALL_CONFIGS else 99, r["concurrency"]))

    print_tables(batch_rows, agentic_agg, server_means)

    print("\nGenerating figures...")
    fd = make_figures(batch_rows, agentic_agg, server_means, server_reps, out)

    print("\nExporting CSVs + Markdown...")
    export_csvs(batch_rows, agentic_agg, agentic_steps, out)
    export_markdown(batch_rows, agentic_agg, server_means, out)

    nf = len(list(fd.glob("*.png")))
    print(f"\nDone: {out}/  ({nf} PNGs, 3 CSVs, 1 MD)")


if __name__ == "__main__":
    main()
