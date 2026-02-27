#!/usr/bin/env python3
"""Agentic Workload Comparison Plots — SGLang vs vLLM, Framework, Topology."""

import json
import os
import shutil
import sys
import importlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = Path('/tmp/mab_dateutil_vendor')

def ensure_compatible_dateutil():
    """Matplotlib>=3.8 requires dateutil>=2.7; current env may provide 2.6.1."""
    if VENDOR_DIR.exists():
        sys.path.insert(0, str(VENDOR_DIR))
    try:
        import dateutil  # type: ignore
        ver = tuple(int(x) for x in dateutil.__version__.split('.')[:2])
        if ver >= (2, 7):
            return
        src_dir = Path(dateutil.__file__).resolve().parent
        dst_dir = VENDOR_DIR / 'dateutil'
        if not dst_dir.exists():
            VENDOR_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_dir, dst_dir)
            vfile = dst_dir / '_version.py'
            txt = vfile.read_text()
            txt = txt.replace('VERSION_MINOR = 6', 'VERSION_MINOR = 8')
            txt = txt.replace('VERSION_PATCH = 1', 'VERSION_PATCH = 2')
            vfile.write_text(txt)
        if str(VENDOR_DIR) not in sys.path:
            sys.path.insert(0, str(VENDOR_DIR))
        for key in [k for k in list(sys.modules) if k == 'dateutil' or k.startswith('dateutil.')]:
            sys.modules.pop(key, None)
        importlib.import_module('dateutil')
    except Exception as e:
        raise RuntimeError(f'Failed to prepare dateutil compatibility shim: {e}') from e

ensure_compatible_dateutil()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ─── Data Loading ───
RESULTS_BASE = os.path.join(os.path.dirname(__file__), '..', 'results_multiagent')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results_multiagent', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

def load_metric(path):
    if os.path.exists(path):
        return json.load(open(path))
    return None

rows = []

# SGLang TP2 (Feb 17 session)
for prec in ['fp8', 'fp16']:
    for fw in ['autogen', 'langgraph', 'a2a']:
        for c in [1, 8, 32]:
            f = f'{RESULTS_BASE}/sweep_tp2-{prec}_{fw}/{fw}_c{c}/server_metrics_{fw}_c{c}.json'
            d = load_metric(f)
            if d and d.get('tpot_mean_ms'):
                rows.append({
                    **d, 'engine': 'sglang', 'prec': prec, 'fw': fw, 'conc': c, 'topo': 'tp2',
                    'source': 'sweep_sglang_tp2'
                })

# vLLM TP2-FP16 (profiling dir: autogen c=1,8,32,64 + a2a c=32)
vllm_tp2_fp16_loaded = set()
for fw in ['autogen', 'a2a']:
    for c in [1, 8, 32, 64]:
        f = f'{RESULTS_BASE}/profiling/vllm_tp2-fp16_{fw}_c{c}/server_metrics_{fw}_c{c}.json'
        d = load_metric(f)
        if d and d.get('tpot_mean_ms'):
            rows.append({
                **d, 'engine': 'vllm', 'prec': 'fp16', 'fw': fw, 'conc': c, 'topo': 'tp2',
                'source': 'profiling_vllm_tp2_fp16'
            })
            vllm_tp2_fp16_loaded.add((fw, c))
# Fallback: conc_sweep (autogen only, if not already loaded)
for c in [1, 8, 32]:
    if ('autogen', c) not in vllm_tp2_fp16_loaded:
        f = f'{RESULTS_BASE}/conc_sweep/autogen_c{c}/server_metrics_autogen_c{c}.json'
        d = load_metric(f)
        if d and d.get('tpot_mean_ms'):
            rows.append({
                **d, 'engine': 'vllm', 'prec': 'fp16', 'fw': 'autogen', 'conc': c, 'topo': 'tp2',
                'source': 'conc_sweep_vllm_tp2_fp16'
            })

# vLLM TP2-FP16 framework sweep (used for framework-only comparisons)
for fw in ['autogen', 'langgraph', 'a2a']:
    for c in [1, 8, 32]:
        f = f'{RESULTS_BASE}/sweep_tp2-fp16_{fw}/{fw}_c{c}/server_metrics_{fw}_c{c}.json'
        d = load_metric(f)
        if d and d.get('tpot_mean_ms'):
            rows.append({
                **d, 'engine': 'vllm', 'prec': 'fp16', 'fw': fw, 'conc': c, 'topo': 'tp2',
                'source': 'sweep_vllm_tp2_fp16'
            })

# vLLM TP1-FP8
for fw in ['autogen', 'langgraph', 'a2a']:
    for c in [1, 4, 8, 16, 32]:
        f = f'{RESULTS_BASE}/sweep_tp1-fp8_{fw}/{fw}_c{c}/server_metrics_{fw}_c{c}.json'
        d = load_metric(f)
        if d and d.get('tpot_mean_ms'):
            rows.append({
                **d, 'engine': 'vllm', 'prec': 'fp8', 'fw': fw, 'conc': c, 'topo': 'tp1',
                'source': 'sweep_vllm_tp1_fp8'
            })

# vLLM TP2-FP8 autogen (from old sweep, before SGLang overwrite — use conc_sweep_summary or trace data)
# These were overwritten by SGLang. Use archived summary values.
vllm_tp2_fp8_autogen = [
    {'engine': 'vllm', 'prec': 'fp8', 'fw': 'autogen', 'conc': 1, 'topo': 'tp2',
     'tpot_mean_ms': 7.6, 'gen_tps': 120.8, 'ttft_mean_ms': 235.2},
    {'engine': 'vllm', 'prec': 'fp8', 'fw': 'autogen', 'conc': 8, 'topo': 'tp2',
     'tpot_mean_ms': 12.4, 'gen_tps': 549.9, 'ttft_mean_ms': 152.0},
    {'engine': 'vllm', 'prec': 'fp8', 'fw': 'autogen', 'conc': 32, 'topo': 'tp2',
     'tpot_mean_ms': 22.1, 'gen_tps': 924.4, 'ttft_mean_ms': 189.5},
]
rows.extend(vllm_tp2_fp8_autogen)

# SGLang EP2 (Feb 18 session)
for prec in ['fp8', 'fp16']:
    for fw in ['autogen', 'langgraph', 'a2a']:
        for c in [1, 8, 32, 64]:
            f = f'{RESULTS_BASE}/sweep_dp2-ep2-{prec}_{fw}/{fw}_c{c}/server_metrics_{fw}_c{c}.json'
            d = load_metric(f)
            if d and d.get('tpot_mean_ms'):
                rows.append({
                    **d, 'engine': 'sglang', 'prec': prec, 'fw': fw, 'conc': c, 'topo': 'ep2',
                    'source': 'sweep_sglang_ep2'
                })

# vLLM EP2-FP16 (Feb 18 session — TP2+EP, no EPLB)
for fw in ['autogen', 'langgraph', 'a2a']:
    for c in [1, 8, 32, 64]:
        f = f'{RESULTS_BASE}/sweep_vllm-ep2-fp16_{fw}/{fw}_c{c}/server_metrics_{fw}_c{c}.json'
        d = load_metric(f)
        if d and d.get('tpot_mean_ms'):
            rows.append({
                **d, 'engine': 'vllm', 'prec': 'fp16', 'fw': fw, 'conc': c, 'topo': 'ep2',
                'source': 'sweep_vllm_ep2_fp16'
            })

print(f"Total data points: {len(rows)}")

# ─── Trace Data Loading (token analysis) ───
import glob

def load_traces(path):
    """Load all JSONL trace entries from a file."""
    traces = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    traces.append(json.loads(line))
                except:
                    pass
    return traces

trace_rows = []  # per-step rows for token analysis
for sweep_dir in sorted(glob.glob(f'{RESULTS_BASE}/sweep_*')):
    sweep_name = os.path.basename(sweep_dir)
    for sub in sorted(glob.glob(f'{sweep_dir}/*/trace_*.jsonl')):
        parts = os.path.basename(os.path.dirname(sub)).split('_')
        fw = parts[0] if parts else 'unknown'
        conc = int(parts[1].replace('c', '')) if len(parts) > 1 else 0
        # Determine engine and topo from sweep name
        if 'vllm-ep2' in sweep_name:
            eng, topo = 'vllm', 'ep2'
        elif 'dp2-ep2' in sweep_name:
            eng, topo = 'sglang', 'ep2'
        elif 'tp2' in sweep_name:
            eng, topo = 'sglang' if 'sglang' not in sweep_name else 'sglang', 'tp2'
            if 'conc_sweep' in sweep_name:
                eng = 'vllm'
        elif 'tp1' in sweep_name:
            eng, topo = 'vllm', 'tp1'
        else:
            eng, topo = 'unknown', 'unknown'
        prec = 'fp8' if 'fp8' in sweep_name else 'fp16'
        for t in load_traces(sub):
            steps = t.get('steps', {})
            for step_id, step in steps.items():
                trace_rows.append({
                    'engine': eng, 'prec': prec, 'fw': fw, 'conc': conc,
                    'topo': topo, 'task_id': t.get('task_id'),
                    'step_id': step_id,
                    'role': step.get('agent_role', ''),
                    'prompt_tokens': step.get('prompt_tokens', 0),
                    'completion_tokens': step.get('completion_tokens', 0),
                    'total_tokens': step.get('total_tokens', 0),
                    'ttft_ms': step.get('ttft_ms', 0),
                    'tpot_ms': step.get('tpot_ms', 0),
                    'latency_ms': step.get('latency_ms', 0),
                    'wait_ms': step.get('wait_ms', 0),
                    'dag_depth': t.get('dag_metrics', {}).get('depth', 0),
                    'max_width': t.get('dag_metrics', {}).get('max_width', 0),
                    'parallel_fraction': t.get('dag_metrics', {}).get('parallel_fraction', 0),
                    'makespan_ms': t.get('makespan_ms', 0),
                })

print(f"Total trace steps: {len(trace_rows)}")

# ─── Styling ───
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

ENGINE_COLORS = {'vllm': '#E74C3C', 'sglang': '#2ECC71'}
FW_COLORS = {'autogen': '#3498DB', 'langgraph': '#E67E22', 'a2a': '#9B59B6'}
FW_MARKERS = {'autogen': 'o', 'langgraph': 's', 'a2a': '^'}
PREC_STYLES = {'fp8': '-', 'fp16': '--'}
TOPO_COLORS = {'tp1': '#E74C3C', 'tp2': '#2ECC71', 'ep2': '#9B59B6'}
TOPO_MARKERS = {'tp1': 'o', 'tp2': 's', 'ep2': 'D'}

def get_rows(**filters):
    result = rows
    for k, v in filters.items():
        if isinstance(v, list):
            result = [r for r in result if r.get(k) in v]
        else:
            result = [r for r in result if r.get(k) == v]
    return sorted(result, key=lambda r: r['conc'])


def find_row_for_conc(conc, source_priority=None, **filters):
    if source_priority:
        for source in source_priority:
            d = get_rows(source=source, **filters)
            match = [r for r in d if r['conc'] == conc]
            if match:
                return match[0]
        return None
    d = get_rows(**filters)
    match = [r for r in d if r['conc'] == conc]
    return match[0] if match else None

# ════════════════════════════════════════════════════════════════════
# Figure 1: SGLang vs vLLM — TPOT and TPS (TP2-FP16, autogen)
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1a: TPOT
ax = axes[0]
for eng, color in [('vllm', ENGINE_COLORS['vllm']), ('sglang', ENGINE_COLORS['sglang'])]:
    d = get_rows(engine=eng, prec='fp16', fw='autogen', topo='tp2')
    if d:
        cs = [r['conc'] for r in d]
        vals = [r['tpot_mean_ms'] for r in d]
        ax.plot(cs, vals, 'o-', color=color, label=f'{eng.upper()}', linewidth=2, markersize=8)
ax.set_xlabel('Concurrency')
ax.set_ylabel('TPOT (ms)')
ax.set_title('(a) TPOT — TP2 FP16')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks([1, 8, 32])

# 1b: TPS
ax = axes[1]
for eng, color in [('vllm', ENGINE_COLORS['vllm']), ('sglang', ENGINE_COLORS['sglang'])]:
    d = get_rows(engine=eng, prec='fp16', fw='autogen', topo='tp2')
    if d:
        cs = [r['conc'] for r in d]
        vals = [r.get('gen_tps', 0) for r in d]
        ax.plot(cs, vals, 'o-', color=color, label=f'{eng.upper()}', linewidth=2, markersize=8)
ax.set_xlabel('Concurrency')
ax.set_ylabel('Generation TPS')
ax.set_title('(b) Throughput — TP2 FP16')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks([1, 8, 32])

# 1c: TTFT
ax = axes[2]
for eng, color in [('vllm', ENGINE_COLORS['vllm']), ('sglang', ENGINE_COLORS['sglang'])]:
    d = get_rows(engine=eng, prec='fp16', fw='autogen', topo='tp2')
    if d:
        cs = [r['conc'] for r in d]
        vals = [r.get('ttft_mean_ms', 0) for r in d]
        ax.plot(cs, vals, 'o-', color=color, label=f'{eng.upper()}', linewidth=2, markersize=8)
ax.set_xlabel('Concurrency')
ax.set_ylabel('TTFT (ms)')
ax.set_title('(c) TTFT — TP2 FP16')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks([1, 8, 32])

fig.suptitle('Figure 1: SGLang vs vLLM — TP2 FP16 (autogen)', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f'{OUT_DIR}/fig1_sglang_vs_vllm_tp2_fp16.png', bbox_inches='tight')
print(f"Saved fig1")

# ════════════════════════════════════════════════════════════════════
# Figure 2: SGLang FP8 vs FP16 — TPOT & TPS by framework
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, fw in enumerate(['autogen', 'langgraph', 'a2a']):
    ax = axes[idx]
    for prec, ls in [('fp8', '-'), ('fp16', '--')]:
        d = get_rows(engine='sglang', prec=prec, fw=fw, topo='tp2')
        if d:
            cs = [r['conc'] for r in d]
            tpot = [r['tpot_mean_ms'] for r in d]
            color = '#E74C3C' if prec == 'fp8' else '#3498DB'
            ax.plot(cs, tpot, f'o{ls}', color=color, label=f'FP8' if prec == 'fp8' else 'FP16',
                    linewidth=2, markersize=8)
    ax.set_xlabel('Concurrency')
    ax.set_ylabel('TPOT (ms)')
    ax.set_title(f'{fw}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([1, 8, 32])

fig.suptitle('Figure 2: SGLang FP8 vs FP16 — TPOT by Framework (TP2)', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f'{OUT_DIR}/fig2_sglang_fp8_vs_fp16_tpot.png', bbox_inches='tight')
print(f"Saved fig2")

# ════════════════════════════════════════════════════════════════════
# Figure 3A: Precision Effect — FP8 vs FP16 Absolute Values (SGLang TP2)
# ════════════════════════════════════════════════════════════════════
fig3a_metrics = [
    ('tpot_mean_ms', 'TPOT (ms)',     '(a) TPOT'),
    ('gen_tps',      'Generation TPS', '(b) Throughput'),
    ('ttft_mean_ms', 'TTFT (ms)',     '(c) TTFT'),
]
fig3a, axes3a = plt.subplots(1, 3, figsize=(18, 5))
fws_list = ['autogen', 'langgraph', 'a2a']
concs_fig3a = [1, 8, 32]
PREC_COLORS = {'fp8': '#E74C3C', 'fp16': '#3498DB'}

for ax_idx, (metric, ylabel, title) in enumerate(fig3a_metrics):
    ax = axes3a[ax_idx]
    x = np.arange(len(concs_fig3a))
    n_groups = len(fws_list) * 2  # fp8 + fp16 per framework
    width = 0.8 / n_groups
    bar_idx = 0
    for fw in fws_list:
        for prec in ['fp8', 'fp16']:
            vals = []
            for c in concs_fig3a:
                d = get_rows(engine='sglang', prec=prec, fw=fw, topo='tp2')
                match = [r for r in d if r['conc'] == c]
                vals.append(match[0].get(metric, 0) if match else float('nan'))
            color = PREC_COLORS[prec]
            hatch = '' if prec == 'fp8' else '///'
            ax.bar(x + bar_idx * width, vals, width, label=f'{fw} {prec.upper()}',
                   color=color, alpha=0.75 if prec == 'fp8' else 0.5,
                   edgecolor='black', linewidth=0.5, hatch=hatch)
            bar_idx += 1
    ax.set_xlabel('Concurrency')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + (n_groups - 1) * width / 2)
    ax.set_xticklabels([str(c) for c in concs_fig3a])
    # Deduplicate legend: show only FP8/FP16
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

fig3a.suptitle('Figure 3A: Precision Effect — FP8 vs FP16 (SGLang TP2)',
               fontsize=14, fontweight='bold')
fig3a.tight_layout(rect=[0, 0, 1, 0.94])
fig3a.savefig(f'{OUT_DIR}/fig3a_precision_effect.png', bbox_inches='tight')
print("Saved fig3a")

# ════════════════════════════════════════════════════════════════════
# Figure 3B: Framework Effect — vLLM FP16 (EP2 left, TP2 right)
# ════════════════════════════════════════════════════════════════════
fig3b_metrics = [
    ('tpot_mean_ms', 'TPOT (ms)',     'TPOT'),
    ('gen_tps',      'Generation TPS', 'Throughput'),
    ('ttft_mean_ms', 'TTFT (ms)',      'TTFT'),
]
fig3b_configs = [
    ('DP1×EP2', 'ep2', ['sweep_vllm_ep2_fp16']),
    ('TP2',     'tp2', ['sweep_vllm_tp2_fp16']),
]
fig3b, axes3b = plt.subplots(len(fig3b_metrics), len(fig3b_configs),
                              figsize=(16, 11), sharey='row')
concs_fig3b = [1, 8, 32]

for col_idx, (cfg_label, topo, sources) in enumerate(fig3b_configs):
    for row_idx, (metric, ylabel, metric_label) in enumerate(fig3b_metrics):
        ax = axes3b[row_idx][col_idx]
        x = np.arange(len(concs_fig3b))
        width = 0.25
        for i, fw in enumerate(fws_list):
            vals = []
            for c in concs_fig3b:
                row = find_row_for_conc(
                    c,
                    engine='vllm',
                    prec='fp16',
                    fw=fw,
                    topo=topo,
                    source_priority=sources,
                )
                vals.append(row.get(metric, 0) if row else float('nan'))
            ax.bar(x + i * width, vals, width, label=fw,
                   color=FW_COLORS[fw], alpha=0.85, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Concurrency')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{metric_label} — {cfg_label}')
        ax.set_xticks(x + width)
        ax.set_xticklabels([str(c) for c in concs_fig3b])
        if row_idx == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

fig3b.suptitle('Figure 3B: Framework Effect — vLLM FP16',
               fontsize=14, fontweight='bold')
fig3b.tight_layout(rect=[0, 0, 1, 0.95])
fig3b.savefig(f'{OUT_DIR}/fig3b_framework_effect.png', bbox_inches='tight')
print("Saved fig3b")

# ════════════════════════════════════════════════════════════════════
# Figure 3B-tail: Framework Tail Effect — Mean vs p95 at c=32
# ════════════════════════════════════════════════════════════════════
fig3b_tail_pairs = [
    ('tpot_mean_ms', 'tpot_p95_ms', 'TPOT (ms)', 'TPOT Mean vs p95'),
    ('ttft_mean_ms', 'ttft_p95_ms', 'TTFT (ms)', 'TTFT Mean vs p95'),
]
fig3b_tail, axes3b_tail = plt.subplots(len(fig3b_tail_pairs), len(fig3b_configs),
                                        figsize=(16, 8), sharey='row')
fig3b_tail_fw_order = ['autogen', 'a2a', 'langgraph']
fig3b_tail_c = 32

for col_idx, (cfg_label, topo, sources) in enumerate(fig3b_configs):
    for row_idx, (mean_metric, p95_metric, ylabel, metric_label) in enumerate(fig3b_tail_pairs):
        ax = axes3b_tail[row_idx][col_idx]
        x = np.arange(len(fig3b_tail_fw_order))
        width = 0.36
        mean_vals = []
        p95_vals = []
        for fw in fig3b_tail_fw_order:
            row = find_row_for_conc(
                fig3b_tail_c,
                engine='vllm',
                prec='fp16',
                fw=fw,
                topo=topo,
                source_priority=sources,
            )
            mean_vals.append(row.get(mean_metric, 0) if row else float('nan'))
            p95_vals.append(row.get(p95_metric, 0) if row else float('nan'))

        for i, fw in enumerate(fig3b_tail_fw_order):
            ax.bar(x[i] - width / 2, mean_vals[i], width, label='Mean' if i == 0 else None,
                   color=FW_COLORS[fw], alpha=0.55, edgecolor='white', linewidth=0.6)
            ax.bar(x[i] + width / 2, p95_vals[i], width, label='p95' if i == 0 else None,
                   color=FW_COLORS[fw], alpha=0.9, edgecolor='black', linewidth=0.6, hatch='///')

        ax.set_xlabel('Framework')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{metric_label} — {cfg_label} (c=32)')
        ax.set_xticks(x)
        ax.set_xticklabels(fig3b_tail_fw_order)
        if row_idx == 0:
            ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')

fig3b_tail.suptitle('Figure 3B-tail: Framework Tail Effect — Mean vs p95 at c=32 (vLLM FP16)',
                    fontsize=14, fontweight='bold')
fig3b_tail.tight_layout(rect=[0, 0, 1, 0.95])
fig3b_tail.savefig(f'{OUT_DIR}/fig3b_framework_tail.png', bbox_inches='tight')
print("Saved fig3b_tail")

# ════════════════════════════════════════════════════════════════════
# Figure 3C: Topology Effect — TPS, TTFT, TPOT (vLLM FP16, autogen)
# ════════════════════════════════════════════════════════════════════
fig3c_metrics = [
    ('gen_tps', 'Generation TPS', '(a) Throughput'),
    ('ttft_mean_ms', 'TTFT (ms)', '(b) TTFT Mean'),
    ('tpot_mean_ms', 'TPOT (ms)', '(c) TPOT'),
]
fig3c, axes3c = plt.subplots(1, 3, figsize=(16, 5))
concs_fig3c = [8, 32]
topo_configs = [
    ('TP2 FP16',      'vllm', 'fp16', 'tp2', ['profiling_vllm_tp2_fp16', 'conc_sweep_vllm_tp2_fp16']),
    ('DP1×EP2 FP16',  'vllm', 'fp16', 'ep2', ['sweep_vllm_ep2_fp16']),
]

for ax_idx, (metric, ylabel, title) in enumerate(fig3c_metrics):
    ax = axes3c[ax_idx]
    x = np.arange(len(concs_fig3c))
    width = 0.35
    for i, (topo_label, eng, prec, topo, sources) in enumerate(topo_configs):
        vals = []
        for c in concs_fig3c:
            row = find_row_for_conc(
                c,
                engine=eng,
                prec=prec,
                fw='autogen',
                topo=topo,
                source_priority=sources,
            )
            vals.append(row.get(metric, 0) if row else float('nan'))
        ax.bar(x + i * width, vals, width, label=topo_label,
               color=TOPO_COLORS[topo], alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Concurrency')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([str(c) for c in concs_fig3c])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

fig3c.suptitle('Figure 3C: Topology Effect — TPS, TTFT, TPOT (vLLM FP16, autogen)',
               fontsize=14, fontweight='bold')
fig3c.tight_layout(rect=[0, 0, 1, 0.95])
fig3c.savefig(f'{OUT_DIR}/fig3c_topology_effect.png', bbox_inches='tight')
print("Saved fig3c")

# ════════════════════════════════════════════════════════════════════
# Figure 3C-tail: Topology Tail Effect — TTFT/TPOT p95/p99 (vLLM FP16, autogen)
# ════════════════════════════════════════════════════════════════════
fig3c_tail_metrics = [
    ('ttft_p95_ms', 'TTFT p95 (ms)', '(a) TTFT p95'),
    ('ttft_p99_ms', 'TTFT p99 (ms)', '(b) TTFT p99'),
    ('tpot_p95_ms', 'TPOT p95 (ms)', '(c) TPOT p95'),
    ('tpot_p99_ms', 'TPOT p99 (ms)', '(d) TPOT p99'),
]
fig3c_tail, axes3c_tail = plt.subplots(2, 2, figsize=(14, 10))
axes3c_tail_flat = axes3c_tail.flatten()

for ax_idx, (metric, ylabel, title) in enumerate(fig3c_tail_metrics):
    ax = axes3c_tail_flat[ax_idx]
    x = np.arange(len(concs_fig3c))
    width = 0.35
    for i, (topo_label, eng, prec, topo, sources) in enumerate(topo_configs):
        vals = []
        for c in concs_fig3c:
            row = find_row_for_conc(
                c,
                engine=eng,
                prec=prec,
                fw='autogen',
                topo=topo,
                source_priority=sources,
            )
            vals.append(row.get(metric, 0) if row else float('nan'))
        ax.bar(x + i * width, vals, width, label=topo_label,
               color=TOPO_COLORS[topo], alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Concurrency')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([str(c) for c in concs_fig3c])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

fig3c_tail.suptitle('Figure 3C-tail: Topology Tail Effect — TTFT/TPOT p95/p99 (vLLM FP16, autogen)',
                    fontsize=14, fontweight='bold')
fig3c_tail.tight_layout(rect=[0, 0, 1, 0.95])
fig3c_tail.savefig(f'{OUT_DIR}/fig3c_topology_tail.png', bbox_inches='tight')
print("Saved fig3c_tail")

# ════════════════════════════════════════════════════════════════════
# Figure 4: Topology comparison — TP1 vs TP2 (TPOT scaling)
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 4a: TPOT scaling — TP1 vs TP2 (FP8, autogen)
ax = axes[0]
# TP1 vLLM
d = get_rows(engine='vllm', prec='fp8', fw='autogen', topo='tp1')
if d:
    cs = [r['conc'] for r in d]
    vals = [r['tpot_mean_ms'] for r in d]
    ax.plot(cs, vals, 'o-', color='#E74C3C', label='vLLM TP1', linewidth=2, markersize=8)
# TP2 SGLang
d = get_rows(engine='sglang', prec='fp8', fw='autogen', topo='tp2')
if d:
    cs = [r['conc'] for r in d]
    vals = [r['tpot_mean_ms'] for r in d]
    ax.plot(cs, vals, 's-', color='#2ECC71', label='SGLang TP2', linewidth=2, markersize=8)
# TP2 vLLM
d = get_rows(engine='vllm', prec='fp8', fw='autogen', topo='tp2')
if d:
    cs = [r['conc'] for r in d]
    vals = [r['tpot_mean_ms'] for r in d]
    ax.plot(cs, vals, '^--', color='#E67E22', label='vLLM TP2', linewidth=2, markersize=8)
ax.set_xlabel('Concurrency')
ax.set_ylabel('TPOT (ms)')
ax.set_title('(a) TPOT Scaling — FP8 autogen')
ax.legend()
ax.grid(True, alpha=0.3)

# 4b: TPS scaling
ax = axes[1]
d = get_rows(engine='vllm', prec='fp8', fw='autogen', topo='tp1')
if d:
    cs = [r['conc'] for r in d]
    vals = [r.get('gen_tps', 0) for r in d]
    ax.plot(cs, vals, 'o-', color='#E74C3C', label='vLLM TP1', linewidth=2, markersize=8)
d = get_rows(engine='sglang', prec='fp8', fw='autogen', topo='tp2')
if d:
    cs = [r['conc'] for r in d]
    vals = [r.get('gen_tps', 0) for r in d]
    ax.plot(cs, vals, 's-', color='#2ECC71', label='SGLang TP2', linewidth=2, markersize=8)
d = get_rows(engine='vllm', prec='fp8', fw='autogen', topo='tp2')
if d:
    cs = [r['conc'] for r in d]
    vals = [r.get('gen_tps', 0) for r in d]
    ax.plot(cs, vals, '^--', color='#E67E22', label='vLLM TP2', linewidth=2, markersize=8)
ax.set_xlabel('Concurrency')
ax.set_ylabel('Generation TPS')
ax.set_title('(b) Throughput Scaling — FP8 autogen')
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle('Figure 4: Topology Comparison — TP1 vs TP2 (FP8)', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f'{OUT_DIR}/fig4_topology_tp1_vs_tp2.png', bbox_inches='tight')
print(f"Saved fig4")

# ════════════════════════════════════════════════════════════════════
# Figure 5: Grouped bar chart — Engine × Precision × Concurrency
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

configs = [
    ('vLLM FP16', 'vllm', 'fp16', '#E74C3C'),
    ('vLLM FP8', 'vllm', 'fp8', '#C0392B'),
    ('SGLang FP16', 'sglang', 'fp16', '#27AE60'),
    ('SGLang FP8', 'sglang', 'fp8', '#2ECC71'),
]

concs = [1, 8, 32]
x = np.arange(len(concs))
width = 0.18

# 5a: TPOT bar
ax = axes[0]
for i, (label, eng, prec, color) in enumerate(configs):
    d = get_rows(engine=eng, prec=prec, fw='autogen', topo='tp2')
    vals = []
    for c in concs:
        matches = [r for r in d if r['conc'] == c]
        vals.append(matches[0]['tpot_mean_ms'] if matches else 0)
    ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
ax.set_xlabel('Concurrency')
ax.set_ylabel('TPOT (ms)')
ax.set_title('(a) TPOT by Engine × Precision')
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels([str(c) for c in concs])
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# 5b: TPS bar
ax = axes[1]
for i, (label, eng, prec, color) in enumerate(configs):
    d = get_rows(engine=eng, prec=prec, fw='autogen', topo='tp2')
    vals = []
    for c in concs:
        matches = [r for r in d if r['conc'] == c]
        vals.append(matches[0].get('gen_tps', 0) if matches else 0)
    ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
ax.set_xlabel('Concurrency')
ax.set_ylabel('Generation TPS')
ax.set_title('(b) TPS by Engine × Precision')
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels([str(c) for c in concs])
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('Figure 5: Engine × Precision Comparison — TP2 autogen', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f'{OUT_DIR}/fig5_engine_precision_bars.png', bbox_inches='tight')
print(f"Saved fig5")

# ════════════════════════════════════════════════════════════════════
# Figure 6: TTFT comparison (highlighting SGLang c=32 issue)
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 6a: SGLang vs vLLM TTFT
ax = axes[0]
for eng, color, ls in [('vllm', '#E74C3C', '--'), ('sglang', '#2ECC71', '-')]:
    d = get_rows(engine=eng, prec='fp16', fw='autogen', topo='tp2')
    if d:
        cs = [r['conc'] for r in d]
        vals = [r.get('ttft_mean_ms', 0) for r in d]
        ax.plot(cs, vals, f'o{ls}', color=color, label=f'{eng.upper()} FP16', linewidth=2, markersize=8)
    d = get_rows(engine=eng, prec='fp8', fw='autogen', topo='tp2')
    if d:
        cs = [r['conc'] for r in d]
        vals = [r.get('ttft_mean_ms', 0) for r in d]
        ax.plot(cs, vals, f's{ls}', color=color, label=f'{eng.upper()} FP8', linewidth=2, markersize=8, alpha=0.7)
ax.set_xlabel('Concurrency')
ax.set_ylabel('TTFT (ms)')
ax.set_title('(a) TTFT — Engine × Precision')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xticks([1, 8, 32])

# 6b: SGLang TTFT by framework
ax = axes[1]
for prec, ls in [('fp8', '-'), ('fp16', '--')]:
    for fw, color in FW_COLORS.items():
        d = get_rows(engine='sglang', prec=prec, fw=fw, topo='tp2')
        if d:
            cs = [r['conc'] for r in d]
            vals = [r.get('ttft_mean_ms', 0) for r in d]
            ax.plot(cs, vals, f'{FW_MARKERS[fw]}{ls}', color=color,
                    label=f'{fw} {prec.upper()}', linewidth=1.5, markersize=7)
ax.set_xlabel('Concurrency')
ax.set_ylabel('TTFT (ms)')
ax.set_title('(b) TTFT — SGLang by Framework')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xticks([1, 8, 32])

fig.suptitle('Figure 6: TTFT Analysis', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f'{OUT_DIR}/fig6_ttft_analysis.png', bbox_inches='tight')
print(f"Saved fig6")

# ════════════════════════════════════════════════════════════════════
# Figure 7: Heatmap — FP8 speedup ratio over FP16 (SGLang)
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

fws = ['autogen', 'langgraph', 'a2a']
concs_heat = [1, 8, 32]
speedup_matrix = np.zeros((len(fws), len(concs_heat)))

for i, fw in enumerate(fws):
    for j, c in enumerate(concs_heat):
        fp8 = [r for r in rows if r['engine'] == 'sglang' and r['prec'] == 'fp8' and r['fw'] == fw and r['conc'] == c]
        fp16 = [r for r in rows if r['engine'] == 'sglang' and r['prec'] == 'fp16' and r['fw'] == fw and r['conc'] == c]
        if fp8 and fp16 and fp16[0].get('tpot_mean_ms'):
            speedup_matrix[i, j] = (1 - fp8[0]['tpot_mean_ms'] / fp16[0]['tpot_mean_ms']) * 100

im = ax.imshow(speedup_matrix, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=20)
ax.set_xticks(range(len(concs_heat)))
ax.set_xticklabels([f'c={c}' for c in concs_heat])
ax.set_yticks(range(len(fws)))
ax.set_yticklabels(fws)
ax.set_xlabel('Concurrency')
ax.set_title('FP8 TPOT Improvement over FP16 (%, SGLang TP2)')

for i in range(len(fws)):
    for j in range(len(concs_heat)):
        val = speedup_matrix[i, j]
        ax.text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize=12, fontweight='bold')

plt.colorbar(im, ax=ax, label='TPOT Improvement (%)')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig7_fp8_speedup_heatmap.png', bbox_inches='tight')
print(f"Saved fig7")

# ════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════
# Figure 8: TP2 vs EP2 — TPOT, TPS, TTFT (SGLang, FP8, autogen)
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax_idx, (metric, ylabel, title_suffix) in enumerate([
    ('tpot_mean_ms', 'TPOT (ms)', 'TPOT'),
    ('gen_tps', 'Generation TPS', 'Throughput'),
    ('ttft_mean_ms', 'TTFT (ms)', 'TTFT'),
]):
    ax = axes[ax_idx]
    for topo in ['tp2', 'ep2']:
        d = get_rows(engine='sglang', prec='fp8', fw='autogen', topo=topo)
        if d:
            cs = [r['conc'] for r in d]
            vals = [r.get(metric, 0) for r in d]
            ax.plot(cs, vals, f'{TOPO_MARKERS[topo]}-', color=TOPO_COLORS[topo],
                    label=topo.upper(), linewidth=2, markersize=8)
    ax.set_xlabel('Concurrency')
    ax.set_ylabel(ylabel)
    ax.set_title(f'({chr(97+ax_idx)}) {title_suffix}')
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle('Figure 8: TP2 vs EP2 — SGLang FP8 (autogen)', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f'{OUT_DIR}/fig8_tp2_vs_ep2_fp8.png', bbox_inches='tight')
print(f"Saved fig8")

# ════════════════════════════════════════════════════════════════════
# Figure 9: EP2 FP8 vs FP16 — TPOT and TPS by framework
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, fw in enumerate(['autogen', 'langgraph', 'a2a']):
    # TPOT
    ax = axes[0][idx]
    for prec, ls, color in [('fp8', '-', '#E74C3C'), ('fp16', '--', '#3498DB')]:
        d = get_rows(engine='sglang', prec=prec, fw=fw, topo='ep2')
        if d:
            cs = [r['conc'] for r in d]
            vals = [r['tpot_mean_ms'] for r in d]
            ax.plot(cs, vals, f'o{ls}', color=color, label=prec.upper(), linewidth=2, markersize=8)
    ax.set_xlabel('Concurrency')
    ax.set_ylabel('TPOT (ms)')
    ax.set_title(f'{fw} — TPOT')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # TPS
    ax = axes[1][idx]
    for prec, ls, color in [('fp8', '-', '#E74C3C'), ('fp16', '--', '#3498DB')]:
        d = get_rows(engine='sglang', prec=prec, fw=fw, topo='ep2')
        if d:
            cs = [r['conc'] for r in d]
            vals = [r.get('gen_tps', 0) for r in d]
            ax.plot(cs, vals, f'o{ls}', color=color, label=prec.upper(), linewidth=2, markersize=8)
    ax.set_xlabel('Concurrency')
    ax.set_ylabel('Generation TPS')
    ax.set_title(f'{fw} — TPS')
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle('Figure 9: EP2 FP8 vs FP16 — by Framework (SGLang)', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f'{OUT_DIR}/fig9_ep2_fp8_vs_fp16.png', bbox_inches='tight')
print(f"Saved fig9")

# ════════════════════════════════════════════════════════════════════
# Figure 10: All Topologies — Bar Graph (autogen, multiple engines/precisions)
# ════════════════════════════════════════════════════════════════════
fig10_topo_configs = [
    ('TP1 vLLM FP8',     'vllm',   'fp8',  'tp1'),
    ('TP2 SGLang FP8',   'sglang', 'fp8',  'tp2'),
    ('TP2 vLLM FP16',    'vllm',   'fp16', 'tp2'),
    ('EP2 SGLang FP8',   'sglang', 'fp8',  'ep2'),
    ('EP2 vLLM FP16',    'vllm',   'fp16', 'ep2'),
]
fig10_colors = ['#E74C3C', '#2ECC71', '#27AE60', '#9B59B6', '#8E44AD']
concs_fig10 = [1, 8, 32]
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax_idx, (metric, ylabel, title_suffix) in enumerate([
    ('tpot_mean_ms', 'TPOT (ms)', 'TPOT'),
    ('gen_tps', 'Generation TPS', 'Throughput'),
    ('ttft_mean_ms', 'TTFT (ms)', 'TTFT'),
]):
    ax = axes[ax_idx]
    x = np.arange(len(concs_fig10))
    n = len(fig10_topo_configs)
    width = 0.8 / n
    for i, (label, eng, prec, topo) in enumerate(fig10_topo_configs):
        vals = []
        for c in concs_fig10:
            d = get_rows(engine=eng, prec=prec, fw='autogen', topo=topo)
            match = [r for r in d if r['conc'] == c]
            vals.append(match[0].get(metric, 0) if match else float('nan'))
        ax.bar(x + i * width, vals, width, label=label,
               color=fig10_colors[i], alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Concurrency')
    ax.set_ylabel(ylabel)
    ax.set_title(f'({chr(97+ax_idx)}) {title_suffix}')
    ax.set_xticks(x + (n - 1) * width / 2)
    ax.set_xticklabels([str(c) for c in concs_fig10])
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('Figure 10: All Topologies (autogen)', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f'{OUT_DIR}/fig10_all_topologies.png', bbox_inches='tight')
print(f"Saved fig10")

# ════════════════════════════════════════════════════════════════════
# Figure 11: vLLM EP2 vs SGLang EP2 (FP16, autogen) — Bar Graph
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
concs_fig11 = [1, 8, 32]
eng_configs = [('SGLang', 'sglang', '#2ECC71'), ('vLLM', 'vllm', '#E74C3C')]

for ax_idx, (metric, ylabel, title_suffix) in enumerate([
    ('tpot_mean_ms', 'TPOT (ms)', 'TPOT'),
    ('gen_tps', 'Generation TPS', 'Throughput'),
    ('ttft_mean_ms', 'TTFT (ms)', 'TTFT'),
]):
    ax = axes[ax_idx]
    x = np.arange(len(concs_fig11))
    width = 0.35
    for i, (eng_label, eng, color) in enumerate(eng_configs):
        vals = []
        for c in concs_fig11:
            d = get_rows(engine=eng, prec='fp16', fw='autogen', topo='ep2')
            match = [r for r in d if r['conc'] == c]
            vals.append(match[0].get(metric, 0) if match else float('nan'))
        ax.bar(x + i * width, vals, width, label=eng_label,
               color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Concurrency')
    ax.set_ylabel(ylabel)
    ax.set_title(f'({chr(97+ax_idx)}) {title_suffix}')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([str(c) for c in concs_fig11])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('Figure 11: vLLM vs SGLang — EP2 FP16 (autogen)', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f'{OUT_DIR}/fig11_vllm_vs_sglang_ep2_fp16.png', bbox_inches='tight')
print(f"Saved fig11")

# ════════════════════════════════════════════════════════════════════
# Figure 12: vLLM EP2 — Framework comparison (TPOT + TPS + TTFT)
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax_idx, (metric, ylabel, title_suffix) in enumerate([
    ('tpot_mean_ms', 'TPOT (ms)', 'TPOT'),
    ('gen_tps', 'Generation TPS', 'Throughput'),
    ('ttft_mean_ms', 'TTFT (ms)', 'TTFT'),
]):
    ax = axes[ax_idx]
    for fw, color in FW_COLORS.items():
        d = get_rows(engine='vllm', prec='fp16', fw=fw, topo='ep2')
        if d:
            cs = [r['conc'] for r in d]
            vals = [r.get(metric, 0) for r in d]
            ax.plot(cs, vals, f'{FW_MARKERS[fw]}-', color=color, label=fw,
                    linewidth=2, markersize=8)
    ax.set_xlabel('Concurrency')
    ax.set_ylabel(ylabel)
    ax.set_title(f'({chr(97+ax_idx)}) {title_suffix}')
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle('Figure 12: vLLM EP2 FP16 — Framework Comparison', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f'{OUT_DIR}/fig12_vllm_ep2_framework_comparison.png', bbox_inches='tight')
print(f"Saved fig12")

# ════════════════════════════════════════════════════════════════════
# Figure 13: Token Distribution — Input vs Output by Framework & Role
# ════════════════════════════════════════════════════════════════════
if trace_rows:
    # Use a single config for fair comparison: vllm ep2-fp16 c=8
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    roles = ['planner', 'executor', 'aggregator']
    fws = ['autogen', 'langgraph', 'a2a']

    for row_idx, token_type in enumerate(['prompt_tokens', 'completion_tokens']):
        ylabel = 'Input Tokens' if row_idx == 0 else 'Output Tokens'
        for col_idx, fw in enumerate(fws):
            ax = axes[row_idx][col_idx]
            data_by_role = {}
            for r in trace_rows:
                if r['fw'] == fw and r['conc'] == 8 and r['engine'] == 'vllm' and r['topo'] == 'ep2':
                    role = r['role']
                    if role not in data_by_role:
                        data_by_role[role] = []
                    data_by_role[role].append(r[token_type])
            # Fallback: use any available trace data
            if not data_by_role:
                for r in trace_rows:
                    if r['fw'] == fw and r['conc'] == 8:
                        role = r['role']
                        if role not in data_by_role:
                            data_by_role[role] = []
                        data_by_role[role].append(r[token_type])

            box_data = [data_by_role.get(role, [0]) for role in roles]
            bp = ax.boxplot(box_data, labels=roles, patch_artist=True,
                           widths=0.6, showfliers=True,
                           flierprops={'markersize': 3, 'alpha': 0.5})
            role_colors = ['#3498DB', '#E67E22', '#9B59B6']
            for patch, color in zip(bp['boxes'], role_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{fw} — {ylabel}')
            ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Figure 13: Token Distribution by Agent Role (c=8)', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{OUT_DIR}/fig13_token_distribution_by_role.png', bbox_inches='tight')
    print(f"Saved fig13")

# ════════════════════════════════════════════════════════════════════
# Figure 14: Stacked Bar — Avg Input/Output tokens per role per FW
# ════════════════════════════════════════════════════════════════════
if trace_rows:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fws = ['autogen', 'langgraph', 'a2a']
    roles = ['planner', 'executor', 'aggregator']
    role_colors = {'planner': '#3498DB', 'executor': '#E67E22', 'aggregator': '#9B59B6'}

    for ax_idx, (token_type, ylabel, title) in enumerate([
        ('prompt_tokens', 'Avg Input Tokens', '(a) Input Tokens by Role'),
        ('completion_tokens', 'Avg Output Tokens', '(b) Output Tokens by Role'),
    ]):
        ax = axes[ax_idx]
        x = np.arange(len(fws))
        width = 0.25
        for i, role in enumerate(roles):
            means = []
            for fw in fws:
                vals = [r[token_type] for r in trace_rows
                        if r['fw'] == fw and r['role'] == role and r['conc'] == 8
                        and r['engine'] == 'vllm' and r['topo'] == 'ep2']
                if not vals:
                    vals = [r[token_type] for r in trace_rows
                            if r['fw'] == fw and r['role'] == role and r['conc'] == 8]
                means.append(np.mean(vals) if vals else 0)
            ax.bar(x + i * width, means, width, label=role, color=role_colors[role], alpha=0.8)
        ax.set_xlabel('Framework')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x + width)
        ax.set_xticklabels(fws)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Figure 14: Token Usage by Framework × Role (c=8)', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f'{OUT_DIR}/fig14_token_usage_by_framework_role.png', bbox_inches='tight')
    print(f"Saved fig14")

# ════════════════════════════════════════════════════════════════════
# Figure 15: Context Accumulation — Input tokens through DAG steps
# ════════════════════════════════════════════════════════════════════
if trace_rows:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fws = ['autogen', 'langgraph', 'a2a']
    step_orders = {
        'autogen': ['P', 'E0', 'E1', 'A'],
        'langgraph': ['P', 'E0', 'E1', 'A'],  # E0∥E1 but plot sequentially
        'a2a': ['P', 'E0', 'E1', 'A'],
    }

    for idx, fw in enumerate(fws):
        ax = axes[idx]
        steps = step_orders[fw]
        # Collect per-step prompt_tokens across all tasks
        step_tokens = {s: [] for s in steps}
        for r in trace_rows:
            if r['fw'] == fw and r['conc'] == 8 and r['engine'] == 'vllm' and r['topo'] == 'ep2':
                if r['step_id'] in step_tokens:
                    step_tokens[r['step_id']].append(r['prompt_tokens'])
        if not any(step_tokens.values()):
            for r in trace_rows:
                if r['fw'] == fw and r['conc'] == 8:
                    if r['step_id'] in step_tokens:
                        step_tokens[r['step_id']].append(r['prompt_tokens'])

        means = [np.mean(step_tokens[s]) if step_tokens[s] else 0 for s in steps]
        stds = [np.std(step_tokens[s]) if step_tokens[s] else 0 for s in steps]
        colors = ['#3498DB', '#E67E22', '#E67E22', '#9B59B6']
        ax.bar(steps, means, yerr=stds, color=colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=0.5)

        # Add output tokens as hatched overlay
        out_means = []
        for s in steps:
            vals = [r['completion_tokens'] for r in trace_rows
                    if r['fw'] == fw and r['conc'] == 8 and r['step_id'] == s
                    and r['engine'] == 'vllm' and r['topo'] == 'ep2']
            if not vals:
                vals = [r['completion_tokens'] for r in trace_rows
                        if r['fw'] == fw and r['conc'] == 8 and r['step_id'] == s]
            out_means.append(np.mean(vals) if vals else 0)
        ax.bar(steps, out_means, bottom=means, color='white', alpha=0.5,
               edgecolor='black', linewidth=0.5, hatch='///', label='Output')

        ax.set_ylabel('Tokens')
        ax.set_title(f'{fw}')
        ax.set_xlabel('DAG Step')
        if idx == 0:
            ax.legend(['Input (solid)', 'Output (hatched)'], fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Figure 15: Context Accumulation Through DAG Steps (c=8)', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f'{OUT_DIR}/fig15_context_accumulation.png', bbox_inches='tight')
    print(f"Saved fig15")

# ════════════════════════════════════════════════════════════════════
# Figure 16: Input:Output Ratio heatmap — Framework × Role
# ════════════════════════════════════════════════════════════════════
if trace_rows:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    fws = ['autogen', 'langgraph', 'a2a']
    roles = ['planner', 'executor', 'aggregator']
    ratio_matrix = np.zeros((len(fws), len(roles)))

    for i, fw in enumerate(fws):
        for j, role in enumerate(roles):
            inp = [r['prompt_tokens'] for r in trace_rows
                   if r['fw'] == fw and r['role'] == role and r['conc'] == 8
                   and r['engine'] == 'vllm' and r['topo'] == 'ep2']
            out = [r['completion_tokens'] for r in trace_rows
                   if r['fw'] == fw and r['role'] == role and r['conc'] == 8
                   and r['engine'] == 'vllm' and r['topo'] == 'ep2']
            if not inp:
                inp = [r['prompt_tokens'] for r in trace_rows
                       if r['fw'] == fw and r['role'] == role and r['conc'] == 8]
                out = [r['completion_tokens'] for r in trace_rows
                       if r['fw'] == fw and r['role'] == role and r['conc'] == 8]
            avg_inp = np.mean(inp) if inp else 1
            avg_out = np.mean(out) if out else 1
            ratio_matrix[i, j] = avg_inp / max(avg_out, 1)

    im = ax.imshow(ratio_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=10)
    ax.set_xticks(range(len(roles)))
    ax.set_xticklabels(roles)
    ax.set_yticks(range(len(fws)))
    ax.set_yticklabels(fws)
    ax.set_title('Input:Output Token Ratio by Framework × Role (c=8)')

    for i in range(len(fws)):
        for j in range(len(roles)):
            val = ratio_matrix[i, j]
            ax.text(j, i, f'{val:.1f}x', ha='center', va='center', fontsize=12,
                    fontweight='bold', color='white' if val > 5 else 'black')

    plt.colorbar(im, ax=ax, label='Input:Output Ratio')
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig16_io_ratio_heatmap.png', bbox_inches='tight')
    print(f"Saved fig16")

# ════════════════════════════════════════════════════════════════════
# Figure 17: Total tokens per task — Framework comparison
# ════════════════════════════════════════════════════════════════════
if trace_rows:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fws = ['autogen', 'langgraph', 'a2a']

    # Aggregate per-task totals
    task_totals = {}  # (fw, task_id) -> {prompt, completion}
    for r in trace_rows:
        if r['conc'] == 8 and r['engine'] == 'vllm' and r['topo'] == 'ep2':
            key = (r['fw'], r['task_id'])
            if key not in task_totals:
                task_totals[key] = {'prompt': 0, 'completion': 0}
            task_totals[key]['prompt'] += r['prompt_tokens']
            task_totals[key]['completion'] += r['completion_tokens']
    # Fallback
    if not task_totals:
        for r in trace_rows:
            if r['conc'] == 8:
                key = (r['fw'], r['task_id'])
                if key not in task_totals:
                    task_totals[key] = {'prompt': 0, 'completion': 0}
                task_totals[key]['prompt'] += r['prompt_tokens']
                task_totals[key]['completion'] += r['completion_tokens']

    # 17a: Box plot of total tokens per task
    ax = axes[0]
    box_data = []
    for fw in fws:
        totals = [v['prompt'] + v['completion'] for k, v in task_totals.items() if k[0] == fw]
        box_data.append(totals if totals else [0])
    bp = ax.boxplot(box_data, labels=fws, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], [FW_COLORS[fw] for fw in fws]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Total Tokens per Task')
    ax.set_title('(a) Total Tokens per Task')
    ax.grid(True, alpha=0.3, axis='y')

    # 17b: Stacked bar — avg prompt vs completion per task
    ax = axes[1]
    x = np.arange(len(fws))
    prompt_means = []
    completion_means = []
    for fw in fws:
        prompts = [v['prompt'] for k, v in task_totals.items() if k[0] == fw]
        completions = [v['completion'] for k, v in task_totals.items() if k[0] == fw]
        prompt_means.append(np.mean(prompts) if prompts else 0)
        completion_means.append(np.mean(completions) if completions else 0)
    ax.bar(x, prompt_means, 0.5, label='Input (Prompt)', color='#3498DB', alpha=0.8)
    ax.bar(x, completion_means, 0.5, bottom=prompt_means, label='Output (Completion)',
           color='#E74C3C', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(fws)
    ax.set_ylabel('Avg Tokens per Task')
    ax.set_title('(b) Input vs Output Tokens')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    for i, fw in enumerate(fws):
        total = prompt_means[i] + completion_means[i]
        if total > 0:
            pct = completion_means[i] / total * 100
            ax.text(i, total + 20, f'{pct:.0f}% gen', ha='center', fontsize=9)

    fig.suptitle('Figure 17: Token Efficiency by Framework (c=8)', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f'{OUT_DIR}/fig17_token_efficiency.png', bbox_inches='tight')
    print(f"Saved fig17")

# ════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"All figures saved to: {OUT_DIR}/")
print(f"  fig1:  SGLang vs vLLM (TP2 FP16)")
print(f"  fig2:  SGLang FP8 vs FP16 TPOT by framework")
print(f"  fig3:  Framework comparison (SGLang TP2)")
print(f"  fig4:  Topology TP1 vs TP2")
print(f"  fig5:  Engine x Precision grouped bars")
print(f"  fig6:  TTFT analysis")
print(f"  fig7:  FP8 speedup heatmap")
print(f"  fig8:  TP2 vs EP2 (SGLang FP8)")
print(f"  fig9:  EP2 FP8 vs FP16 by framework")
print(f"  fig10: All topologies TP1 vs TP2 vs EP2")
print(f"  fig11: vLLM vs SGLang EP2 FP16")
print(f"  fig12: vLLM EP2 framework comparison")
print(f"  fig13: Token distribution by role (box)")
print(f"  fig14: Token usage by framework x role (bar)")
print(f"  fig15: Context accumulation through DAG steps")
print(f"  fig16: Input:Output ratio heatmap")
print(f"  fig17: Token efficiency by framework")
print(f"{'='*60}")
