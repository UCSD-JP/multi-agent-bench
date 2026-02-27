#!/usr/bin/env python3
"""Build a standardized real-experiment naming map + status tracker (H100/H200)."""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path("/home/jp/paper_resource/multi-agent-bench")
GPUSIM = Path("/home/jp/paper_resource/gpusim")
OUT_DIR = ROOT / "results_multiagent" / "analysis_4gpu8gpu"
ALIAS_DIR = ROOT / "results_multiagent" / "real_named"


@dataclass
class Spec:
    exp_id: str
    canonical_name: str
    hardware: str
    gpu_count: int
    interconnect: str
    engine: str
    topology: str
    precision: str
    workload: str
    frameworks: str
    required_concurrency: str
    source_paths: list[str]
    notes: str


def _parse_c_from_path(path: str) -> int | None:
    m = re.search(r"_c(\d+)", path)
    return int(m.group(1)) if m else None


def _iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        if not p.exists():
            continue
        if p.is_file():
            yield p
            continue
        for f in p.rglob("*"):
            if f.is_file():
                yield f


def _read_required_cs(spec: Spec) -> list[int]:
    if not spec.required_concurrency.strip():
        return []
    return [int(x) for x in spec.required_concurrency.split(",")]


def _collect(spec: Spec) -> dict:
    src_paths = [Path(p) for p in spec.source_paths]
    missing = [str(p) for p in src_paths if not p.exists()]
    files = list(_iter_files(src_paths))

    trace_cs: set[int] = set()
    server_cs: set[int] = set()
    valid_server_cs: set[int] = set()
    summary_cs: set[int] = set()
    has_batch_summary = False
    has_nccl_summary = False
    latest_mtime = None

    for f in files:
        latest_mtime = f.stat().st_mtime if latest_mtime is None else max(latest_mtime, f.stat().st_mtime)
        ps = f.as_posix()
        name = f.name

        c = _parse_c_from_path(ps)
        if name.startswith("trace_tasks") and c is not None:
            trace_cs.add(c)

        if name.startswith("server_metrics") and name.endswith(".json"):
            if c is not None:
                server_cs.add(c)
            try:
                d = json.loads(f.read_text())
                tpot_count = int(d.get("tpot_count") or 0)
                ttft_count = int(d.get("ttft_count") or 0)
                if (tpot_count > 0 or ttft_count > 0) and c is not None:
                    valid_server_cs.add(c)
            except Exception:
                pass

        if "batch_sweep" in ps and name == "summary.json":
            has_batch_summary = True

        if name == "nccl_allreduce_summary.json":
            has_nccl_summary = True

        if "summary" in name and name.endswith(".json"):
            try:
                d = json.loads(f.read_text())
            except Exception:
                d = None
            if isinstance(d, list):
                for row in d:
                    if isinstance(row, dict) and "concurrency" in row:
                        try:
                            summary_cs.add(int(row["concurrency"]))
                        except Exception:
                            pass

    req = _read_required_cs(spec)
    req_set = set(req)
    status = "unknown"

    if missing:
        status = "missing"
    elif spec.workload == "nccl":
        status = "complete" if has_nccl_summary else "missing"
    elif req_set and req_set.issubset(valid_server_cs):
        status = "complete"
    elif req_set and req_set.issubset(summary_cs) and not valid_server_cs:
        status = "complete_summary_only"
    elif req_set and req_set.issubset(trace_cs) and not valid_server_cs:
        status = "partial_trace_only"
    elif has_batch_summary and not valid_server_cs:
        status = "partial_batch_only"
    elif valid_server_cs:
        status = "partial_missing_concurrency"
    else:
        status = "partial"

    if status == "complete":
        next_step = "-"
    elif status == "complete_summary_only":
        next_step = "Save per-run server_metrics + trace for reproducibility"
    elif status == "partial_trace_only":
        next_step = "Re-run same c set and persist valid server_metrics (tpot_count>0)"
    elif status == "partial_batch_only":
        next_step = "Add agentic sweep (c=1,8,32,64,128) with valid server_metrics"
    elif status == "partial_missing_concurrency":
        miss = sorted(req_set - valid_server_cs)
        next_step = f"Fill missing valid server_metrics for c={miss}"
    elif status == "missing":
        next_step = "Run from scratch for this hardware/topology condition"
    else:
        next_step = "Investigate artifacts and rerun with standardized output"

    latest_utc = (
        datetime.fromtimestamp(latest_mtime, tz=timezone.utc).isoformat() if latest_mtime is not None else ""
    )

    return {
        "exp_id": spec.exp_id,
        "canonical_name": spec.canonical_name,
        "hardware": spec.hardware,
        "gpu_count": spec.gpu_count,
        "interconnect": spec.interconnect,
        "engine": spec.engine,
        "topology": spec.topology,
        "precision": spec.precision,
        "workload": spec.workload,
        "frameworks": spec.frameworks,
        "required_concurrency": spec.required_concurrency,
        "trace_concurrency_found": ",".join(str(x) for x in sorted(trace_cs)),
        "server_concurrency_found": ",".join(str(x) for x in sorted(server_cs)),
        "valid_server_concurrency_found": ",".join(str(x) for x in sorted(valid_server_cs)),
        "summary_concurrency_found": ",".join(str(x) for x in sorted(summary_cs)),
        "has_batch_summary": has_batch_summary,
        "has_nccl_summary": has_nccl_summary,
        "status": status,
        "next_step": next_step,
        "latest_artifact_utc": latest_utc,
        "source_paths": " | ".join(spec.source_paths),
        "notes": spec.notes,
    }


def _write_aliases(specs: list[Spec]) -> None:
    ALIAS_DIR.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        dst = ALIAS_DIR / spec.canonical_name
        dst.mkdir(parents=True, exist_ok=True)
        # Remove stale links first.
        for old in dst.glob("src*"):
            if old.is_symlink() or old.exists():
                old.unlink()
        for i, src in enumerate(spec.source_paths, start=1):
            link = dst / f"src{i}"
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(src, link)


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _write_md(rows: list[dict], path: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    lines: list[str] = []
    lines.append("# Real Experiment Tracker (H100/H200)")
    lines.append("")
    lines.append(f"- Updated (UTC): `{now}`")
    lines.append("- Naming format: `real__{gpu}__{ng}g__{fabric}__{engine}__{topology}__{precision}__{workload}`")
    lines.append(f"- Alias root: `{ALIAS_DIR}`")
    lines.append("")
    lines.append("## Status Table")
    lines.append("")
    lines.append("| Experiment | HW | Cond | Required c | Valid server c | Status | Next step |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        cond = (
            f"{r['interconnect']} / {r['engine']} / {r['topology']} / "
            f"{r['precision']} / {r['workload']} / {r['frameworks']}"
        )
        lines.append(
            f"| `{r['canonical_name']}` | {r['hardware']}x{r['gpu_count']} | {cond} | "
            f"{r['required_concurrency'] or '-'} | {r['valid_server_concurrency_found'] or '-'} | "
            f"**{r['status']}** | {r['next_step']} |"
        )
    lines.append("")
    lines.append("## Naming Map")
    lines.append("")
    lines.append("| Canonical Folder | Sources |")
    lines.append("|---|---|")
    for r in rows:
        lines.append(f"| `{r['canonical_name']}` | `{r['source_paths']}` |")
    lines.append("")
    lines.append("## Needed Experiments (Priority)")
    lines.append("")
    needed = [r for r in rows if r["status"] != "complete"]
    if not needed:
        lines.append("- None")
    else:
        for r in needed:
            lines.append(f"- `{r['canonical_name']}`: {r['next_step']}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    specs = [
        Spec(
            exp_id="H100-2G-TP2-FP16-SGLANG",
            canonical_name="real__h100__2g__sysphb__sglang__tp2__fp16__agentic_3fw",
            hardware="H100",
            gpu_count=2,
            interconnect="SYS/PHB(no_p2p)",
            engine="sglang",
            topology="TP2",
            precision="FP16",
            workload="agentic",
            frameworks="autogen/langgraph/a2a",
            required_concurrency="1,8,32",
            source_paths=[
                str(ROOT / "results_multiagent" / "sweep_tp2-fp16_autogen"),
                str(ROOT / "results_multiagent" / "sweep_tp2-fp16_langgraph"),
                str(ROOT / "results_multiagent" / "sweep_tp2-fp16_a2a"),
            ],
            notes="Paladin 2xH100 baseline",
        ),
        Spec(
            exp_id="H100-2G-TP2-FP8-SGLANG",
            canonical_name="real__h100__2g__sysphb__sglang__tp2__fp8__agentic_3fw",
            hardware="H100",
            gpu_count=2,
            interconnect="SYS/PHB(no_p2p)",
            engine="sglang",
            topology="TP2",
            precision="FP8",
            workload="agentic",
            frameworks="autogen/langgraph/a2a",
            required_concurrency="1,8,32",
            source_paths=[
                str(ROOT / "results_multiagent" / "sweep_tp2-fp8_autogen"),
                str(ROOT / "results_multiagent" / "sweep_tp2-fp8_langgraph"),
                str(ROOT / "results_multiagent" / "sweep_tp2-fp8_a2a"),
            ],
            notes="Paladin 2xH100 baseline",
        ),
        Spec(
            exp_id="H100-2G-EP2-FP16-SGLANG",
            canonical_name="real__h100__2g__sysphb__sglang__dp2_ep2__fp16__agentic_3fw",
            hardware="H100",
            gpu_count=2,
            interconnect="SYS/PHB(no_p2p)",
            engine="sglang",
            topology="DP2xEP2",
            precision="FP16",
            workload="agentic",
            frameworks="autogen/langgraph/a2a",
            required_concurrency="1,8,32,64",
            source_paths=[
                str(ROOT / "results_multiagent" / "sweep_dp2-ep2-fp16_autogen"),
                str(ROOT / "results_multiagent" / "sweep_dp2-ep2-fp16_langgraph"),
                str(ROOT / "results_multiagent" / "sweep_dp2-ep2-fp16_a2a"),
            ],
            notes="Paladin 2xH100 baseline",
        ),
        Spec(
            exp_id="H100-2G-EP2-FP8-SGLANG",
            canonical_name="real__h100__2g__sysphb__sglang__dp2_ep2__fp8__agentic_3fw",
            hardware="H100",
            gpu_count=2,
            interconnect="SYS/PHB(no_p2p)",
            engine="sglang",
            topology="DP2xEP2",
            precision="FP8",
            workload="agentic",
            frameworks="autogen/langgraph/a2a",
            required_concurrency="1,8,32,64",
            source_paths=[
                str(ROOT / "results_multiagent" / "sweep_dp2-ep2-fp8_autogen"),
                str(ROOT / "results_multiagent" / "sweep_dp2-ep2-fp8_langgraph"),
                str(ROOT / "results_multiagent" / "sweep_dp2-ep2-fp8_a2a"),
            ],
            notes="Paladin 2xH100 baseline",
        ),
        Spec(
            exp_id="H100-2G-EP2-FP16-VLLM",
            canonical_name="real__h100__2g__sysphb__vllm__dp2_ep2__fp16__agentic_3fw",
            hardware="H100",
            gpu_count=2,
            interconnect="SYS/PHB(no_p2p)",
            engine="vllm",
            topology="DP2xEP2",
            precision="FP16",
            workload="agentic",
            frameworks="autogen/langgraph/a2a",
            required_concurrency="1,8,32,64",
            source_paths=[
                str(ROOT / "results_multiagent" / "sweep_vllm-ep2-fp16_autogen"),
                str(ROOT / "results_multiagent" / "sweep_vllm-ep2-fp16_langgraph"),
                str(ROOT / "results_multiagent" / "sweep_vllm-ep2-fp16_a2a"),
            ],
            notes="Paladin 2xH100 baseline",
        ),
        Spec(
            exp_id="H100-2G-TP2-FP8-VLLM",
            canonical_name="real__h100__2g__sysphb__vllm__tp2__fp8__agentic_autogen",
            hardware="H100",
            gpu_count=2,
            interconnect="SYS/PHB(no_p2p)",
            engine="vllm",
            topology="TP2",
            precision="FP8",
            workload="agentic",
            frameworks="autogen",
            required_concurrency="1,8,32,64,128",
            source_paths=[str(ROOT / "results_multiagent" / "conc_sweep")],
            notes="Autogen only, includes c=200 test artifact",
        ),
        Spec(
            exp_id="H100-4G-TP4-FP16",
            canonical_name="real__h100__4g__pcie__unknown__tp4__fp16__agentic_autogen",
            hardware="H100",
            gpu_count=4,
            interconnect="PCIe",
            engine="unknown",
            topology="TP4",
            precision="FP16",
            workload="agentic",
            frameworks="autogen",
            required_concurrency="1,8,32,64",
            source_paths=[str(GPUSIM / "results_from_real_H100" / "tp4_fp16_4xH100_PCIe")],
            notes="summary json exists, per-run server_metrics files absent",
        ),
        Spec(
            exp_id="H200-8G-TP8-FP16",
            canonical_name="real__h200__8g__nvlink__unknown__tp8__fp16__agentic_autogen",
            hardware="H200",
            gpu_count=8,
            interconnect="NVLink",
            engine="unknown",
            topology="TP8",
            precision="FP16",
            workload="agentic",
            frameworks="autogen",
            required_concurrency="1,8,32,64,128",
            source_paths=[str(GPUSIM / "results_from_real_H100" / "tp8_fp16_8xH200_NVLink")],
            notes="r1-r3 repeats exist",
        ),
        Spec(
            exp_id="H200-8G-TP8EP-BF16",
            canonical_name="real__h200__8g__nvlink__unknown__tp8_ep__bf16__agentic_autogen",
            hardware="H200",
            gpu_count=8,
            interconnect="NVLink",
            engine="unknown",
            topology="TP8+EP",
            precision="BF16",
            workload="agentic",
            frameworks="autogen",
            required_concurrency="1,8,32,64,128",
            source_paths=[str(GPUSIM / "results_from_real_H100" / "tp8ep_bf16_8xH200_NVLink")],
            notes="r1-r3 repeats exist",
        ),
        Spec(
            exp_id="H200-8G-TP4DP2-FP16",
            canonical_name="real__h200__8g__nvlink__unknown__tp4_dp2__fp16__agentic_autogen",
            hardware="H200",
            gpu_count=8,
            interconnect="NVLink",
            engine="unknown",
            topology="TP4+DP2",
            precision="FP16",
            workload="agentic",
            frameworks="autogen",
            required_concurrency="1,8,32,64,128",
            source_paths=[str(GPUSIM / "results_from_real_H100" / "tp4dp2_fp16_8xH200_NVLink")],
            notes="trace exists for c=1/8/32/64/128, server_metrics only c=1 fail",
        ),
        Spec(
            exp_id="H200-8G-TP4DP2EP-BF16",
            canonical_name="real__h200__8g__nvlink__unknown__tp4_dp2_ep__bf16__agentic_autogen",
            hardware="H200",
            gpu_count=8,
            interconnect="NVLink",
            engine="unknown",
            topology="TP4+DP2+EP",
            precision="BF16",
            workload="agentic",
            frameworks="autogen",
            required_concurrency="1,8,32,64,128",
            source_paths=[str(GPUSIM / "results_from_real_H100" / "tp4dp2ep_bf16_8xH200_NVLink")],
            notes="trace exists for c=1/8/32/64/128, server_metrics only c=1 fail",
        ),
        Spec(
            exp_id="H200-8G-TP2DP4EP-BF16",
            canonical_name="real__h200__8g__nvlink__unknown__tp2_dp4_ep__bf16__agentic_autogen",
            hardware="H200",
            gpu_count=8,
            interconnect="NVLink",
            engine="unknown",
            topology="TP2+DP4+EP",
            precision="BF16",
            workload="agentic",
            frameworks="autogen",
            required_concurrency="1,8,32,64,128",
            source_paths=[str(GPUSIM / "results_from_real_H100" / "tp2dp4ep_bf16_8xH200_NVLink")],
            notes="trace exists for c=1/8/32/64/128, server_metrics only c=1 fail",
        ),
        Spec(
            exp_id="H200-8G-NCCL",
            canonical_name="real__h200__8g__nvlink__na__na__na__nccl",
            hardware="H200",
            gpu_count=8,
            interconnect="NVLink",
            engine="na",
            topology="na",
            precision="na",
            workload="nccl",
            frameworks="na",
            required_concurrency="",
            source_paths=[str(GPUSIM / "results_from_real_H100" / "nccl_h200_8gpu_nvlink")],
            notes="nccl_allreduce_summary.json exists",
        ),
        Spec(
            exp_id="H100-4G-NCCL",
            canonical_name="real__h100__4g__pcie__na__na__na__nccl",
            hardware="H100",
            gpu_count=4,
            interconnect="PCIe",
            engine="na",
            topology="na",
            precision="na",
            workload="nccl",
            frameworks="na",
            required_concurrency="",
            source_paths=[str(GPUSIM / "results_from_real_H100" / "nccl_h100_4gpu_pcie")],
            notes="target artifact currently missing",
        ),
        Spec(
            exp_id="H200-4G-TP4-FP16",
            canonical_name="real__h200__4g__nvlink_or_pcie__unknown__tp4__fp16__agentic_autogen",
            hardware="H200",
            gpu_count=4,
            interconnect="NVLink_or_PCIe",
            engine="unknown",
            topology="TP4",
            precision="FP16",
            workload="agentic",
            frameworks="autogen",
            required_concurrency="1,8,32,64,128",
            source_paths=[str(GPUSIM / "results_from_real_H100" / "tp4_fp16_4xH200_unknown")],
            notes="needed for 4xH200 vs 8xH200 scaling isolation",
        ),
        Spec(
            exp_id="H100-2G-TP2-FP16-P2P",
            canonical_name="real__h100__2g__pcie_p2p_or_nvlink__unknown__tp2__fp16__agentic_autogen",
            hardware="H100",
            gpu_count=2,
            interconnect="PCIe_P2P_or_NVLink",
            engine="unknown",
            topology="TP2",
            precision="FP16",
            workload="agentic",
            frameworks="autogen",
            required_concurrency="1,8,32,64",
            source_paths=[str(GPUSIM / "results_from_real_H100" / "tp2_fp16_2xH100_p2p_or_nvlink")],
            notes="needed to isolate SYS/PHB vs P2P/NVLink communication effect",
        ),
    ]

    rows = [_collect(s) for s in specs]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_aliases(specs)
    _write_csv(rows, OUT_DIR / "real_experiment_tracker.csv")
    _write_md(rows, OUT_DIR / "real_experiment_tracker.md")
    print(f"Wrote: {OUT_DIR / 'real_experiment_tracker.csv'}")
    print(f"Wrote: {OUT_DIR / 'real_experiment_tracker.md'}")
    print(f"Alias root: {ALIAS_DIR}")


if __name__ == "__main__":
    main()
