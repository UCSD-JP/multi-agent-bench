#!/usr/bin/env python3
"""Phase A data integrity closure for agentic benchmark results.

Builds canonical + quality-gated artifacts from sweep run outputs.

Outputs:
  - canonical_comparison.csv
  - quality_report.csv
  - dropped_rows_with_reason.csv
  - rerun_targets.csv
  - phaseA_summary.md
  - phaseA_policy.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PRIMARY_METRICS = [
    "server_tpot_mean_ms",
    "server_ttft_mean_ms",
    "server_tps_mean",
]

META_FIELDS = [
    "engine",
    "model",
    "precision",
    "parallelism_config",
    "topology",
    "framework",
    "concurrency",
]

CONDITION_KEYS = ["engine", "parallelism_config", "precision", "framework", "concurrency"]


@dataclass
class DropRow:
    source_path: str
    run_id: str
    reason: str
    detail: str


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values, dtype=float), q))


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.array(values, dtype=float)))


def resolve_path_safe(path_text: str) -> str:
    try:
        return str(Path(path_text).resolve())
    except Exception:
        return path_text


def infer_topology(path_text: str, default_topology: str) -> tuple[str, str]:
    low = path_text.lower()
    if "nvlink" in low:
        return "NVLink", "path_keyword"
    if "pcie" in low or "sys" in low or "phb" in low:
        return "PCIe/SYS", "path_keyword"
    if "p2p" in low:
        return "P2P", "path_keyword"
    if "cxl" in low:
        return "CXL", "path_keyword"
    return default_topology, "default_assumption"


def canonical_parallelism(tokens: list[str]) -> str:
    norm = [t.upper() for t in tokens if t]
    if not norm:
        return "UNKNOWN"
    if len(norm) == 1:
        return norm[0]
    return "x".join(norm)


def extract_dp_size(parallelism_config: str) -> int:
    up = str(parallelism_config).upper()
    matches = re.findall(r"DP(\d+)", up)
    if not matches:
        return 1
    try:
        return max(int(x) for x in matches)
    except Exception:
        return 1


def parse_sweep_metadata(sweep_name: str) -> dict[str, str] | None:
    if not sweep_name.startswith("sweep_"):
        return None
    core = sweep_name[len("sweep_") :]
    if "_" not in core:
        return None
    stack, framework_hint = core.rsplit("_", 1)
    tokens = stack.split("-")
    if len(tokens) < 2:
        return None

    engine = "sglang"
    if tokens[0] in {"sglang", "vllm"}:
        engine = tokens[0]
        tokens = tokens[1:]
    if len(tokens) < 2:
        return None

    precision = tokens[-1].lower()
    parallelism_config = canonical_parallelism(tokens[:-1])
    return {
        "engine": engine,
        "precision": precision,
        "parallelism_config": parallelism_config,
        "framework_hint": framework_hint.lower(),
    }


def parse_framework_and_concurrency(server_metrics_name: str) -> tuple[str | None, int | None]:
    m = re.match(r"server_metrics_([a-zA-Z0-9_-]+)_c(\d+)\.json$", server_metrics_name)
    if not m:
        return None, None
    return m.group(1).lower(), int(m.group(2))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_trace_stats(path: Path) -> dict[str, float]:
    makespans: list[float] = []
    task_input_tokens: list[float] = []
    task_output_tokens: list[float] = []
    dispatch_waits: list[float] = []
    total_requests = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            makespan = row.get("makespan_ms")
            if isinstance(makespan, (int, float)):
                makespans.append(float(makespan))

            steps = row.get("steps", {})
            if isinstance(steps, dict):
                total_requests += len(steps)
                in_sum = 0.0
                out_sum = 0.0
                for step in steps.values():
                    if not isinstance(step, dict):
                        continue
                    p = step.get("prompt_tokens")
                    c = step.get("completion_tokens")
                    w = step.get("wait_ms")
                    if isinstance(p, (int, float)):
                        in_sum += float(p)
                    if isinstance(c, (int, float)):
                        out_sum += float(c)
                    if isinstance(w, (int, float)):
                        dispatch_waits.append(float(w))
                task_input_tokens.append(in_sum)
                task_output_tokens.append(out_sum)

    return {
        "e2e_makespan_mean_ms": mean_or_nan(makespans),
        "e2e_makespan_p95_ms": percentile(makespans, 95),
        "e2e_makespan_p99_ms": percentile(makespans, 99),
        "num_tasks": float(len(makespans)),
        "total_requests": float(total_requests),
        "input_token_mean": mean_or_nan(task_input_tokens),
        "output_token_mean": mean_or_nan(task_output_tokens),
        "dispatch_wait_mean_ms": mean_or_nan(dispatch_waits),
    }


def find_sweep_name(data_root: Path, metrics_path: Path) -> str | None:
    rel = metrics_path.relative_to(data_root)
    for part in rel.parts:
        if part.startswith("sweep_"):
            return part
    return None


def build_rows(
    data_root: Path,
    model_name: str,
    default_topology: str,
    source_policy: str,
    exclude_vllm_dp: bool,
) -> tuple[pd.DataFrame, list[DropRow], int]:
    rows: list[dict[str, Any]] = []
    dropped: list[DropRow] = []
    candidates = 0
    metric_run_dirs: set[str] = set()

    for metrics_path in sorted(data_root.rglob("server_metrics_*.json")):
        candidates += 1
        run_dir = metrics_path.parent
        run_rel = run_dir.relative_to(data_root).as_posix()
        metric_run_dirs.add(resolve_path_safe(str(run_dir)))

        if source_policy == "raw_sweep_only" and "consolidated_mab_gpusim" in run_rel:
            dropped.append(
                DropRow(
                    source_path=str(metrics_path),
                    run_id=run_rel,
                    reason="policy_excluded_consolidated",
                    detail="Excluded by source policy raw_sweep_only",
                )
            )
            continue

        sweep_name = find_sweep_name(data_root, metrics_path)
        if not sweep_name:
            dropped.append(
                DropRow(
                    source_path=str(metrics_path),
                    run_id=run_rel,
                    reason="missing_sweep_context",
                    detail="No parent directory named sweep_*",
                )
            )
            continue

        meta = parse_sweep_metadata(sweep_name)
        if meta is None:
            dropped.append(
                DropRow(
                    source_path=str(metrics_path),
                    run_id=run_rel,
                    reason="bad_sweep_name",
                    detail=f"Unable to parse sweep metadata from {sweep_name}",
                )
            )
            continue

        dp_size = extract_dp_size(meta["parallelism_config"])
        if exclude_vllm_dp and meta["engine"] == "vllm" and dp_size > 1:
            dropped.append(
                DropRow(
                    source_path=str(metrics_path),
                    run_id=run_rel,
                    reason="runtime_unstable_vllm_dp_moe",
                    detail=f"Excluded by policy: engine=vllm with DP={dp_size} (>1)",
                )
            )
            continue

        framework, concurrency = parse_framework_and_concurrency(metrics_path.name)
        if framework is None or concurrency is None:
            dropped.append(
                DropRow(
                    source_path=str(metrics_path),
                    run_id=run_rel,
                    reason="bad_server_metrics_name",
                    detail=f"Unexpected file name {metrics_path.name}",
                )
            )
            continue

        trace_files = sorted(run_dir.glob("trace_tasks_*.jsonl"))
        if not trace_files:
            dropped.append(
                DropRow(
                    source_path=str(metrics_path),
                    run_id=run_rel,
                    reason="missing_trace_jsonl",
                    detail="No trace_tasks_*.jsonl in same run directory",
                )
            )
            continue

        try:
            server = load_json(metrics_path)
        except Exception as exc:
            dropped.append(
                DropRow(
                    source_path=str(metrics_path),
                    run_id=run_rel,
                    reason="invalid_server_metrics_json",
                    detail=str(exc),
                )
            )
            continue

        try:
            trace_stats = load_trace_stats(trace_files[0])
        except Exception as exc:
            dropped.append(
                DropRow(
                    source_path=str(trace_files[0]),
                    run_id=run_rel,
                    reason="invalid_trace_jsonl",
                    detail=str(exc),
                )
            )
            continue

        topology, topology_source = infer_topology(str(metrics_path), default_topology)
        rows.append(
            {
                "engine": meta["engine"],
                "model": model_name,
                "parallelism_config": meta["parallelism_config"],
                "dp_size": dp_size,
                "precision": meta["precision"],
                "framework": framework,
                "concurrency": int(concurrency),
                "run_id": run_rel,
                "topology": topology,
                "topology_source": topology_source,
                "source_server_metrics_path": str(metrics_path),
                "resolved_server_metrics_path": resolve_path_safe(str(metrics_path)),
                "source_trace_path": str(trace_files[0]),
                "server_tpot_mean_ms": server.get("tpot_mean_ms"),
                "server_tpot_p95_ms": server.get("tpot_p95_ms"),
                "server_tpot_p99_ms": server.get("tpot_p99_ms"),
                "server_ttft_mean_ms": server.get("ttft_mean_ms"),
                "server_ttft_p95_ms": server.get("ttft_p95_ms"),
                "server_ttft_p99_ms": server.get("ttft_p99_ms"),
                "server_tps_mean": server.get("gen_tps"),
                "server_duration_s": server.get("duration_s"),
                "server_e2e_mean_ms": server.get("e2e_mean_ms"),
                "ttft_count": server.get("ttft_count"),
                "tpot_count": server.get("tpot_count"),
                "e2e_count": server.get("e2e_count"),
                **trace_stats,
            }
        )

    # Coverage visibility: trace-only runs are dropped explicitly so they appear in Phase A report.
    # This is important when new runs were produced with server metrics disabled.
    trace_seen_dirs: set[str] = set()
    for trace_path in sorted(data_root.rglob("trace_tasks_*.jsonl")):
        run_dir = trace_path.parent
        run_dir_resolved = resolve_path_safe(str(run_dir))
        if run_dir_resolved in trace_seen_dirs:
            continue
        trace_seen_dirs.add(run_dir_resolved)

        run_rel = run_dir.relative_to(data_root).as_posix()
        if source_policy == "raw_sweep_only" and "consolidated_mab_gpusim" in run_rel:
            continue

        if run_dir_resolved in metric_run_dirs:
            continue
        if list(run_dir.glob("server_metrics_*.json")):
            continue

        dropped.append(
            DropRow(
                source_path=str(trace_path),
                run_id=run_rel,
                reason="missing_server_metrics_json",
                detail="trace_tasks exists but server_metrics_*.json is missing in same run directory",
            )
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            by=["engine", "parallelism_config", "precision", "framework", "concurrency", "run_id"]
        ).reset_index(drop=True)
    return df, dropped, candidates


def dedupe_by_resolved_path(df: pd.DataFrame) -> tuple[pd.DataFrame, list[DropRow]]:
    if df.empty:
        return df, []

    work = df.copy()
    # Prefer non-consolidated paths if both direct and consolidated links exist.
    work["_is_consolidated"] = work["run_id"].str.contains("consolidated_mab_gpusim", na=False)
    work = work.sort_values(by=["resolved_server_metrics_path", "_is_consolidated", "run_id"]).reset_index(drop=True)

    duplicates = work.duplicated(subset=["resolved_server_metrics_path"], keep="first")
    dup_df = work[duplicates].copy()
    keep_df = work[~duplicates].copy()

    dropped: list[DropRow] = []
    if not dup_df.empty:
        keep_map = keep_df.set_index("resolved_server_metrics_path")["run_id"].to_dict()
        for _, r in dup_df.iterrows():
            resolved = r["resolved_server_metrics_path"]
            kept_run = keep_map.get(resolved, "unknown")
            dropped.append(
                DropRow(
                    source_path=str(r.get("source_server_metrics_path", "")),
                    run_id=str(r.get("run_id", "")),
                    reason="duplicate_resolved_source",
                    detail=f"Same resolved server_metrics as kept run_id={kept_run}",
                )
            )

    keep_df = keep_df.drop(columns=["_is_consolidated"])
    return keep_df.reset_index(drop=True), dropped


def _is_nan(x: Any) -> bool:
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def apply_quality_gate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.assign(gate_pass=False, gate_reason="empty")

    work = df.copy()
    reasons: list[list[str]] = [[] for _ in range(len(work))]

    for i, row in work.iterrows():
        for field in META_FIELDS:
            val = row.get(field)
            if _is_nan(val) or (isinstance(val, str) and not val.strip()):
                reasons[i].append(f"missing_meta:{field}")

    for i, row in work.iterrows():
        for m in PRIMARY_METRICS:
            val = row.get(m)
            if _is_nan(val):
                reasons[i].append(f"missing_metric:{m}")
            elif isinstance(val, (int, float)) and not math.isfinite(float(val)):
                reasons[i].append(f"non_finite_metric:{m}")

    for i, row in work.iterrows():
        ttft_count = row.get("ttft_count")
        tpot_count = row.get("tpot_count")
        if _is_nan(ttft_count) or int(ttft_count) <= 0:
            reasons[i].append("ttft_count<=0")
        if _is_nan(tpot_count) or int(tpot_count) <= 0:
            reasons[i].append("tpot_count<=0")

    for i, row in work.iterrows():
        tpot = row.get("server_tpot_mean_ms")
        ttft = row.get("server_ttft_mean_ms")
        tps = row.get("server_tps_mean")
        if isinstance(tpot, (int, float)) and float(tpot) <= 0:
            reasons[i].append("server_tpot_mean_ms<=0")
        if isinstance(ttft, (int, float)) and float(ttft) <= 0:
            reasons[i].append("server_ttft_mean_ms<=0")
        if isinstance(tps, (int, float)) and float(tps) <= 0:
            reasons[i].append("server_tps_mean<=0")

    group_index_map: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for idx, row in work.iterrows():
        key = tuple(row[k] for k in CONDITION_KEYS)
        group_index_map[key].append(idx)

    for _, idxs in group_index_map.items():
        if len(idxs) < 3:
            continue
        g = work.loc[idxs]
        for m in PRIMARY_METRICS:
            vals = g[m].astype(float).to_numpy()
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            if mad == 0:
                continue
            for local_pos, idx in enumerate(idxs):
                z = 0.6745 * (vals[local_pos] - med) / mad
                if abs(z) > 6.0:
                    reasons[idx].append(f"outlier:{m}:robust_z={z:.2f}")

    work["gate_reason"] = [";".join(r) for r in reasons]
    work["gate_pass"] = [len(r) == 0 for r in reasons]
    return work


def build_rerun_targets(quality_df: pd.DataFrame, repeat_target: int, repeat_target_c1: int) -> pd.DataFrame:
    if quality_df.empty:
        return pd.DataFrame()

    pass_df = quality_df[quality_df["gate_pass"]].copy()
    fail_df = quality_df[~quality_df["gate_pass"]].copy()

    valid_count = pass_df.groupby(CONDITION_KEYS).size().to_dict()

    all_keys = set(valid_count.keys())
    for _, r in fail_df.iterrows():
        all_keys.add(tuple(r[k] for k in CONDITION_KEYS))

    rows: list[dict[str, Any]] = []
    for key in sorted(all_keys):
        key_dict = dict(zip(CONDITION_KEYS, key))
        c = int(key_dict["concurrency"])
        required = repeat_target_c1 if c == 1 else repeat_target
        current_valid = int(valid_count.get(key, 0))
        needed = max(required - current_valid, 0)

        cond_fail = fail_df[
            (fail_df["engine"] == key_dict["engine"])
            & (fail_df["parallelism_config"] == key_dict["parallelism_config"])
            & (fail_df["precision"] == key_dict["precision"])
            & (fail_df["framework"] == key_dict["framework"])
            & (fail_df["concurrency"] == key_dict["concurrency"])
        ]

        fail_reasons = ";".join(sorted(set(cond_fail["gate_reason"].astype(str).tolist()))) if len(cond_fail) else ""

        rerun_type = None
        priority = 99
        if len(cond_fail):
            has_zero = cond_fail["gate_reason"].str.contains("ttft_count<=0|tpot_count<=0|missing_metric:server_tpot_mean_ms|missing_metric:server_ttft_mean_ms", regex=True).any()
            has_outlier = cond_fail["gate_reason"].str.contains("outlier:", regex=False).any()
            if has_zero:
                rerun_type = "zero_count_or_missing_metric"
                priority = 0
            elif has_outlier:
                rerun_type = "outlier_replacement"
                priority = 1

        if rerun_type is None and needed > 0:
            rerun_type = "low_repeat"
            priority = 2

        if rerun_type is None:
            continue

        # Prefer failed row as representative source; fallback to any pass row.
        source_run = ""
        if len(cond_fail):
            source_run = str(cond_fail.iloc[0].get("run_id", ""))
        else:
            cond_pass = pass_df[
                (pass_df["engine"] == key_dict["engine"])
                & (pass_df["parallelism_config"] == key_dict["parallelism_config"])
                & (pass_df["precision"] == key_dict["precision"])
                & (pass_df["framework"] == key_dict["framework"])
                & (pass_df["concurrency"] == key_dict["concurrency"])
            ]
            if len(cond_pass):
                source_run = str(cond_pass.iloc[0].get("run_id", ""))

        additional_needed = needed
        if rerun_type in {"zero_count_or_missing_metric", "outlier_replacement"} and additional_needed <= 0:
            additional_needed = 1

        rows.append(
            {
                "priority": priority,
                "rerun_type": rerun_type,
                **key_dict,
                "required_valid_runs": required,
                "current_valid_runs": current_valid,
                "additional_runs_needed": int(additional_needed),
                "example_source_run_id": source_run,
                "fail_reasons": fail_reasons,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(by=["priority", "engine", "parallelism_config", "precision", "framework", "concurrency"]).reset_index(drop=True)


def write_policy_md(
    out_path: Path,
    source_policy: str,
    dedupe_mode: str,
    repeat_target: int,
    repeat_target_c1: int,
    exclude_vllm_dp: bool,
) -> None:
    lines = [
        "# Phase A Policy",
        "",
        "## Source Policy",
        f"- source_policy: `{source_policy}`",
        "- `consolidated_dedupe` means consolidated links are allowed, then deduplicated by resolved source path.",
        "",
        "## Engine Exclusion Policy",
        f"- exclude_vllm_dp: `{exclude_vllm_dp}`",
        "- If enabled, rows with `engine=vllm` and `DP>1` are excluded.",
        "- Drop reason label: `runtime_unstable_vllm_dp_moe`.",
        "",
        "## Deduplication Rule",
        f"- dedupe_mode: `{dedupe_mode}`",
        "- Rule: count each `source_server_metrics_path.resolve()` at most once.",
        "- Duplicate rows are dropped with reason `duplicate_resolved_source`.",
        "",
        "## Repeat Policy",
        f"- target valid repeats (c>1): `{repeat_target}`",
        f"- target valid repeats (c=1): `{repeat_target_c1}`",
        "",
        "## Quality Gates",
        "- `ttft_count > 0`",
        "- `tpot_count > 0`",
        "- required metadata present",
        "- primary metrics present and finite",
        "- robust outlier screening (within repeated condition groups)",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    candidates: int,
    ingest_dropped: list[DropRow],
    parsed_rows_before_dedupe: int,
    parsed_rows_after_dedupe: int,
    quality_df: pd.DataFrame,
    rerun_df: pd.DataFrame,
    out_path: Path,
) -> None:
    ingest_counter = Counter(d.reason for d in ingest_dropped)
    quality_fail = quality_df[~quality_df["gate_pass"]]
    quality_counter = Counter()
    for reason_str in quality_fail["gate_reason"].tolist():
        for r in str(reason_str).split(";"):
            if r:
                quality_counter[r] += 1

    lines = [
        "# Phase A Summary",
        "",
        f"- Candidate server_metrics files scanned: {candidates}",
        f"- Parsed rows before dedupe: {parsed_rows_before_dedupe}",
        f"- Parsed rows after dedupe: {parsed_rows_after_dedupe}",
        f"- Ingest dropped (including policy+dedupe): {len(ingest_dropped)}",
        f"- Quality pass rows: {int(quality_df['gate_pass'].sum()) if not quality_df.empty else 0}",
        f"- Quality failed rows: {int((~quality_df['gate_pass']).sum()) if not quality_df.empty else 0}",
        f"- Rerun targets: {len(rerun_df)}",
        "",
        "## Ingest Drop Reasons",
    ]
    if ingest_counter:
        for k, v in sorted(ingest_counter.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- none")

    lines += ["", "## Quality Fail Reasons"]
    if quality_counter:
        for k, v in sorted(quality_counter.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- none")

    lines += ["", "## Rerun Priority Counts"]
    if not rerun_df.empty:
        for k, v in rerun_df["rerun_type"].value_counts().to_dict().items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- none")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/jp/paper_resource/multi-agent-bench/results_multiagent"),
        help="Root containing sweep_* results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/analysis/phaseA"),
        help="Directory for canonical + quality artifacts.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen3-next-80b",
        help="Model label written to canonical rows.",
    )
    parser.add_argument(
        "--default-topology",
        type=str,
        default="PCIe/SYS",
        help="Fallback topology when path does not include topology keywords.",
    )
    parser.add_argument(
        "--source-policy",
        choices=["consolidated_dedupe", "raw_sweep_only"],
        default="consolidated_dedupe",
        help="Data source policy. consolidated_dedupe is B-option.",
    )
    parser.add_argument(
        "--dedupe-mode",
        choices=["resolved_path", "none"],
        default="resolved_path",
        help="Deduplication mode.",
    )
    parser.add_argument(
        "--repeat-target",
        type=int,
        default=3,
        help="Required valid repeats for c>1.",
    )
    parser.add_argument(
        "--repeat-target-c1",
        type=int,
        default=1,
        help="Required valid repeats for c=1.",
    )
    parser.add_argument(
        "--exclude-vllm-dp",
        dest="exclude_vllm_dp",
        action="store_true",
        default=True,
        help="Exclude vLLM rows with DP>1 (default: enabled).",
    )
    parser.add_argument(
        "--include-vllm-dp",
        dest="exclude_vllm_dp",
        action="store_false",
        help="Disable exclusion of vLLM rows with DP>1.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, ingest_dropped, candidates = build_rows(
        data_root=args.data_root,
        model_name=args.model_name,
        default_topology=args.default_topology,
        source_policy=args.source_policy,
        exclude_vllm_dp=args.exclude_vllm_dp,
    )
    parsed_before = len(df)

    if args.dedupe_mode == "resolved_path":
        df, dedupe_dropped = dedupe_by_resolved_path(df)
        ingest_dropped.extend(dedupe_dropped)
    parsed_after = len(df)

    quality = apply_quality_gate(df)
    canonical = quality[quality["gate_pass"]].copy()
    dropped_quality = quality[~quality["gate_pass"]].copy()
    rerun_targets = build_rerun_targets(quality, args.repeat_target, args.repeat_target_c1)

    canonical_path = args.output_dir / "canonical_comparison.csv"
    quality_path = args.output_dir / "quality_report.csv"
    dropped_path = args.output_dir / "dropped_rows_with_reason.csv"
    rerun_path = args.output_dir / "rerun_targets.csv"
    summary_path = args.output_dir / "phaseA_summary.md"
    policy_path = args.output_dir / "phaseA_policy.md"

    canonical.to_csv(canonical_path, index=False)
    quality.to_csv(quality_path, index=False)
    rerun_targets.to_csv(rerun_path, index=False)

    ingest_drop_df = pd.DataFrame(
        [
            {
                "source_path": d.source_path,
                "run_id": d.run_id,
                "reason": d.reason,
                "detail": d.detail,
                "drop_stage": "ingest",
            }
            for d in ingest_dropped
        ]
    )
    if dropped_quality.empty:
        quality_drop_df = pd.DataFrame(columns=["source_path", "run_id", "reason", "detail", "drop_stage"])
    else:
        quality_drop_df = pd.DataFrame(
            {
                "source_path": dropped_quality.get("source_server_metrics_path"),
                "run_id": dropped_quality.get("run_id"),
                "reason": dropped_quality.get("gate_reason"),
                "detail": dropped_quality.get("gate_reason"),
                "drop_stage": "quality",
            }
        )
    dropped_all = pd.concat([ingest_drop_df, quality_drop_df], ignore_index=True)
    dropped_all.to_csv(dropped_path, index=False)

    write_policy_md(
        policy_path,
        source_policy=args.source_policy,
        dedupe_mode=args.dedupe_mode,
        repeat_target=args.repeat_target,
        repeat_target_c1=args.repeat_target_c1,
        exclude_vllm_dp=args.exclude_vllm_dp,
    )
    write_summary(
        candidates,
        ingest_dropped,
        parsed_before,
        parsed_after,
        quality,
        rerun_targets,
        summary_path,
    )

    print(f"[phaseA] canonical: {canonical_path} rows={len(canonical)}")
    print(f"[phaseA] quality:   {quality_path} rows={len(quality)}")
    print(f"[phaseA] dropped:   {dropped_path} rows={len(dropped_all)}")
    print(f"[phaseA] rerun:     {rerun_path} rows={len(rerun_targets)}")
    print(f"[phaseA] policy:    {policy_path}")
    print(f"[phaseA] summary:   {summary_path}")


if __name__ == "__main__":
    main()
