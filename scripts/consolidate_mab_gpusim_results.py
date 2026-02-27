#!/usr/bin/env python3
"""Consolidate MAB + GPUSim result files into one indexed location (via symlinks)."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


MAB_ROOT = Path("/home/jp/paper_resource/multi-agent-bench")
GPUSIM_ROOT = Path("/home/jp/paper_resource/gpusim")

# Result roots to consolidate.
SOURCE_ROOTS: list[tuple[str, Path]] = [
    ("mab", MAB_ROOT / "results_multiagent"),
    ("mab", MAB_ROOT / "paper"),
    ("gpusim", GPUSIM_ROOT / "results_from_real_H100"),
    ("gpusim", GPUSIM_ROOT / "results"),
    ("gpusim", GPUSIM_ROOT / "results_agentic_sim"),
    ("gpusim", GPUSIM_ROOT / "configs" / "simulation"),
]
RUN_HISTORY_CSV = MAB_ROOT / "results_multiagent" / "consolidation_runs_history.csv"


@dataclass
class Record:
    repo: str
    category: str
    source_root: str
    source_path: str
    rel_path: str
    size_bytes: int
    mtime_utc: str
    link_path: str


def split_kind(r: Record) -> str:
    """Classify each record into real/sim/docs_misc buckets."""
    root_name = Path(r.source_root).name
    if root_name in {"results_from_real_H100", "results_multiagent"}:
        return "real"
    if root_name in {"results", "results_agentic_sim", "simulation"}:
        return "sim"
    return "docs_misc"


def is_calibration_record(r: Record) -> bool:
    needle = f"{r.category} {r.source_path} {r.rel_path}".lower()
    return "calib" in needle or "calibration" in needle or "calibrated" in needle


def utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def categorize(repo: str, src_root: Path, rel: Path) -> str:
    p = str(rel).replace("\\", "/")
    name = rel.name
    root_name = src_root.name

    if repo == "mab":
        if root_name == "paper":
            return "mab_report_docs"
        if "/figures/" in f"/{p}/" or p.startswith("figures/"):
            return "mab_figures"
        if name.startswith("server_metrics_"):
            return "mab_server_metrics"
        if name.startswith("trace_") or name.endswith(".jsonl"):
            return "mab_trace_jsonl"
        if "summary" in name:
            return "mab_summaries"
        return "mab_other"

    # gpusim
    if root_name == "simulation":
        if "calib" in name.lower() or "calibrated" in name.lower():
            return "gpusim_calibration_configs"
        return "gpusim_simulation_configs"

    if root_name == "results_from_real_H100":
        if "nccl" in p.lower():
            return "gpusim_real_nccl"
        if "/batch_sweep" in f"/{p}/":
            return "gpusim_real_batch_sweep"
        if "/agentic_sweep/" in f"/{p}/":
            if name.startswith("server_metrics_"):
                return "gpusim_real_agentic_server_metrics"
            if "trace_tasks_" in name:
                return "gpusim_real_agentic_traces"
            return "gpusim_real_agentic_other"
        if "summary" in name:
            return "gpusim_real_summaries"
        return "gpusim_real_other"

    if root_name == "results_agentic_sim":
        if "summary" in name:
            return "gpusim_agentic_sim_summaries"
        return "gpusim_agentic_sim_other"

    # root_name == "results"
    if "/results_multiagent/" in f"/{p}/":
        if "/figures/" in f"/{p}/":
            return "gpusim_results_multiagent_figures"
        if name.startswith("server_metrics_"):
            return "gpusim_results_multiagent_server_metrics"
        if name.endswith(".jsonl"):
            return "gpusim_results_multiagent_traces"
        return "gpusim_results_multiagent_other"

    if "/figures/" in f"/{p}/":
        return "gpusim_results_figures"
    if "summary" in name:
        return "gpusim_results_summaries"
    if name.endswith(".jsonl"):
        return "gpusim_results_jsonl"
    if name.endswith(".json"):
        return "gpusim_results_json"
    return "gpusim_results_other"


def iter_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def append_history_row(path: Path, row: dict[str, object]) -> None:
    """Append one history row, evolving CSV schema if new columns were added."""
    new_fields = list(row.keys())
    if not path.exists():
        with path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=new_fields)
            writer.writeheader()
            writer.writerow(row)
        return

    with path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        old_fields = reader.fieldnames or []
        old_rows = list(reader)

    if old_fields == new_fields:
        with path.open("a", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=new_fields)
            writer.writerow(row)
        return

    # Expand schema while preserving previous rows.
    merged_fields = old_fields + [f for f in new_fields if f not in old_fields]
    normalized_rows: list[dict[str, object]] = []
    for r in old_rows:
        normalized_rows.append({k: r.get(k, "") for k in merged_fields})
    normalized_rows.append({k: row.get(k, "") for k in merged_fields})

    # Backfill calibration columns for historical rows when possible.
    if "calibration_files" in merged_fields and "output_root" in merged_fields:
        for r in normalized_rows:
            if str(r.get("calibration_files", "")).strip():
                continue
            out = str(r.get("output_root", "")).strip()
            if not out:
                continue
            sc = Path(out) / "catalog" / "summary_calibration.csv"
            if not sc.exists():
                continue
            try:
                with sc.open("r", newline="") as fp:
                    reader = csv.DictReader(fp)
                    first = next(reader, None)
                if first:
                    r["calibration_files"] = first.get("calibration_file_count", "")
                    r["calibration_size_bytes"] = first.get("calibration_total_size_bytes", "")
            except Exception:
                continue

    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=merged_fields)
        writer.writeheader()
        for r in normalized_rows:
            writer.writerow(r)


def main() -> None:
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d_%H%M%S")
    out_root = MAB_ROOT / "results_multiagent" / f"consolidated_mab_gpusim_{stamp}"
    links_root = out_root / "links"
    catalog_root = out_root / "catalog"
    links_root.mkdir(parents=True, exist_ok=True)
    catalog_root.mkdir(parents=True, exist_ok=True)

    records: list[Record] = []
    missing_roots: list[str] = []

    for repo, src_root in SOURCE_ROOTS:
        if not src_root.exists():
            missing_roots.append(str(src_root))
            continue

        for f in iter_files(src_root):
            # Avoid self-recursion when output folder is inside a source root.
            if out_root in f.parents:
                continue
            # Avoid re-ingesting previously consolidated snapshots.
            if any(p.name.startswith("consolidated_mab_gpusim_") for p in f.parents):
                continue
            if src_root.is_file():
                rel = Path(src_root.name)
            else:
                rel = f.relative_to(src_root)
            cat = categorize(repo, src_root, rel)

            link_rel = Path(repo) / src_root.name / rel
            link_path = links_root / link_rel
            link_path.parent.mkdir(parents=True, exist_ok=True)
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            os.symlink(f, link_path)

            st = f.stat()
            records.append(
                Record(
                    repo=repo,
                    category=cat,
                    source_root=str(src_root),
                    source_path=str(f),
                    rel_path=str(rel),
                    size_bytes=st.st_size,
                    mtime_utc=utc_iso(st.st_mtime),
                    link_path=str(link_path),
                )
            )

    # Full catalog CSV
    catalog_csv = catalog_root / "results_catalog.csv"
    with catalog_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(asdict(records[0]).keys()) if records else [])
        if records:
            writer.writeheader()
            for r in records:
                writer.writerow(asdict(r))

    # Split catalogs: real / sim / docs_misc
    split_files = {
        "real": catalog_root / "results_catalog_real.csv",
        "sim": catalog_root / "results_catalog_sim.csv",
        "docs_misc": catalog_root / "results_catalog_docs_misc.csv",
    }
    split_rows = {"real": [], "sim": [], "docs_misc": []}
    for r in records:
        split_rows[split_kind(r)].append(r)
    fieldnames = list(asdict(records[0]).keys()) if records else []
    for k, path in split_files.items():
        with path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
            for r in split_rows[k]:
                writer.writerow(asdict(r))

    # Calibration-only catalog
    calibration_csv = catalog_root / "results_catalog_calibration.csv"
    calibration_records = [r for r in records if is_calibration_record(r)]
    with calibration_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
        for r in calibration_records:
            writer.writerow(asdict(r))

    # Summary by category
    by_cat = Counter(r.category for r in records)
    bytes_by_cat = Counter()
    for r in records:
        bytes_by_cat[r.category] += r.size_bytes

    summary_cat_csv = catalog_root / "summary_by_category.csv"
    with summary_cat_csv.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["category", "file_count", "total_size_bytes"])
        for cat in sorted(by_cat):
            writer.writerow([cat, by_cat[cat], bytes_by_cat[cat]])

    # Summary by repo/root
    by_repo_root = Counter((r.repo, Path(r.source_root).name) for r in records)
    bytes_by_repo_root = Counter()
    for r in records:
        key = (r.repo, Path(r.source_root).name)
        bytes_by_repo_root[key] += r.size_bytes

    summary_repo_csv = catalog_root / "summary_by_repo_root.csv"
    with summary_repo_csv.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["repo", "source_root_name", "file_count", "total_size_bytes"])
        for key in sorted(by_repo_root):
            writer.writerow([key[0], key[1], by_repo_root[key], bytes_by_repo_root[key]])

    # Summary by split kind
    by_split = Counter(split_kind(r) for r in records)
    bytes_by_split = Counter()
    for r in records:
        bytes_by_split[split_kind(r)] += r.size_bytes
    summary_split_csv = catalog_root / "summary_by_split_kind.csv"
    with summary_split_csv.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["split_kind", "file_count", "total_size_bytes"])
        for k in ["real", "sim", "docs_misc"]:
            writer.writerow([k, by_split[k], bytes_by_split[k]])

    # Summary for calibration-only files
    calibration_count = len(calibration_records)
    calibration_size = sum(r.size_bytes for r in calibration_records)
    summary_calibration_csv = catalog_root / "summary_calibration.csv"
    with summary_calibration_csv.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["calibration_file_count", "calibration_total_size_bytes"])
        writer.writerow([calibration_count, calibration_size])

    # JSON metadata
    meta = {
        "generated_at_utc": now.isoformat(),
        "output_root": str(out_root),
        "link_mode": "symlink",
        "source_roots": [str(p) for _, p in SOURCE_ROOTS],
        "missing_source_roots": missing_roots,
        "total_files": len(records),
        "total_size_bytes": sum(r.size_bytes for r in records),
        "total_files_real": by_split["real"],
        "total_files_sim": by_split["sim"],
        "total_files_docs_misc": by_split["docs_misc"],
        "total_files_calibration": calibration_count,
        "counts_by_category": dict(sorted(by_cat.items())),
    }
    with (catalog_root / "meta.json").open("w") as fp:
        json.dump(meta, fp, indent=2)

    # Human-readable summary
    readme = out_root / "README.md"
    with readme.open("w") as fp:
        fp.write("# Consolidated Results (MAB + GPUSim)\n\n")
        fp.write(f"- Generated at (UTC): `{now.isoformat()}`\n")
        fp.write(f"- Output root: `{out_root}`\n")
        fp.write(f"- Total files: `{len(records)}`\n")
        fp.write(f"- Total size (bytes): `{sum(r.size_bytes for r in records)}`\n")
        fp.write("- Link mode: symlink (original files remain in place)\n\n")
        fp.write("## Source Roots\n")
        for _, p in SOURCE_ROOTS:
            fp.write(f"- `{p}`\n")
        if missing_roots:
            fp.write("\n## Missing Source Roots\n")
            for p in missing_roots:
                fp.write(f"- `{p}`\n")
        fp.write("\n## Catalog Files\n")
        fp.write(f"- `{catalog_csv}`\n")
        fp.write(f"- `{summary_cat_csv}`\n")
        fp.write(f"- `{summary_repo_csv}`\n")
        fp.write(f"- `{summary_split_csv}`\n")
        fp.write(f"- `{summary_calibration_csv}`\n")
        fp.write(f"- `{catalog_root / 'meta.json'}`\n")
        fp.write(f"- `{split_files['real']}`\n")
        fp.write(f"- `{split_files['sim']}`\n")
        fp.write(f"- `{split_files['docs_misc']}`\n")
        fp.write(f"- `{calibration_csv}`\n")
        fp.write("\n## Top Categories by File Count\n")
        for cat, n in sorted(by_cat.items(), key=lambda kv: kv[1], reverse=True)[:20]:
            fp.write(f"- `{cat}`: {n} files\n")
        fp.write("\n## Split Counts\n")
        fp.write(f"- `real`: {by_split['real']} files\n")
        fp.write(f"- `sim`: {by_split['sim']} files\n")
        fp.write(f"- `docs_misc`: {by_split['docs_misc']} files\n")
        fp.write(f"- `calibration`: {calibration_count} files\n")

    # Append cumulative run history.
    history_row = {
        "run_stamp": stamp,
        "generated_at_utc": now.isoformat(),
        "output_root": str(out_root),
        "total_files": len(records),
        "total_size_bytes": sum(r.size_bytes for r in records),
        "real_files": by_split["real"],
        "real_size_bytes": bytes_by_split["real"],
        "sim_files": by_split["sim"],
        "sim_size_bytes": bytes_by_split["sim"],
        "docs_misc_files": by_split["docs_misc"],
        "docs_misc_size_bytes": bytes_by_split["docs_misc"],
        "calibration_files": calibration_count,
        "calibration_size_bytes": calibration_size,
    }
    append_history_row(RUN_HISTORY_CSV, history_row)

    print(f"Created consolidated folder: {out_root}")
    print(f"Total files indexed: {len(records)}")
    print(f"Catalog: {catalog_csv}")
    print(f"Summary: {summary_cat_csv}")
    print(f"Summary: {summary_repo_csv}")
    print(f"Summary: {summary_split_csv}")
    print(f"Summary: {summary_calibration_csv}")
    print(f"Catalog (real): {split_files['real']}")
    print(f"Catalog (sim): {split_files['sim']}")
    print(f"Catalog (calibration): {calibration_csv}")
    print(f"Run history: {RUN_HISTORY_CSV}")
    print(f"README: {readme}")


if __name__ == "__main__":
    main()
