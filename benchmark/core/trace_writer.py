"""JSONL trace output writers compatible with analyze_agentic_traces.py."""

import json
import os
from dataclasses import asdict
from typing import List

from .types import StepRecord, TaskRecord


def _step_to_dict(s: StepRecord) -> dict:
    """Serialize a StepRecord to dict, including v2 fields when present."""
    d = {
        "agent_role": s.agent_role,
        "deps": s.deps,
        "ready_ts": s.ready_ts,
        "start_ts": s.start_ts,
        "end_ts": s.end_ts,
        "wait_ms": s.wait_ms,
        "latency_ms": s.latency_ms,
        "ttft_ms": s.ttft_ms,
        "tpot_ms": s.tpot_ms,
        "prompt_tokens": s.prompt_tokens,
        "completion_tokens": s.completion_tokens,
        "total_tokens": s.total_tokens,
        "bytes_in": s.bytes_in,
        "bytes_out": s.bytes_out,
        "ok": s.ok,
        "error": s.error,
    }
    # v2 fields — only serialize when set
    if s.start_ns is not None:
        d["start_ns"] = s.start_ns
    if s.first_token_ns is not None:
        d["first_token_ns"] = s.first_token_ns
    if s.end_ns is not None:
        d["end_ns"] = s.end_ns
    if s.status != "ok":
        d["status"] = s.status
    return d


def write_trace_jsonl(records: List[TaskRecord], output_dir: str, suffix: str = "") -> str:
    """Write task records to JSONL trace file. Returns the output path."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"trace_tasks_{len(records)}_exec2{suffix}.jsonl"
    out_path = os.path.join(output_dir, filename)

    with open(out_path, "w", encoding="utf-8") as f:
        for tr in records:
            row = {
                "task_id": tr.task_id,
                "task_start_ts": tr.task_start_ts,
                "task_end_ts": tr.task_end_ts,
                "makespan_ms": tr.makespan_ms,
                "critical_path_ms": tr.critical_path_ms,
                "messages_count": tr.messages_count,
                "tokens_exchanged": tr.tokens_exchanged,
                "bytes_exchanged": tr.bytes_exchanged,
                "total_idle_wait_ms": tr.total_idle_wait_ms,
                "framework": tr.framework,
                "steps": {
                    sid: _step_to_dict(s)
                    for sid, s in tr.steps.items()
                },
            }
            # AutoGen-specific optional fields
            if tr.selector_overhead_ms > 0:
                row["selector_overhead_ms"] = tr.selector_overhead_ms
            if tr.turn_order:
                row["turn_order"] = tr.turn_order
            # v2 fields
            if tr.schema_version >= 2:
                row["schema_version"] = tr.schema_version
            if tr.dag_metrics is not None:
                row["dag_metrics"] = asdict(tr.dag_metrics)
            if tr.role_token_stats:
                row["role_token_stats"] = [asdict(rs) for rs in tr.role_token_stats]
            f.write(json.dumps(row) + "\n")

    return out_path


def write_prompts_jsonl(records: List[TaskRecord], output_dir: str, suffix: str = "") -> str:
    """Write per-task prompt metadata (task_id, step counts) for debugging."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"prompts_{len(records)}{suffix}.jsonl"
    out_path = os.path.join(output_dir, filename)

    with open(out_path, "w", encoding="utf-8") as f:
        for tr in records:
            row = {
                "task_id": tr.task_id,
                "framework": tr.framework,
                "makespan_ms": tr.makespan_ms,
                "step_count": len(tr.steps),
                "steps": list(tr.steps.keys()),
            }
            if tr.turn_order:
                row["turn_order"] = tr.turn_order
            f.write(json.dumps(row) + "\n")

    return out_path
