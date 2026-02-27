"""Tests for v2 trace schema: serialization, deserialization, backward compat."""

import json
import os
import tempfile

import pytest

from benchmark.core.dag_metrics import compute_dag_metrics, compute_role_token_stats
from benchmark.core.trace_writer import write_trace_jsonl
from benchmark.core.types import (
    DagMetrics,
    RoleTokenStats,
    StepRecord,
    TaskRecord,
)


def _make_v2_task_record() -> TaskRecord:
    """Create a v2 TaskRecord with all fields populated."""
    steps = {
        "P": StepRecord(
            step_id="P",
            agent_role="planner",
            deps=[],
            ready_ts=1000.0,
            start_ts=1000.1,
            end_ts=1001.0,
            wait_ms=100.0,
            latency_ms=900.0,
            ttft_ms=50.0,
            tpot_ms=10.0,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            bytes_in=400,
            bytes_out=200,
            ok=True,
            start_ns=1000000000,
            first_token_ns=1050000000,
            end_ns=1900000000,
            status="ok",
        ),
        "E0": StepRecord(
            step_id="E0",
            agent_role="executor",
            deps=["P"],
            ready_ts=1001.0,
            start_ts=1001.1,
            end_ts=1002.0,
            wait_ms=100.0,
            latency_ms=900.0,
            ttft_ms=40.0,
            tpot_ms=12.0,
            prompt_tokens=200,
            completion_tokens=80,
            total_tokens=280,
            bytes_in=800,
            bytes_out=320,
            ok=True,
            start_ns=2000000000,
            first_token_ns=2040000000,
            end_ns=2900000000,
            status="ok",
        ),
        "A": StepRecord(
            step_id="A",
            agent_role="aggregator",
            deps=["E0"],
            ready_ts=1002.0,
            start_ts=1002.1,
            end_ts=1003.0,
            wait_ms=100.0,
            latency_ms=900.0,
            prompt_tokens=300,
            completion_tokens=60,
            total_tokens=360,
            bytes_in=1200,
            bytes_out=240,
            ok=True,
            start_ns=3000000000,
            end_ns=3900000000,
            status="ok",
        ),
    }

    dag_metrics = DagMetrics(
        depth=3,
        max_width=1,
        fanout_max=1,
        fanin_max=1,
        critical_path_steps=["P", "E0", "A"],
        critical_path_len=3,
        parallel_fraction=0.0,
    )

    role_token_stats = [
        RoleTokenStats(role="aggregator", count=1, prompt_mean=300.0, output_mean=60.0),
        RoleTokenStats(role="executor", count=1, prompt_mean=200.0, output_mean=80.0),
        RoleTokenStats(role="planner", count=1, prompt_mean=100.0, output_mean=50.0),
    ]

    return TaskRecord(
        task_id=0,
        task_start_ts=1000.0,
        task_end_ts=1003.0,
        makespan_ms=3000.0,
        steps=steps,
        messages_count=3,
        tokens_exchanged=790,
        bytes_exchanged=3160,
        critical_path_ms=2700.0,
        total_idle_wait_ms=300.0,
        framework="langgraph",
        schema_version=2,
        dag_metrics=dag_metrics,
        role_token_stats=role_token_stats,
    )


class TestV2Serialization:
    """Test v2 TaskRecord → JSONL → round-trip."""

    def test_write_read_roundtrip(self):
        """Write v2 trace to JSONL and verify all fields."""
        tr = _make_v2_task_record()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_trace_jsonl([tr], tmpdir, suffix="_test")
            assert os.path.exists(path)

            with open(path, "r") as f:
                data = json.loads(f.readline())

        # Schema version
        assert data["schema_version"] == 2

        # DAG metrics
        assert "dag_metrics" in data
        dm = data["dag_metrics"]
        assert dm["depth"] == 3
        assert dm["max_width"] == 1
        assert dm["critical_path_steps"] == ["P", "E0", "A"]
        assert dm["parallel_fraction"] == 0.0

        # Role token stats
        assert "role_token_stats" in data
        rts = data["role_token_stats"]
        assert len(rts) == 3
        roles = {r["role"] for r in rts}
        assert roles == {"planner", "executor", "aggregator"}

        # Step v2 fields
        step_p = data["steps"]["P"]
        assert step_p["start_ns"] == 1000000000
        assert step_p["first_token_ns"] == 1050000000
        assert step_p["end_ns"] == 1900000000
        # status="ok" is not written (default)
        assert "status" not in step_p

        step_e0 = data["steps"]["E0"]
        assert step_e0["start_ns"] == 2000000000

    def test_v1_backward_compat(self):
        """v1-style TaskRecord (no v2 fields) serializes without errors."""
        steps = {
            "P": StepRecord(step_id="P", agent_role="planner", deps=[]),
        }
        tr = TaskRecord(
            task_id=1,
            task_start_ts=100.0,
            task_end_ts=101.0,
            makespan_ms=1000.0,
            steps=steps,
            messages_count=1,
            tokens_exchanged=0,
            bytes_exchanged=0,
            critical_path_ms=0.0,
            total_idle_wait_ms=0.0,
        )
        # schema_version defaults to 2 now, but dag_metrics is None

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_trace_jsonl([tr], tmpdir, suffix="_v1compat")
            with open(path, "r") as f:
                data = json.loads(f.readline())

        assert data["schema_version"] == 2
        # No dag_metrics since it was None
        assert "dag_metrics" not in data
        # Steps should serialize fine without ns fields
        assert "start_ns" not in data["steps"]["P"]

    def test_error_step_status(self):
        """Step with status='error' is serialized."""
        steps = {
            "P": StepRecord(
                step_id="P", agent_role="planner", deps=[],
                ok=False, error="timeout", status="error",
            ),
        }
        tr = TaskRecord(
            task_id=2,
            task_start_ts=0.0,
            task_end_ts=1.0,
            makespan_ms=1000.0,
            steps=steps,
            messages_count=1,
            tokens_exchanged=0,
            bytes_exchanged=0,
            critical_path_ms=0.0,
            total_idle_wait_ms=0.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_trace_jsonl([tr], tmpdir, suffix="_err")
            with open(path, "r") as f:
                data = json.loads(f.readline())

        assert data["steps"]["P"]["status"] == "error"
        assert data["steps"]["P"]["error"] == "timeout"


class TestDagMetricsIntegration:
    """Test DagMetrics dataclass integration with TaskRecord."""

    def test_dag_metrics_asdict(self):
        """DagMetrics can be converted to dict via dataclasses.asdict."""
        from dataclasses import asdict
        dm = DagMetrics(depth=3, max_width=2, fanout_max=2, fanin_max=2,
                        critical_path_steps=["P", "E0", "A"],
                        critical_path_len=3, parallel_fraction=0.25)
        d = asdict(dm)
        assert d["depth"] == 3
        assert d["parallel_fraction"] == 0.25

    def test_role_token_stats_asdict(self):
        """RoleTokenStats can be converted to dict."""
        from dataclasses import asdict
        rts = RoleTokenStats(role="executor", count=2, prompt_mean=250.0,
                             prompt_std=50.0, output_mean=100.0, output_std=20.0)
        d = asdict(rts)
        assert d["role"] == "executor"
        assert d["count"] == 2
