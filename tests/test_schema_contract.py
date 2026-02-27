"""Tests for v2 trace schema contract (SCHEMA_CONTRACT.md).

Validates that core types satisfy the required/optional field expectations
defined in the schema contract.
"""

from benchmark.core.types import DagMetrics, StepRecord, TaskRecord


class TestStepRecordContract:
    """StepRecord must expose all REQUIRED fields with correct defaults."""

    def test_required_fields_exist(self):
        sr = StepRecord(step_id="E0", agent_role="executor", deps=["P"])
        assert sr.step_id == "E0"
        assert sr.agent_role == "executor"
        assert sr.deps == ["P"]
        assert sr.prompt_tokens == 0
        assert sr.completion_tokens == 0
        assert sr.latency_ms == 0.0

    def test_status_defaults_to_ok(self):
        sr = StepRecord(step_id="P", agent_role="planner", deps=[])
        assert sr.status == "ok"

    def test_ns_fields_default_none(self):
        sr = StepRecord(step_id="P", agent_role="planner", deps=[])
        assert sr.start_ns is None
        assert sr.first_token_ns is None
        assert sr.end_ns is None

    def test_ns_fields_round_trip(self):
        sr = StepRecord(
            step_id="P", agent_role="planner", deps=[],
            start_ns=1000, first_token_ns=2000, end_ns=3000,
        )
        assert sr.start_ns == 1000
        assert sr.first_token_ns == 2000
        assert sr.end_ns == 3000


class TestTaskRecordContract:
    """TaskRecord must satisfy task-level schema contract."""

    def test_schema_version_defaults_to_2(self):
        tr = TaskRecord(
            task_id=0, task_start_ts=0.0, task_end_ts=1.0,
            makespan_ms=1000.0, steps={}, messages_count=0,
            tokens_exchanged=0, bytes_exchanged=0,
            critical_path_ms=0.0, total_idle_wait_ms=0.0,
        )
        assert tr.schema_version == 2

    def test_dag_metrics_optional(self):
        tr = TaskRecord(
            task_id=0, task_start_ts=0.0, task_end_ts=1.0,
            makespan_ms=1000.0, steps={}, messages_count=0,
            tokens_exchanged=0, bytes_exchanged=0,
            critical_path_ms=0.0, total_idle_wait_ms=0.0,
        )
        assert tr.dag_metrics is None

    def test_dag_metrics_present(self):
        dm = DagMetrics(depth=3, max_width=2)
        tr = TaskRecord(
            task_id=0, task_start_ts=0.0, task_end_ts=1.0,
            makespan_ms=1000.0, steps={}, messages_count=0,
            tokens_exchanged=0, bytes_exchanged=0,
            critical_path_ms=0.0, total_idle_wait_ms=0.0,
            dag_metrics=dm,
        )
        assert tr.dag_metrics.depth == 3
        assert tr.dag_metrics.max_width == 2
