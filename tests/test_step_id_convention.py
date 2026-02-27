"""Tests for step instance ID convention (SCHEMA_CONTRACT.md).

Validates that all three adapters produce step IDs using the # separator
for repeated invocations.
"""

from benchmark.autogen_ext.runner import AutoGenRunner
from benchmark.langgraph_ext.callback_adapter import LangGraphTraceAdapter
from benchmark.a2a_ext.otel_adapter import OTelTraceAdapter


class TestAutoGenStepIdConvention:
    """AutoGen runner step ID generation uses # separator."""

    def test_first_invocation_no_suffix(self):
        counts = {}
        assert AutoGenRunner._speaker_to_step_id("Planner", counts) == "P"
        counts2 = {}
        assert AutoGenRunner._speaker_to_step_id("Aggregator", counts2) == "A"

    def test_repeated_planner_uses_hash(self):
        counts = {}
        AutoGenRunner._speaker_to_step_id("Planner", counts)  # P
        sid = AutoGenRunner._speaker_to_step_id("Planner", counts)
        assert sid == "P#1"

    def test_repeated_aggregator_uses_hash(self):
        counts = {}
        AutoGenRunner._speaker_to_step_id("Aggregator", counts)  # A
        sid = AutoGenRunner._speaker_to_step_id("Aggregator", counts)
        assert sid == "A#1"

    def test_no_underscore_in_repeated_ids(self):
        """Repeated step IDs must use # not _ as separator."""
        counts = {}
        for _ in range(5):
            sid = AutoGenRunner._speaker_to_step_id("Planner", counts)
        # After 5 calls: P, P#1, P#2, P#3, P#4
        assert "_" not in sid
        assert "#" in sid

    def test_executor_index_preserved(self):
        counts = {}
        sid = AutoGenRunner._speaker_to_step_id("Executor0", counts)
        assert sid == "E0"

    def test_step_ids_unique_within_task(self):
        """All generated step IDs must be unique."""
        counts = {}
        speakers = ["Planner", "Executor0", "Executor1", "Aggregator",
                     "Planner", "Executor0", "Executor1", "Aggregator"]
        sids = [AutoGenRunner._speaker_to_step_id(s, counts) for s in speakers]
        assert len(sids) == len(set(sids)), f"Duplicate IDs: {sids}"


class TestLangGraphStepIdConvention:
    """LangGraph callback adapter step ID generation uses # separator."""

    def _build_adapter_with_nodes(self, node_names):
        """Build adapter and simulate node executions to generate step IDs."""
        role_map = {}
        for n in set(node_names):
            if "planner" in n:
                role_map[n] = "planner"
            elif "executor" in n:
                role_map[n] = "executor"
            elif "aggregator" in n:
                role_map[n] = "aggregator"
        adapter = LangGraphTraceAdapter(role_map=role_map)
        # Manually populate _nodes to test step ID generation
        from benchmark.langgraph_ext.callback_adapter import _NodeState
        for name in node_names:
            role = role_map.get(name, "unknown")
            state = _NodeState(name, role, [])
            state.start_ts = 1.0
            state.end_ts = 2.0
            state.start_ns = 1000
            state.end_ns = 2000
            state.prompt_tokens = 10
            state.completion_tokens = 5
            adapter._nodes.append(state)
        adapter._task_start_ts = 1.0
        adapter._task_end_ts = 2.0
        adapter._task_start_ns = 1000
        adapter._task_end_ns = 2000
        return adapter

    def test_first_invocation_no_suffix(self):
        adapter = self._build_adapter_with_nodes(["planner"])
        tr = adapter.to_task_record()
        assert "P" in tr.steps

    def test_repeated_planner_uses_hash(self):
        adapter = self._build_adapter_with_nodes(["planner", "planner"])
        tr = adapter.to_task_record()
        assert "P" in tr.steps
        assert "P#1" in tr.steps

    def test_no_underscore_in_repeated_ids(self):
        adapter = self._build_adapter_with_nodes(
            ["planner", "planner", "planner"]
        )
        tr = adapter.to_task_record()
        for sid in tr.steps:
            if sid != "P":
                assert "_" not in sid, f"Underscore in repeated ID: {sid}"
                assert "#" in sid

    def test_step_ids_unique(self):
        adapter = self._build_adapter_with_nodes(
            ["planner", "executor_0", "executor_1", "aggregator",
             "planner", "executor_0"]
        )
        tr = adapter.to_task_record()
        sids = list(tr.steps.keys())
        assert len(sids) == len(set(sids)), f"Duplicate IDs: {sids}"


class TestOTelStepIdConvention:
    """A2A OTel adapter step ID generation uses # separator."""

    def _build_adapter_with_spans(self, agent_names):
        """Build adapter and add spans to test step ID generation."""
        adapter = OTelTraceAdapter()
        from benchmark.a2a_ext.otel_adapter import _SpanRecord
        for name in agent_names:
            rec = _SpanRecord(name)
            lower = name.lower()
            if "planner" in lower:
                rec.role = "planner"
            elif "executor" in lower:
                rec.role = "executor"
            elif "aggregator" in lower:
                rec.role = "aggregator"
            rec.start_ns = 1_000_000_000
            rec.end_ns = 2_000_000_000
            rec.prompt_tokens = 10
            rec.completion_tokens = 5
            adapter._spans.append(rec)
        return adapter

    def test_first_invocation_no_suffix(self):
        adapter = self._build_adapter_with_spans(["planner_agent"])
        tr = adapter.to_task_record()
        assert "P" in tr.steps

    def test_repeated_planner_uses_hash(self):
        adapter = self._build_adapter_with_spans(
            ["planner_agent", "planner_agent"]
        )
        tr = adapter.to_task_record()
        assert "P" in tr.steps
        assert "P#1" in tr.steps

    def test_no_underscore_in_repeated_ids(self):
        adapter = self._build_adapter_with_spans(
            ["planner_agent", "planner_agent", "planner_agent"]
        )
        tr = adapter.to_task_record()
        for sid in tr.steps:
            if sid != "P":
                assert "_" not in sid, f"Underscore in repeated ID: {sid}"
                assert "#" in sid

    def test_step_ids_unique(self):
        adapter = self._build_adapter_with_spans(
            ["planner_agent", "executor_0", "executor_1",
             "aggregator_agent", "planner_agent", "executor_0"]
        )
        tr = adapter.to_task_record()
        sids = list(tr.steps.keys())
        assert len(sids) == len(set(sids)), f"Duplicate IDs: {sids}"
