"""
LangGraph Trace Adapter — generic callback handler for any StateGraph.

Attaches to a compiled LangGraph graph via LangChain's BaseCallbackHandler
to capture per-node timing, TTFT, and token usage without modifying the
graph definition.

Usage:
    adapter = LangGraphTraceAdapter(
        role_map={"planner": "planner", "executor_0": "executor", ...},
        dep_map={"executor_0": ["planner"], "aggregator": ["executor_0", "executor_1"]},
    )
    result = await compiled.ainvoke(input, config={"callbacks": [adapter]})
    task_record = adapter.to_task_record(task_id=0)
"""

import time
from typing import Any, Dict, List, Optional, Sequence, Union

from benchmark.core.dag_metrics import compute_dag_metrics, compute_role_token_stats
from benchmark.core.types import DagMetrics, RoleTokenStats, StepRecord, TaskRecord

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    # Provide a stub so the module can be imported without langchain
    _LANGCHAIN_AVAILABLE = False

    class BaseCallbackHandler:  # type: ignore[no-redef]
        pass


class _NodeState:
    """Internal state for a single graph node execution."""

    __slots__ = (
        "node_name", "role", "deps", "start_ts", "start_ns",
        "end_ts", "end_ns", "first_token_ns", "first_token_ts",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "status", "error", "chunk_count",
    )

    def __init__(self, node_name: str, role: str, deps: List[str]):
        self.node_name = node_name
        self.role = role
        self.deps = deps
        self.start_ts: float = 0.0
        self.start_ns: int = 0
        self.end_ts: float = 0.0
        self.end_ns: int = 0
        self.first_token_ns: Optional[int] = None
        self.first_token_ts: Optional[float] = None
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.status: str = "ok"
        self.error: Optional[str] = None
        self.chunk_count: int = 0


class LangGraphTraceAdapter(BaseCallbackHandler):
    """Callback handler that captures trace data from any LangGraph execution.

    Args:
        role_map: Maps graph node names to agent roles
            (e.g., {"planner": "planner", "executor_0": "executor"}).
        dep_map: Maps graph node names to dependency node names
            (e.g., {"executor_0": ["planner"], "aggregator": ["executor_0"]}).
        framework: Framework name for the TaskRecord (default "langgraph").
    """

    def __init__(
        self,
        role_map: Optional[Dict[str, str]] = None,
        dep_map: Optional[Dict[str, List[str]]] = None,
        framework: str = "langgraph",
    ):
        super().__init__()
        self.role_map = role_map or {}
        self.dep_map = dep_map or {}
        self.framework = framework

        self._nodes: List[_NodeState] = []
        self._active_run_to_idx: Dict[str, int] = {}  # run_id -> index in _nodes
        self._task_start_ts: float = 0.0
        self._task_start_ns: int = 0
        self._task_end_ts: float = 0.0
        self._task_end_ns: int = 0

    # -- Chain callbacks (node start/end) --

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        # Extract node name from serialized info
        name = serialized.get("name", "") or serialized.get("id", [""])[-1]
        if not name or name in ("RunnableSequence", "RunnableLambda"):
            return

        # Track task-level start
        if not self._task_start_ts:
            self._task_start_ts = time.time()
            self._task_start_ns = time.monotonic_ns()

        role = self.role_map.get(name, "unknown")
        deps = self.dep_map.get(name, [])

        state = _NodeState(name, role, deps)
        state.start_ts = time.time()
        state.start_ns = time.monotonic_ns()

        self._nodes.append(state)
        self._active_run_to_idx[str(run_id)] = len(self._nodes) - 1

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        idx = self._active_run_to_idx.pop(str(run_id), None)
        if idx is not None:
            state = self._nodes[idx]
            state.end_ts = time.time()
            state.end_ns = time.monotonic_ns()

        self._task_end_ts = time.time()
        self._task_end_ns = time.monotonic_ns()

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        idx = self._active_run_to_idx.pop(str(run_id), None)
        if idx is not None:
            state = self._nodes[idx]
            state.end_ts = time.time()
            state.end_ns = time.monotonic_ns()
            state.status = "error"
            state.error = str(error)

        self._task_end_ts = time.time()
        self._task_end_ns = time.monotonic_ns()

    # -- LLM callbacks (token capture) --

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        # Estimate prompt tokens from byte length
        idx = self._active_run_to_idx.get(str(parent_run_id))
        if idx is not None:
            total_bytes = sum(len(p.encode("utf-8")) for p in prompts)
            self._nodes[idx].prompt_tokens = max(1, total_bytes // 4)

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        # Capture TTFT on first token
        idx = self._active_run_to_idx.get(str(parent_run_id))
        if idx is not None:
            state = self._nodes[idx]
            state.chunk_count += 1
            if state.first_token_ns is None:
                state.first_token_ns = time.monotonic_ns()
                state.first_token_ts = time.time()

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        idx = self._active_run_to_idx.get(str(parent_run_id))
        if idx is None:
            return
        state = self._nodes[idx]

        # 4-stage token extraction fallback
        prompt_t, completion_t, total_t = self._extract_tokens(response)
        if prompt_t is not None:
            state.prompt_tokens = prompt_t
        if completion_t is not None:
            state.completion_tokens = completion_t
        if total_t is not None:
            state.total_tokens = total_t

        # Fallback: use chunk count for completion tokens
        if state.completion_tokens == 0 and state.chunk_count > 0:
            state.completion_tokens = state.chunk_count

    @staticmethod
    def _extract_tokens(response: Any) -> tuple:
        """4-stage fallback token extraction from LLM response.

        Returns (prompt_tokens, completion_tokens, total_tokens) — any may be None.
        """
        prompt_t = completion_t = total_t = None

        # Stage 1: response.llm_output["token_usage"]
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage") or response.llm_output.get("usage")
            if usage:
                prompt_t = usage.get("prompt_tokens")
                completion_t = usage.get("completion_tokens")
                total_t = usage.get("total_tokens")
                if prompt_t is not None:
                    return prompt_t, completion_t, total_t

        # Stage 2: generation.generation_info["usage"]
        if hasattr(response, "generations") and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    info = getattr(gen, "generation_info", None) or {}
                    usage = info.get("usage")
                    if usage:
                        prompt_t = usage.get("prompt_tokens", prompt_t)
                        completion_t = usage.get("completion_tokens", completion_t)
                        total_t = usage.get("total_tokens", total_t)
                        if prompt_t is not None:
                            return prompt_t, completion_t, total_t

        # Stage 3: generation.message.usage_metadata (OpenAI v2)
        if hasattr(response, "generations") and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg:
                        um = getattr(msg, "usage_metadata", None)
                        if um:
                            prompt_t = getattr(um, "input_tokens", prompt_t)
                            completion_t = getattr(um, "output_tokens", completion_t)
                            total_t = getattr(um, "total_tokens", total_t)
                            if prompt_t is not None:
                                return prompt_t, completion_t, total_t

        return prompt_t, completion_t, total_t

    # -- Output --

    def _node_to_step_id(self, name: str) -> str:
        """Map node name to short step ID (P, E0, E1, A, etc.)."""
        role = self.role_map.get(name, "")
        if role == "planner":
            return "P"
        elif role == "executor":
            # Extract index from name like "executor_0"
            parts = name.rsplit("_", 1)
            idx = parts[-1] if len(parts) > 1 and parts[-1].isdigit() else "0"
            return f"E{idx}"
        elif role == "aggregator":
            return "A"
        return name[:3].upper()

    def to_task_record(self, task_id: int = 0) -> TaskRecord:
        """Build a TaskRecord from captured callback data."""
        # Generate unique step IDs: first occurrence gets base ID (e.g. "P"),
        # subsequent occurrences get suffixed IDs (e.g. "P#1", "P#2").
        sid_counts: Dict[str, int] = {}
        node_sids: List[str] = []
        for state in self._nodes:
            base_sid = self._node_to_step_id(state.node_name)
            count = sid_counts.get(base_sid, 0)
            sid = base_sid if count == 0 else f"{base_sid}#{count}"
            sid_counts[base_sid] = count + 1
            node_sids.append(sid)

        steps: Dict[str, StepRecord] = {}
        steps_raw: Dict[str, Dict] = {}  # for dag_metrics

        for i, state in enumerate(self._nodes):
            sid = node_sids[i]
            # Map dep node names to the step ID of their most recent occurrence
            dep_sids = []
            for d in state.deps:
                d_base = self._node_to_step_id(d)
                # Find the latest occurrence of this dep that precedes current node
                d_sid = d_base  # fallback
                for j in range(i - 1, -1, -1):
                    if self._nodes[j].node_name == d:
                        d_sid = node_sids[j]
                        break
                dep_sids.append(d_sid)

            latency_ms = (state.end_ts - state.start_ts) * 1000 if state.end_ts else 0.0
            ttft_ms = None
            if state.first_token_ts and state.start_ts:
                ttft_ms = (state.first_token_ts - state.start_ts) * 1000

            tpot_ms = None
            if state.first_token_ts and state.end_ts and state.completion_tokens and state.completion_tokens > 1:
                tpot_ms = ((state.end_ts - state.first_token_ts) * 1000) / (state.completion_tokens - 1)

            sr = StepRecord(
                step_id=sid,
                agent_role=state.role,
                deps=dep_sids,
                ready_ts=state.start_ts,
                start_ts=state.start_ts,
                end_ts=state.end_ts,
                wait_ms=0.0,
                latency_ms=latency_ms,
                ttft_ms=ttft_ms,
                tpot_ms=tpot_ms,
                prompt_tokens=state.prompt_tokens,
                completion_tokens=state.completion_tokens,
                total_tokens=state.total_tokens or (state.prompt_tokens + state.completion_tokens),
                bytes_in=state.prompt_tokens * 4,
                bytes_out=state.completion_tokens * 4,
                ok=state.status == "ok",
                error=state.error,
                start_ns=state.start_ns if state.start_ns else None,
                first_token_ns=state.first_token_ns,
                end_ns=state.end_ns if state.end_ns else None,
                status=state.status,
            )
            steps[sid] = sr
            steps_raw[sid] = {
                "deps": dep_sids,
                "agent_role": state.role,
                "prompt_tokens": state.prompt_tokens,
                "completion_tokens": state.completion_tokens,
            }

        # Compute DAG metrics
        dag_raw = compute_dag_metrics(steps_raw)
        dag_metrics = DagMetrics(**dag_raw)

        # Compute role token stats
        role_stats_raw = compute_role_token_stats(steps_raw)
        role_token_stats = [RoleTokenStats(**rs) for rs in role_stats_raw]

        makespan_ms = (self._task_end_ts - self._task_start_ts) * 1000 if self._task_end_ts else 0.0
        tokens_exchanged = sum(r.total_tokens for r in steps.values())
        bytes_exchanged = sum(r.bytes_in + r.bytes_out for r in steps.values())

        return TaskRecord(
            task_id=task_id,
            task_start_ts=self._task_start_ts,
            task_end_ts=self._task_end_ts,
            makespan_ms=makespan_ms,
            steps=steps,
            messages_count=len(steps),
            tokens_exchanged=tokens_exchanged,
            bytes_exchanged=bytes_exchanged,
            critical_path_ms=0.0,  # caller can compute from step latencies
            total_idle_wait_ms=0.0,
            framework=self.framework,
            schema_version=2,
            dag_metrics=dag_metrics,
            role_token_stats=role_token_stats,
        )
