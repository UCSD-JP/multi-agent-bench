"""
Experimental OpenTelemetry SpanProcessor adapter for A2A distributed traces.

Captures agent name, token counts, and timing from OTel spans emitted by
A2A server implementations. For use in actual distributed A2A deployments.

Usage:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    adapter = OTelTraceAdapter()
    provider = TracerProvider()
    provider.add_span_processor(adapter)
    trace.set_tracer_provider(provider)

    # ... run A2A workflow ...
    task_record = adapter.to_task_record(task_id=0)

NOTE: This is experimental. A2A deployment patterns vary widely; this
adapter handles the common case of one span per agent invocation with
token counts in span attributes.
"""

import time
from typing import Dict, List, Optional

from benchmark.core.dag_metrics import compute_dag_metrics, compute_role_token_stats
from benchmark.core.types import DagMetrics, RoleTokenStats, StepRecord, TaskRecord

try:
    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

    class SpanProcessor:  # type: ignore[no-redef]
        pass

    class ReadableSpan:  # type: ignore[no-redef]
        pass


class _SpanRecord:
    __slots__ = (
        "agent_name", "role", "start_ns", "end_ns",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "status",
    )

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.role = "unknown"
        self.start_ns: int = 0
        self.end_ns: int = 0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.status: str = "ok"


class OTelTraceAdapter(SpanProcessor):
    """OTel SpanProcessor that captures A2A agent spans into TaskRecord format.

    Expects spans with attributes:
      - "agent.name" or "a2a.agent.name": agent identifier
      - "agent.role" or "a2a.agent.role": role (planner/executor/aggregator)
      - "llm.prompt_tokens": prompt token count
      - "llm.completion_tokens": completion token count
      - "llm.total_tokens": total token count
    """

    def __init__(self, dep_map: Optional[Dict[str, List[str]]] = None):
        self._spans: List[_SpanRecord] = []
        self.dep_map = dep_map or {}

    def on_start(self, span: "ReadableSpan", parent_context=None) -> None:
        pass

    def on_end(self, span: "ReadableSpan") -> None:
        attrs = span.attributes or {}

        agent_name = (
            attrs.get("agent.name")
            or attrs.get("a2a.agent.name")
            or span.name
        )
        if not agent_name:
            return

        rec = _SpanRecord(str(agent_name))
        rec.role = str(attrs.get("agent.role", attrs.get("a2a.agent.role", "unknown")))
        rec.start_ns = span.start_time or 0
        rec.end_ns = span.end_time or 0
        rec.prompt_tokens = int(attrs.get("llm.prompt_tokens", 0))
        rec.completion_tokens = int(attrs.get("llm.completion_tokens", 0))
        rec.total_tokens = int(attrs.get("llm.total_tokens", 0))

        if span.status and span.status.is_ok is False:
            rec.status = "error"

        self._spans.append(rec)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 0) -> bool:
        return True

    def to_task_record(self, task_id: int = 0) -> TaskRecord:
        """Build a TaskRecord from captured OTel spans."""
        steps: Dict[str, StepRecord] = {}
        steps_raw: Dict[str, Dict] = {}

        # Generate unique step IDs for repeated agent invocations
        sid_counts: Dict[str, int] = {}
        span_sids: List[str] = []
        for i, rec in enumerate(self._spans):
            base_sid = self._name_to_step_id(rec.agent_name, i)
            count = sid_counts.get(base_sid, 0)
            sid = base_sid if count == 0 else f"{base_sid}#{count}"
            sid_counts[base_sid] = count + 1
            span_sids.append(sid)

        for i, rec in enumerate(self._spans):
            sid = span_sids[i]
            deps = self.dep_map.get(rec.agent_name, [])

            # Convert ns timestamps to seconds for wall-clock fields
            start_ts = rec.start_ns / 1e9 if rec.start_ns else 0.0
            end_ts = rec.end_ns / 1e9 if rec.end_ns else 0.0
            latency_ms = (rec.end_ns - rec.start_ns) / 1e6 if (rec.end_ns and rec.start_ns) else 0.0

            sr = StepRecord(
                step_id=sid,
                agent_role=rec.role,
                deps=deps,
                ready_ts=start_ts,
                start_ts=start_ts,
                end_ts=end_ts,
                latency_ms=latency_ms,
                prompt_tokens=rec.prompt_tokens,
                completion_tokens=rec.completion_tokens,
                total_tokens=rec.total_tokens or (rec.prompt_tokens + rec.completion_tokens),
                bytes_in=rec.prompt_tokens * 4,
                bytes_out=rec.completion_tokens * 4,
                ok=rec.status == "ok",
                start_ns=rec.start_ns,
                end_ns=rec.end_ns,
                status=rec.status,
            )
            steps[sid] = sr
            steps_raw[sid] = {
                "deps": deps,
                "agent_role": rec.role,
                "prompt_tokens": rec.prompt_tokens,
                "completion_tokens": rec.completion_tokens,
            }

        dag_raw = compute_dag_metrics(steps_raw)
        dag_metrics = DagMetrics(**dag_raw)
        role_stats_raw = compute_role_token_stats(steps_raw)
        role_token_stats = [RoleTokenStats(**rs) for rs in role_stats_raw]

        all_start = min((s.start_ts for s in steps.values()), default=0.0)
        all_end = max((s.end_ts for s in steps.values()), default=0.0)
        makespan_ms = (all_end - all_start) * 1000

        return TaskRecord(
            task_id=task_id,
            task_start_ts=all_start,
            task_end_ts=all_end,
            makespan_ms=makespan_ms,
            steps=steps,
            messages_count=len(steps),
            tokens_exchanged=sum(s.total_tokens for s in steps.values()),
            bytes_exchanged=sum(s.bytes_in + s.bytes_out for s in steps.values()),
            critical_path_ms=0.0,
            total_idle_wait_ms=0.0,
            framework="a2a",
            schema_version=2,
            dag_metrics=dag_metrics,
            role_token_stats=role_token_stats,
        )

    @staticmethod
    def _name_to_step_id(name: str, index: int) -> str:
        lower = name.lower()
        if "planner" in lower:
            return "P"
        elif "executor" in lower:
            import re
            m = re.search(r"(\d+)", name)
            return f"E{m.group(1)}" if m else f"E{index}"
        elif "aggregator" in lower:
            return "A"
        return f"S{index}"
