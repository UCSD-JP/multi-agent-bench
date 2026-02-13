"""
LangGraph runner — StateGraph-based multi-agent workflow.

Uses LangGraph's StateGraph with explicit edges:
  planner → fan-out(executor0, executor1, ...) → aggregator

Key difference from AutoGen:
  - Graph topology is coded (not LLM-selected)
  - Supports true parallel fan-out via multiple edges from one node
  - Each node calls our streaming client for TTFT/TPOT measurement

Install: pip install -r requirements-langgraph.txt
"""

import operator
import time
from typing import Annotated, Any, Dict, List, Optional

from benchmark.core.metrics import critical_path_dag_ms
from benchmark.core.prompts import PromptBuilder
from benchmark.core.types import LLMCallMetrics, StepRecord, TaskRecord
from benchmark.runners.base import RunContext, WorkflowRunner

try:
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False


if _LANGGRAPH_AVAILABLE:

    class WorkflowState(TypedDict):
        """State that flows through the graph."""
        user_prompt: str
        plan: str
        executor_outputs: Annotated[list, operator.add]
        final_answer: str
        # Internal: metrics collection
        step_records: Annotated[list, operator.add]


class LangGraphRunner(WorkflowRunner):

    @property
    def name(self) -> str:
        return "langgraph"

    async def run_task(
        self,
        task_id: int,
        prompt: str,
        context: RunContext,
    ) -> TaskRecord:
        if not _LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph is not installed. "
                "Install with: pip install -r requirements-langgraph.txt"
            )

        task_start = time.time()

        # Build the graph dynamically based on executor count
        graph = self._build_graph(context)
        compiled = graph.compile()

        # Run the graph
        result = await compiled.ainvoke({
            "user_prompt": prompt,
            "plan": "",
            "executor_outputs": [],
            "final_answer": "",
            "step_records": [],
        })

        task_end = time.time()

        # Convert collected step records to StepRecord objects
        steps: Dict[str, StepRecord] = {}
        for rec_dict in result.get("step_records", []):
            sid = rec_dict["step_id"]
            steps[sid] = StepRecord(**rec_dict)

        makespan_ms = (task_end - task_start) * 1000
        messages_count = len(steps)
        tokens_exchanged = sum(r.total_tokens for r in steps.values())
        bytes_exchanged = sum(r.bytes_in + r.bytes_out for r in steps.values())
        total_idle_wait_ms = sum(r.wait_ms for r in steps.values())

        step_durations = {sid: r.latency_ms for sid, r in steps.items()}
        deps_map = {sid: r.deps for sid, r in steps.items()}
        cp_ms = critical_path_dag_ms(step_durations, deps_map) if steps else 0.0

        return TaskRecord(
            task_id=task_id,
            task_start_ts=task_start,
            task_end_ts=task_end,
            makespan_ms=makespan_ms,
            steps=steps,
            messages_count=messages_count,
            tokens_exchanged=tokens_exchanged,
            bytes_exchanged=bytes_exchanged,
            critical_path_ms=cp_ms,
            total_idle_wait_ms=total_idle_wait_ms,
            framework="langgraph",
        )

    def _build_graph(self, context: RunContext) -> "StateGraph":
        """Build a StateGraph: planner → fan-out executors → aggregator."""

        async def planner_node(state: "WorkflowState") -> dict:
            content = PromptBuilder.planner(state["user_prompt"])
            messages = [
                {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
            rec = await self._call_llm(
                context, messages, step_id="P", agent_role="planner", deps=[]
            )
            return {
                "plan": rec["out_text"],
                "step_records": [rec["step_record"]],
            }

        def _make_executor_node(exec_idx: int):
            """Factory to create executor node functions with captured index."""

            async def executor_node(state: "WorkflowState") -> dict:
                step_id = f"E{exec_idx}"
                content = PromptBuilder.executor(
                    state["user_prompt"], state["plan"], step_id
                )
                messages = [
                    {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ]
                rec = await self._call_llm(
                    context, messages, step_id=step_id,
                    agent_role="executor", deps=["P"]
                )
                return {
                    "executor_outputs": [
                        {"step_id": step_id, "text": rec["out_text"]}
                    ],
                    "step_records": [rec["step_record"]],
                }

            return executor_node

        async def aggregator_node(state: "WorkflowState") -> dict:
            exec_outputs = {
                o["step_id"]: o["text"] for o in state["executor_outputs"]
            }
            executor_deps = [o["step_id"] for o in state["executor_outputs"]]
            content = PromptBuilder.aggregator(
                state["user_prompt"], state["plan"], exec_outputs
            )
            messages = [
                {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
            rec = await self._call_llm(
                context, messages, step_id="A",
                agent_role="aggregator", deps=executor_deps
            )
            return {
                "final_answer": rec["out_text"],
                "step_records": [rec["step_record"]],
            }

        # Assemble graph
        builder = StateGraph(WorkflowState)
        builder.add_node("planner", planner_node)
        for i in range(context.executors):
            builder.add_node(f"executor_{i}", _make_executor_node(i))
        builder.add_node("aggregator", aggregator_node)

        # Edges: START → planner → fan-out executors → aggregator → END
        builder.add_edge(START, "planner")
        for i in range(context.executors):
            builder.add_edge("planner", f"executor_{i}")
            builder.add_edge(f"executor_{i}", "aggregator")
        builder.add_edge("aggregator", END)

        return builder

    async def _call_llm(
        self,
        context: RunContext,
        messages: List[Dict[str, str]],
        step_id: str,
        agent_role: str,
        deps: List[str],
    ) -> dict:
        """Call LLM via streaming client and return output + step record dict."""
        bytes_in = sum(
            len((m.get("content", "") or "").encode("utf-8")) for m in messages
        )

        ready_ts = time.time()

        async with context.llm_semaphore:
            start_ts = time.time()
            wait_ms = (start_ts - ready_ts) * 1000

            call = await context.streaming_client.chat_completions_stream(
                client=context.http_client,
                model=context.model,
                messages=messages,
                temperature=context.temperature,
            )
            end_ts = call.end_ts

        bytes_out = len(call.out_text.encode("utf-8"))

        step_record = {
            "step_id": step_id,
            "agent_role": agent_role,
            "deps": deps,
            "ready_ts": ready_ts,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "wait_ms": max(0.0, wait_ms),
            "latency_ms": (end_ts - start_ts) * 1000,
            "ttft_ms": call.ttft_ms,
            "tpot_ms": call.tpot_ms,
            "prompt_tokens": int(call.prompt_tokens or 0),
            "completion_tokens": int(call.out_tokens or 0),
            "total_tokens": int(call.total_tokens or 0),
            "bytes_in": bytes_in,
            "bytes_out": bytes_out,
            "ok": call.ok,
            "error": call.error,
        }

        return {"out_text": call.out_text, "step_record": step_record}
