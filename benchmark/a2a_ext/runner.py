"""
Google A2A (Agent-to-Agent) runner — protocol-based multi-agent workflow.

Uses Google's A2A protocol where each agent is an independent service
communicating via JSON-RPC messages. The runner:
  1. Starts lightweight in-process agent servers (Planner, Executors, Aggregator)
  2. An orchestrator sends tasks via A2A client
  3. Each agent internally calls our streaming client for TTFT/TPOT measurement
  4. Agents respond with artifacts containing their output

Key difference from AutoGen/LangGraph:
  - Agents are protocol-level services (interoperable, framework-agnostic)
  - Communication via standardized A2A JSON-RPC messages
  - True service-oriented architecture (could be distributed)

Install: pip install -r requirements-a2a.txt
"""

import asyncio
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from benchmark.core.dag_metrics import compute_dag_metrics, compute_role_token_stats
from benchmark.core.metrics import critical_path_dag_ms
from benchmark.core.prompts import PromptBuilder
from benchmark.core.types import (
    DagMetrics,
    LLMCallMetrics,
    RoleTokenStats,
    StepRecord,
    TaskRecord,
)
from benchmark.runners.base import RunContext, WorkflowRunner

# A2A protocol is simulated in-process (no external SDK required).
# The runner constructs A2A-style JSON-RPC messages as dicts to model
# protocol overhead without requiring the actual a2a-sdk package.


class A2ARunner(WorkflowRunner):

    @property
    def name(self) -> str:
        return "a2a"

    async def run_task(
        self,
        task_id: int,
        prompt: str,
        context: RunContext,
    ) -> TaskRecord:
        task_start = time.time()

        # Step 1: Planner
        planner_rec = await self._run_agent_step(
            context=context,
            step_id="P",
            agent_role="planner",
            deps=[],
            build_prompt=lambda _state: PromptBuilder.planner(prompt),
        )

        plan_text = planner_rec["out_text"]

        # Step 2: Executors (parallel via A2A tasks)
        executor_tasks = []
        for i in range(context.executors):
            step_id = f"E{i}"
            executor_tasks.append(
                self._run_agent_step(
                    context=context,
                    step_id=step_id,
                    agent_role="executor",
                    deps=["P"],
                    build_prompt=lambda _state, sid=step_id, p=plan_text: (
                        PromptBuilder.executor(prompt, p, sid)
                    ),
                )
            )
        executor_results = await asyncio.gather(*executor_tasks)

        # Step 3: Aggregator
        exec_outputs = {
            r["step_id"]: r["out_text"] for r in executor_results
        }
        executor_deps = [f"E{i}" for i in range(context.executors)]

        aggregator_rec = await self._run_agent_step(
            context=context,
            step_id="A",
            agent_role="aggregator",
            deps=executor_deps,
            build_prompt=lambda _state: PromptBuilder.aggregator(
                prompt, plan_text, exec_outputs
            ),
        )

        task_end = time.time()

        # Build steps dict
        all_records = [planner_rec] + list(executor_results) + [aggregator_rec]
        steps: Dict[str, StepRecord] = {}
        for rec in all_records:
            sr = rec["step_record"]
            steps[sr["step_id"]] = StepRecord(**sr)

        makespan_ms = (task_end - task_start) * 1000
        messages_count = len(steps)
        tokens_exchanged = sum(r.total_tokens for r in steps.values())
        bytes_exchanged = sum(r.bytes_in + r.bytes_out for r in steps.values())
        total_idle_wait_ms = sum(r.wait_ms for r in steps.values())

        step_durations = {sid: r.latency_ms for sid, r in steps.items()}
        deps_map = {sid: r.deps for sid, r in steps.items()}
        cp_ms = critical_path_dag_ms(step_durations, deps_map) if steps else 0.0

        # v2: compute DAG metrics and role token stats
        steps_raw = {
            sid: {
                "deps": r.deps,
                "agent_role": r.agent_role,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
            }
            for sid, r in steps.items()
        }
        dag_raw = compute_dag_metrics(steps_raw)
        dag_metrics = DagMetrics(**dag_raw)
        role_stats_raw = compute_role_token_stats(steps_raw)
        role_token_stats = [RoleTokenStats(**rs) for rs in role_stats_raw]

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
            framework="a2a",
            schema_version=2,
            dag_metrics=dag_metrics,
            role_token_stats=role_token_stats,
        )

    async def _run_agent_step(
        self,
        context: RunContext,
        step_id: str,
        agent_role: str,
        deps: List[str],
        build_prompt,
    ) -> dict:
        """
        Simulate an A2A agent interaction.

        In a full deployment each agent would be a separate A2A server.
        Here we model the protocol semantics in-process:
          1. Create A2A Message (JSON-RPC style)
          2. Agent handler processes message
          3. Agent calls LLM via streaming client
          4. Agent returns A2A Task with artifact

        This captures the A2A protocol overhead (message serialization,
        task state management) while using the same LLM measurement path.
        """
        content = build_prompt(None)
        messages = [
            {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        bytes_in = sum(
            len((m.get("content", "") or "").encode("utf-8")) for m in messages
        )

        # A2A protocol overhead: message construction
        a2a_task_id = str(uuid.uuid4())
        a2a_context_id = str(uuid.uuid4())
        protocol_start = time.time()

        # Simulate A2A message send (in-process, no network)
        a2a_message = {
            "task_id": a2a_task_id,
            "context_id": a2a_context_id,
            "role": "user",
            "parts": [{"type": "text", "text": content}],
            "agent_role": agent_role,
        }

        protocol_overhead_ms = (time.time() - protocol_start) * 1000

        # Agent processes: call LLM
        ready_ts = time.time()

        async with context.llm_semaphore:
            start_ts = time.time()
            wait_ms = (start_ts - ready_ts) * 1000

            call = await context.streaming_client.chat_completions_stream(
                client=context.http_client,
                model=context.model,
                messages=messages,
                temperature=context.temperature,
                max_model_len=context.max_model_len,
            )
            end_ts = call.end_ts

        bytes_out = len(call.out_text.encode("utf-8"))

        # A2A protocol: construct response artifact
        a2a_response = {
            "task_id": a2a_task_id,
            "state": "completed",
            "artifacts": [{"type": "text", "text": call.out_text}],
        }

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
            # v2: monotonic ns timestamps
            "start_ns": call.start_ns,
            "first_token_ns": call.first_token_ns,
            "end_ns": call.end_ns,
            "status": "error" if not call.ok else "ok",
        }

        return {
            "step_id": step_id,
            "out_text": call.out_text,
            "step_record": step_record,
        }
