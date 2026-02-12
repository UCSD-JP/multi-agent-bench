"""Raw runner — hardcoded diamond DAG (P → E0∥E1 → A), baseline implementation."""

import asyncio
import time
from typing import Any, Dict, List

from benchmark.core.metrics import critical_path_dag_ms
from benchmark.core.prompts import PromptBuilder
from benchmark.core.types import StepRecord, StepSpec, TaskRecord

from .base import RunContext, WorkflowRunner


def build_workflow(executors: int) -> List[StepSpec]:
    """
    Diamond DAG: planner -> parallel executors -> aggregator.
    """
    specs: List[StepSpec] = []
    specs.append(StepSpec(step_id="P", agent_role="planner", deps=[]))
    for i in range(executors):
        specs.append(StepSpec(step_id=f"E{i}", agent_role="executor", deps=["P"]))
    specs.append(StepSpec(step_id="A", agent_role="aggregator", deps=[f"E{i}" for i in range(executors)]))
    return specs


class RawRunner(WorkflowRunner):

    @property
    def name(self) -> str:
        return "raw"

    async def run_task(
        self,
        task_id: int,
        prompt: str,
        context: RunContext,
    ) -> TaskRecord:
        workflow = build_workflow(context.executors)
        task_start = time.time()

        done: Dict[str, StepRecord] = {}
        state: Dict[str, Any] = {"user_prompt": prompt, "planner": "", "executions": {}}

        async def run_one_step(step: StepSpec) -> StepRecord:
            # Wait until deps finished
            while True:
                if all(d in done for d in step.deps):
                    break
                await asyncio.sleep(0.001)

            rts = time.time()
            if step.deps:
                rts = max(done[d].end_ts for d in step.deps)
            else:
                rts = task_start

            # Compose prompt via PromptBuilder
            if step.agent_role == "planner":
                content = PromptBuilder.planner(state["user_prompt"])
            elif step.agent_role == "executor":
                content = PromptBuilder.executor(
                    state["user_prompt"], state.get("planner", ""), step.step_id
                )
            elif step.agent_role == "aggregator":
                content = PromptBuilder.aggregator(
                    state["user_prompt"], state.get("planner", ""), state["executions"]
                )
            else:
                content = f"Role={step.agent_role}\n\nUser:\n{state['user_prompt']}\n"

            messages = [
                {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]

            bytes_in = sum(
                len((m.get("content", "") or "").encode("utf-8")) for m in messages
            )

            async with context.llm_semaphore:
                start_ts = time.time()
                wait_ms = (start_ts - rts) * 1000

                call = await context.streaming_client.chat_completions_stream(
                    client=context.http_client,
                    model=context.model,
                    messages=messages,
                    temperature=context.temperature,
                )
                end_ts = call.end_ts

            bytes_out = len(call.out_text.encode("utf-8"))

            rec = StepRecord(
                step_id=step.step_id,
                agent_role=step.agent_role,
                deps=list(step.deps),
                ready_ts=rts,
                start_ts=start_ts,
                end_ts=end_ts,
                wait_ms=max(0.0, wait_ms),
                latency_ms=(end_ts - start_ts) * 1000,
                ttft_ms=call.ttft_ms,
                tpot_ms=call.tpot_ms,
                prompt_tokens=int(call.prompt_tokens or 0),
                completion_tokens=int(call.out_tokens or 0),
                total_tokens=int(call.total_tokens or 0),
                bytes_in=bytes_in,
                bytes_out=bytes_out,
                ok=call.ok,
                error=call.error,
            )

            # Update shared state
            if call.ok:
                if step.agent_role == "planner":
                    state["planner"] = call.out_text
                elif step.agent_role == "executor":
                    state["executions"][step.step_id] = call.out_text
                elif step.agent_role == "aggregator":
                    state["final"] = call.out_text

            return rec

        # Launch all steps; each waits for its deps internally
        step_tasks = {s.step_id: asyncio.create_task(run_one_step(s)) for s in workflow}

        for sid, t in step_tasks.items():
            rec = await t
            done[sid] = rec

        task_end = time.time()
        makespan_ms = (task_end - task_start) * 1000

        messages_count = len(workflow)
        tokens_exchanged = sum(r.total_tokens for r in done.values())
        bytes_exchanged = sum(r.bytes_in + r.bytes_out for r in done.values())
        total_idle_wait_ms = sum(r.wait_ms for r in done.values())

        step_durations = {sid: r.latency_ms for sid, r in done.items()}
        deps_map = {sid: r.deps for sid, r in done.items()}
        cp_ms = critical_path_dag_ms(step_durations, deps_map)

        return TaskRecord(
            task_id=task_id,
            task_start_ts=task_start,
            task_end_ts=task_end,
            makespan_ms=makespan_ms,
            steps=done,
            messages_count=messages_count,
            tokens_exchanged=tokens_exchanged,
            bytes_exchanged=bytes_exchanged,
            critical_path_ms=cp_ms,
            total_idle_wait_ms=total_idle_wait_ms,
            framework="raw",
        )
