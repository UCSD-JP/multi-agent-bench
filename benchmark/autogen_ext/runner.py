"""
AutoGenRunner — SelectorGroupChat-based multi-agent workflow.

Uses AutoGen's SelectorGroupChat where an LLM dynamically selects the next
speaker, contrasting with the raw runner's hardcoded diamond DAG.
"""

import time
from typing import Dict, List

from benchmark.core.metrics import critical_path_dag_ms
from benchmark.core.types import LLMCallMetrics, StepRecord, TaskRecord
from benchmark.runners.base import RunContext, WorkflowRunner

from .instrumented_client import InstrumentedModelClient

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import (
        MaxMessageTermination,
        TextMentionTermination,
    )
    from autogen_agentchat.teams import SelectorGroupChat

    _AUTOGEN_AVAILABLE = True
except ImportError:
    _AUTOGEN_AVAILABLE = False


# Selector prompt template that guides LLM speaker selection
_SELECTOR_PROMPT = """\
You are managing a team of agents to complete a user task.
The agents are:
- Planner: Creates a plan with numbered steps. Should speak FIRST.
- Executor0: Executes part of the plan. Should speak AFTER the Planner.
- Executor1: Executes another part of the plan. Should speak AFTER the Planner.
- Aggregator: Combines all outputs into a final answer. Should speak LAST after all Executors.

Given the conversation so far, select the next agent to speak.
Only return the agent name (Planner, Executor0, Executor1, or Aggregator).
When the Aggregator has produced the final answer, the task is complete.
"""


class AutoGenRunner(WorkflowRunner):

    @property
    def name(self) -> str:
        return "autogen"

    async def run_task(
        self,
        task_id: int,
        prompt: str,
        context: RunContext,
    ) -> TaskRecord:
        if not _AUTOGEN_AVAILABLE:
            raise ImportError(
                "AutoGen is not installed. "
                "Install with: pip install 'autogen-agentchat' 'autogen-ext[openai]'"
            )

        task_start = time.time()

        # Create instrumented model client for agent LLM calls
        agent_client = InstrumentedModelClient(
            streaming_client=context.streaming_client,
            http_client=context.http_client,
            model=context.model,
            semaphore=context.llm_semaphore,
            temperature=context.temperature,
        )

        # Selector uses same client (its calls are tracked separately)
        selector_client = InstrumentedModelClient(
            streaming_client=context.streaming_client,
            http_client=context.http_client,
            model=context.model,
            semaphore=context.llm_semaphore,
            temperature=context.temperature,
        )

        # Create agents
        planner = AssistantAgent(
            name="Planner",
            model_client=agent_client,
            system_message=(
                "You are the Planner agent. Given a user request, produce a short plan "
                "with numbered steps. Be concise."
            ),
        )

        executors = []
        for i in range(context.executors):
            executor = AssistantAgent(
                name=f"Executor{i}",
                model_client=agent_client,
                system_message=(
                    f"You are Executor{i}. Execute your assigned part of the plan. "
                    "Be concise but correct. Focus on your specific portion."
                ),
            )
            executors.append(executor)

        aggregator = AssistantAgent(
            name="Aggregator",
            model_client=agent_client,
            system_message=(
                "You are the Aggregator agent. Combine the Planner's plan and "
                "all Executors' outputs into one final, coherent answer. "
                "End your response with TERMINATE."
            ),
        )

        # Termination: stop when Aggregator says TERMINATE or max messages reached
        # Expected flow: 1 (Planner) + N (Executors) + 1 (Aggregator) + margin
        max_msgs = 2 + context.executors + 5  # some margin for selector flexibility
        termination = (
            TextMentionTermination("TERMINATE")
            | MaxMessageTermination(max_messages=max_msgs)
        )

        # Build SelectorGroupChat
        participants = [planner] + executors + [aggregator]
        team = SelectorGroupChat(
            participants=participants,
            model_client=selector_client,
            termination_condition=termination,
            selector_prompt=_SELECTOR_PROMPT,
            allow_repeated_speaker=False,
        )

        # Run the team
        result = await team.run(task=prompt)

        task_end = time.time()

        # Collect agent metrics (excludes selector overhead)
        agent_metrics = agent_client.pop_metrics()
        selector_metrics = selector_client.pop_metrics()

        # Build step records from the agent call metrics
        # Each agent LLM call maps to a step in execution order
        steps: Dict[str, StepRecord] = {}
        turn_order: List[str] = []
        prev_step_id = None

        for i, m in enumerate(agent_metrics):
            # Determine agent role from call order
            # AutoGen calls agents in the order selected by the selector
            step_id = self._infer_step_id(i, len(agent_metrics), context.executors)
            agent_role = self._role_from_step_id(step_id)
            turn_order.append(step_id)

            # Dependencies: each step depends on the previous (sequential in SelectorGroupChat)
            deps = [prev_step_id] if prev_step_id else []

            # Compute bytes
            bytes_in = int((m.prompt_tokens or 0) * 4)  # rough estimate
            bytes_out = len(m.out_text.encode("utf-8"))

            ready_ts = m.start_ts
            if prev_step_id and prev_step_id in steps:
                ready_ts = steps[prev_step_id].end_ts

            wait_ms = max(0.0, (m.start_ts - ready_ts) * 1000)

            steps[step_id] = StepRecord(
                step_id=step_id,
                agent_role=agent_role,
                deps=deps,
                ready_ts=ready_ts,
                start_ts=m.start_ts,
                end_ts=m.end_ts,
                wait_ms=wait_ms,
                latency_ms=(m.end_ts - m.start_ts) * 1000,
                ttft_ms=m.ttft_ms,
                tpot_ms=m.tpot_ms,
                prompt_tokens=int(m.prompt_tokens or 0),
                completion_tokens=int(m.out_tokens or 0),
                total_tokens=int(m.total_tokens or 0),
                bytes_in=bytes_in,
                bytes_out=bytes_out,
                ok=m.ok,
                error=m.error,
            )
            prev_step_id = step_id

        # Selector overhead
        selector_overhead_ms = sum(
            (m.end_ts - m.start_ts) * 1000 for m in selector_metrics
        )

        makespan_ms = (task_end - task_start) * 1000
        messages_count = len(agent_metrics)
        tokens_exchanged = sum(r.total_tokens for r in steps.values())
        bytes_exchanged = sum(r.bytes_in + r.bytes_out for r in steps.values())
        total_idle_wait_ms = sum(r.wait_ms for r in steps.values())

        step_durations = {sid: r.latency_ms for sid, r in steps.items()}
        deps_map = {sid: r.deps for sid, r in steps.items()}
        cp_ms = critical_path_dag_ms(step_durations, deps_map) if steps else 0.0

        # Cleanup
        await agent_client.close()
        await selector_client.close()

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
            framework="autogen",
            selector_overhead_ms=selector_overhead_ms,
            turn_order=turn_order,
        )

    @staticmethod
    def _infer_step_id(call_idx: int, total_calls: int, executors: int) -> str:
        """
        Map sequential call index to step ID.
        Expected order: P, E0, E1, ..., A
        If more calls than expected, append suffix.
        """
        if call_idx == 0:
            return "P"
        elif call_idx <= executors:
            return f"E{call_idx - 1}"
        elif call_idx == executors + 1:
            return "A"
        else:
            # Extra calls beyond expected pattern
            return f"X{call_idx}"

    @staticmethod
    def _role_from_step_id(step_id: str) -> str:
        if step_id == "P":
            return "planner"
        elif step_id.startswith("E"):
            return "executor"
        elif step_id == "A":
            return "aggregator"
        else:
            return "unknown"
