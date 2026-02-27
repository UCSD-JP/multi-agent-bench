"""
AutoGenRunner — orchestrated multi-agent workflow using AutoGen agents.

Uses AutoGen's AssistantAgent with InstrumentedModelClient for TTFT/TPOT
measurement.  Agents are called sequentially via on_messages() in a fixed
P → E0 → E1 → ... → A order.  This avoids SelectorGroupChat's internal
selector-retry infinite loops while preserving AutoGen's agent abstraction
and model client protocol.
"""

import asyncio
import re
import time
from typing import Dict, List

from benchmark.core.dag_metrics import compute_dag_metrics, compute_role_token_stats
from benchmark.core.metrics import critical_path_dag_ms
from benchmark.core.types import (
    DagMetrics,
    LLMCallMetrics,
    RoleTokenStats,
    StepRecord,
    TaskRecord,
)
from benchmark.runners.base import RunContext, WorkflowRunner

from .instrumented_client import InstrumentedModelClient

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import TextMessage

    _AUTOGEN_AVAILABLE = True
except ImportError:
    _AUTOGEN_AVAILABLE = False


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
            max_model_len=context.max_model_len,
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
                "all Executors' outputs into one final, coherent answer."
            ),
        )

        # Orchestrate: P → E0 → E1 → ... → A
        # Each agent receives the conversation history via on_messages()
        conversation: List[TextMessage] = []
        agent_order = [planner] + executors + [aggregator]
        turn_order: List[str] = []

        # Initial user message
        user_msg = TextMessage(content=prompt, source="user")
        conversation.append(user_msg)

        for agent in agent_order:
            # Count metrics before this call
            n_before = len(agent_client._call_records)

            # Send conversation to agent
            response = await agent.on_messages(conversation, cancellation_token=None)

            # Extract the agent's response
            if response.chat_message is not None:
                conversation.append(response.chat_message)

            turn_order.append(agent.name)

        task_end = time.time()

        # Collect all metrics
        agent_metrics = agent_client.pop_metrics()

        # Build step records
        steps: Dict[str, StepRecord] = {}
        prev_step_id = None
        _name_counts: Dict[str, int] = {}

        for i, m in enumerate(agent_metrics):
            if i < len(turn_order):
                speaker = turn_order[i]
                step_id = self._speaker_to_step_id(speaker, _name_counts)
                agent_role = self._role_from_speaker(speaker)
            else:
                step_id = f"X{i}"
                agent_role = "unknown"

            deps = [prev_step_id] if prev_step_id else []

            bytes_in = int((m.prompt_tokens or 0) * 4)
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
                # v2: monotonic ns timestamps
                start_ns=m.start_ns,
                first_token_ns=m.first_token_ns,
                end_ns=m.end_ns,
                status="error" if not m.ok else "ok",
            )
            prev_step_id = step_id

        makespan_ms = (task_end - task_start) * 1000
        messages_count = len(agent_metrics)
        tokens_exchanged = sum(r.total_tokens for r in steps.values())
        bytes_exchanged = sum(r.bytes_in + r.bytes_out for r in steps.values())
        total_idle_wait_ms = sum(r.wait_ms for r in steps.values())

        step_durations = {sid: r.latency_ms for sid, r in steps.items()}
        deps_map = {sid: r.deps for sid, r in steps.items()}
        cp_ms = critical_path_dag_ms(step_durations, deps_map) if steps else 0.0

        await agent_client.close()

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
            framework="autogen",
            selector_overhead_ms=0.0,
            turn_order=list(steps.keys()),
            schema_version=2,
            dag_metrics=dag_metrics,
            role_token_stats=role_token_stats,
        )

    @staticmethod
    def _speaker_to_step_id(speaker: str, counts: Dict[str, int]) -> str:
        """Map actual AutoGen speaker name to a short step ID."""
        n = counts.get(speaker, 0)
        counts[speaker] = n + 1

        lower = speaker.lower()
        if "planner" in lower:
            return "P" if n == 0 else f"P#{n}"
        elif "executor" in lower:
            m = re.search(r"(\d+)", speaker)
            idx = m.group(1) if m else "0"
            base = f"E{idx}"
            return base if n == 0 else f"{base}#{n}"
        elif "aggregator" in lower:
            return "A" if n == 0 else f"A#{n}"
        else:
            return f"{speaker[:3].upper()}#{n}" if n > 0 else speaker[:3].upper()

    @staticmethod
    def _role_from_speaker(speaker: str) -> str:
        """Map actual AutoGen agent name to a role label."""
        lower = speaker.lower()
        if "planner" in lower:
            return "planner"
        elif "executor" in lower:
            return "executor"
        elif "aggregator" in lower:
            return "aggregator"
        else:
            return "unknown"
