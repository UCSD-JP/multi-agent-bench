"""
Google A2A (Agent-to-Agent) runner — protocol-based multi-agent workflow.

Uses Google's A2A protocol where agents communicate via standardized
JSON-RPC messages, contrasting with framework-internal orchestration.

Install: pip install -r requirements-a2a.txt
"""

import time
from typing import Dict, List

from benchmark.core.metrics import critical_path_dag_ms
from benchmark.core.types import StepRecord, TaskRecord
from benchmark.runners.base import RunContext, WorkflowRunner

try:
    import a2a

    _A2A_AVAILABLE = True
except ImportError:
    _A2A_AVAILABLE = False


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
        if not _A2A_AVAILABLE:
            raise ImportError(
                "Google A2A SDK is not installed. "
                "Install with: pip install -r requirements-a2a.txt"
            )

        # TODO: Implement A2A protocol workflow
        # Skeleton:
        #   1. Start agent servers (Planner, Executor, Aggregator) as A2A agents
        #   2. Each agent exposes JSON-RPC endpoint with AgentCard
        #   3. Orchestrator sends tasks via A2A protocol messages
        #   4. Agents respond with streaming artifacts
        #   5. Collect StepRecords from message exchange times
        #
        # Key difference from AutoGen/LangGraph:
        #   - Agents are independent services (protocol-level decoupling)
        #   - Communication via standardized JSON-RPC (interoperable)
        #   - True distributed execution possible
        #   - Framework-agnostic: any A2A-compliant agent can participate

        raise NotImplementedError(
            "A2A runner is not yet implemented. "
            "Contributions welcome — see benchmark/a2a_ext/runner.py"
        )
