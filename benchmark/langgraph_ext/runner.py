"""
LangGraph runner — StateGraph-based multi-agent workflow.

Uses LangGraph's StateGraph where nodes are agent functions and edges
define the execution flow, contrasting with AutoGen's dynamic speaker selection.

Install: pip install -r requirements-langgraph.txt
"""

import time
from typing import Dict, List

from benchmark.core.metrics import critical_path_dag_ms
from benchmark.core.prompts import PromptBuilder
from benchmark.core.types import StepRecord, TaskRecord
from benchmark.runners.base import RunContext, WorkflowRunner

try:
    from langgraph.graph import StateGraph, END

    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False


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

        # TODO: Implement LangGraph StateGraph workflow
        # Skeleton:
        #   1. Define TypedDict state schema
        #   2. Create agent node functions (planner, executor, aggregator)
        #      - Each calls context.streaming_client for TTFT/TPOT measurement
        #   3. Build StateGraph: planner -> fan-out executors -> aggregator
        #   4. Compile and invoke graph
        #   5. Collect StepRecords from node execution times
        #
        # Key difference from AutoGen:
        #   - Edges are explicit (coded graph), not LLM-selected
        #   - Supports conditional edges for dynamic routing
        #   - Native async support with fan-out/fan-in patterns

        raise NotImplementedError(
            "LangGraph runner is not yet implemented. "
            "Contributions welcome — see benchmark/langgraph_ext/runner.py"
        )
