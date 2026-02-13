"""Runner registry — maps --framework name to WorkflowRunner class."""

from typing import Dict, List, Type

from .base import WorkflowRunner

_RUNNERS: Dict[str, Type[WorkflowRunner]] = {}

DEFAULT_FRAMEWORK = "autogen"


def register(name: str, cls: Type[WorkflowRunner]) -> None:
    _RUNNERS[name] = cls


def get_runner(name: str) -> WorkflowRunner:
    if name not in _RUNNERS:
        available = ", ".join(sorted(_RUNNERS.keys())) or "(none)"
        raise ValueError(
            f"Unknown framework '{name}'. Available: {available}\n"
            f"Install the required package — see requirements-<framework>.txt"
        )
    return _RUNNERS[name]()


def available_frameworks() -> List[str]:
    return sorted(_RUNNERS.keys())


def default_framework() -> str:
    return DEFAULT_FRAMEWORK


# === Lazy registration: skip if not installed ===

# AutoGen (default)
try:
    from benchmark.autogen_ext.runner import AutoGenRunner
    register("autogen", AutoGenRunner)
except ImportError:
    pass

# LangGraph
try:
    from benchmark.langgraph_ext.runner import LangGraphRunner
    register("langgraph", LangGraphRunner)
except ImportError:
    pass

# Google A2A
try:
    from benchmark.a2a_ext.runner import A2ARunner
    register("a2a", A2ARunner)
except ImportError:
    pass
