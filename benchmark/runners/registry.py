"""Runner registry — maps --framework name to WorkflowRunner class."""

from typing import Dict, Type

from .base import WorkflowRunner
from .raw_runner import RawRunner

_RUNNERS: Dict[str, Type[WorkflowRunner]] = {}


def register(name: str, cls: Type[WorkflowRunner]) -> None:
    _RUNNERS[name] = cls


def get_runner(name: str) -> WorkflowRunner:
    if name not in _RUNNERS:
        available = ", ".join(sorted(_RUNNERS.keys())) or "(none)"
        raise ValueError(f"Unknown framework '{name}'. Available: {available}")
    return _RUNNERS[name]()


def available_frameworks():
    return sorted(_RUNNERS.keys())


# Built-in
register("raw", RawRunner)

# Lazy: autogen (skip if not installed)
try:
    from benchmark.autogen_ext.runner import AutoGenRunner
    register("autogen", AutoGenRunner)
except ImportError:
    pass
