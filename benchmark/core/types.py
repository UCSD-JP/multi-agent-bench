"""Core data types for multi-agent benchmark tracing."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LLMCallMetrics:
    ok: bool
    start_ts: float
    first_token_ts: Optional[float]
    end_ts: float
    ttft_ms: Optional[float]
    tpot_ms: Optional[float]
    out_tokens: Optional[int]
    prompt_tokens: Optional[int]
    total_tokens: Optional[int]
    out_text: str
    error: Optional[str] = None
    # v2: monotonic nanosecond timestamps (no clock skew for deltas)
    start_ns: Optional[int] = None
    first_token_ns: Optional[int] = None
    end_ns: Optional[int] = None


@dataclass
class StepSpec:
    step_id: str
    agent_role: str
    deps: List[str] = field(default_factory=list)


@dataclass
class StepRecord:
    step_id: str
    agent_role: str
    deps: List[str]
    ready_ts: float = 0.0
    start_ts: float = 0.0
    end_ts: float = 0.0
    wait_ms: float = 0.0
    latency_ms: float = 0.0
    ttft_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    ok: bool = True
    error: Optional[str] = None
    # v2: monotonic nanosecond timestamps + status
    start_ns: Optional[int] = None
    first_token_ns: Optional[int] = None
    end_ns: Optional[int] = None
    status: str = "ok"


@dataclass
class DagMetrics:
    """DAG topology metrics computed from step dependency graph."""
    depth: int = 0
    max_width: int = 0
    fanout_max: int = 0
    fanin_max: int = 0
    critical_path_steps: List[str] = field(default_factory=list)
    critical_path_len: int = 0
    parallel_fraction: float = 0.0


@dataclass
class RoleTokenStats:
    """Per-role aggregate token statistics."""
    role: str
    count: int = 0
    prompt_mean: float = 0.0
    prompt_std: float = 0.0
    output_mean: float = 0.0
    output_std: float = 0.0


@dataclass
class TaskRecord:
    task_id: int
    task_start_ts: float
    task_end_ts: float
    makespan_ms: float
    steps: Dict[str, StepRecord]
    messages_count: int
    tokens_exchanged: int
    bytes_exchanged: int
    critical_path_ms: float
    total_idle_wait_ms: float
    framework: str = "raw"
    selector_overhead_ms: float = 0.0
    turn_order: List[str] = field(default_factory=list)
    # v2 fields
    schema_version: int = 2
    dag_metrics: Optional[DagMetrics] = None
    role_token_stats: List[RoleTokenStats] = field(default_factory=list)
