# Trace Schema Contract (v2)

Canonical definition of the v2 trace schema shared between **multi-agent-bench** (producer) and **gpusim** (consumer).

Both repos MUST keep their implementations consistent with this contract.

---

## Task-level Fields

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `task_id` | int | REQUIRED | Unique task identifier |
| `schema_version` | int | REQUIRED | v2 = `2` |
| `makespan_ms` | float | REQUIRED | Wall-clock task duration |
| `steps` | dict[str, StepRecord] | REQUIRED | Keyed by step instance ID |
| `critical_path_ms` | float | RECOMMENDED | DAG critical path duration |
| `dag_metrics` | object | RECOMMENDED | See DagMetrics below |
| `role_token_stats` | list[RoleTokenStats] | OPTIONAL | Per-role token aggregates |

## Step-level Fields (StepRecord)

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `agent_role` | str | REQUIRED | `"planner"`, `"executor"`, `"aggregator"` |
| `deps` | list[str] | REQUIRED | Dependency step instance IDs |
| `latency_ms` | float | REQUIRED | End-to-end step latency (for `status="ok"` steps) |
| `prompt_tokens` | int | REQUIRED | Input token count |
| `completion_tokens` | int | REQUIRED | Output token count |
| `start_ns` / `end_ns` | int | RECOMMENDED | `time.monotonic_ns()` timestamps |
| `first_token_ns` | int | OPTIONAL | First token monotonic timestamp |
| `status` | str | REQUIRED | `"ok"` (default) or `"error"` |

## Step Instance ID Convention

Step instance IDs uniquely identify each step invocation within a task.

- **Base IDs**: `P` (planner), `E0`/`E1`/... (executors), `A` (aggregator)
- **First invocation**: bare base ID (e.g., `P`, `E0`, `A`)
- **Repeated invocations**: `{base}#{count}` (e.g., `P#1`, `E0#2`, `A#1`)
- The `#` separator distinguishes instance suffixes from the executor index digit.

Examples:
```
P       — first planner call
P#1     — second planner call
E0      — first call to executor 0
E0#1    — second call to executor 0
A       — first aggregator call
```

## v1 Backward Compatibility

- Missing `schema_version` implies v1.
- v1 `ok=false` maps to `status="error"` on ingest.
- Missing `*_ns` fields default to `None`.
- v1 step IDs may use `_` as separator (e.g., `P_1`); consumers MUST accept both `_` and `#`.

## Quality Gate Tiers

Three tiers classify trace data quality. See `benchmark/core/quality_tiers.py` for canonical constants.

| Tier | step_ok_rate | task_ok_rate | Additional |
|------|-------------|-------------|------------|
| **VALIDATED** | 100% | 100% | All steps have ttft, tpot, completion > 0 |
| **USABLE** | >= 95% | >= 90% | — |
| **EXPLORATORY** | < 95% | < 90% | — |

**MAPE aggregation policy**: When computing sim-vs-real MAPE, use only data points at USABLE tier or above. EXPLORATORY points may be included for directional analysis but MUST be flagged.

## DagMetrics Schema

| Field | Type | Notes |
|-------|------|-------|
| `depth` | int | Longest path in DAG (edges) |
| `max_width` | int | Maximum parallel steps at any level |
| `fanout_max` | int | Maximum out-degree of any step |
| `fanin_max` | int | Maximum in-degree of any step |
| `critical_path_steps` | list[str] | Step IDs on the critical path |
| `critical_path_len` | int | Number of steps on critical path |
| `parallel_fraction` | float | Fraction of steps that run in parallel |
