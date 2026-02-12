"""Aggregation helpers: percentile, critical path DAG computation."""

from typing import Dict, List, Optional


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    idx = int(len(s) * p / 100)
    idx = min(idx, len(s) - 1)
    return s[idx]


def critical_path_dag_ms(step_durations_ms: Dict[str, float], deps: Dict[str, List[str]]) -> float:
    """Longest path in DAG using step durations as weights."""
    nodes = list(step_durations_ms.keys())
    best: Dict[str, float] = {n: step_durations_ms[n] for n in nodes}

    changed = True
    while changed:
        changed = False
        for n in nodes:
            if deps.get(n):
                cand = step_durations_ms[n] + max(best[d] for d in deps[n])
                if cand > best[n] + 1e-9:
                    best[n] = cand
                    changed = True
    return max(best.values()) if best else 0.0
