"""Pure-Python DAG topology analysis utilities.

Computes depth, width, fan-out/fan-in, critical path, and parallel fraction
from a step dependency graph. No external dependencies.
"""

from collections import defaultdict, deque
from typing import Dict, List, Optional


def compute_dag_metrics(steps: Dict[str, Dict]) -> dict:
    """Compute DAG topology metrics from a step dependency map.

    Args:
        steps: Mapping of step_id -> {"deps": [step_id, ...], ...}.
            Only the "deps" key is required; other keys are ignored.

    Returns:
        dict with keys: depth, max_width, fanout_max, fanin_max,
        critical_path_steps, critical_path_len, parallel_fraction.
    """
    if not steps:
        return {
            "depth": 0,
            "max_width": 0,
            "fanout_max": 0,
            "fanin_max": 0,
            "critical_path_steps": [],
            "critical_path_len": 0,
            "parallel_fraction": 0.0,
        }

    total_steps = len(steps)

    # Build adjacency: children map and in-degree / deps
    children: Dict[str, List[str]] = defaultdict(list)
    deps_map: Dict[str, List[str]] = {}
    in_degree: Dict[str, int] = {}

    for sid, info in steps.items():
        dep_list = info.get("deps", [])
        deps_map[sid] = list(dep_list)
        in_degree[sid] = len(dep_list)
        for d in dep_list:
            children[d].append(sid)

    # BFS topological order → level assignment
    level: Dict[str, int] = {}
    queue: deque = deque()
    for sid in steps:
        if in_degree[sid] == 0:
            level[sid] = 0
            queue.append(sid)

    while queue:
        node = queue.popleft()
        for child in children[node]:
            candidate = level[node] + 1
            level[child] = max(level.get(child, 0), candidate)
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    # Cycle detection: if BFS didn't assign levels to all nodes, there's a cycle
    if len(level) < total_steps:
        unvisited = set(steps.keys()) - set(level.keys())
        raise ValueError(
            f"Cycle detected in DAG: nodes not reachable via topological sort: "
            f"{sorted(unvisited)}"
        )

    # Depth and max_width
    if level:
        depth = max(level.values()) + 1
        width_per_level: Dict[int, int] = defaultdict(int)
        for lv in level.values():
            width_per_level[lv] += 1
        max_width = max(width_per_level.values())
    else:
        depth = 0
        max_width = 0

    # Fan-out max (max number of children)
    fanout_max = max((len(children[sid]) for sid in steps), default=0)

    # Fan-in max (max number of deps)
    fanin_max = max((len(deps_map[sid]) for sid in steps), default=0)

    # Critical path via memoized DFS (longest path in DAG)
    cp_len_cache: Dict[str, int] = {}
    cp_next: Dict[str, Optional[str]] = {}

    def _cp_len(sid: str) -> int:
        if sid in cp_len_cache:
            return cp_len_cache[sid]
        best = 0
        best_child: Optional[str] = None
        for child in children[sid]:
            cl = _cp_len(child)
            if cl > best:
                best = cl
                best_child = child
        cp_len_cache[sid] = 1 + best
        cp_next[sid] = best_child
        return cp_len_cache[sid]

    for sid in steps:
        _cp_len(sid)

    # Find the root of the critical path
    if cp_len_cache:
        cp_start = max(cp_len_cache, key=cp_len_cache.get)  # type: ignore[arg-type]
        critical_path_len = cp_len_cache[cp_start]
        critical_path_steps: List[str] = []
        node: Optional[str] = cp_start
        while node is not None:
            critical_path_steps.append(node)
            node = cp_next.get(node)
    else:
        critical_path_len = 0
        critical_path_steps = []

    # Parallel fraction: 1 - (critical_path_len / total_steps)
    if total_steps > 0:
        parallel_fraction = 1.0 - (critical_path_len / total_steps)
    else:
        parallel_fraction = 0.0

    return {
        "depth": depth,
        "max_width": max_width,
        "fanout_max": fanout_max,
        "fanin_max": fanin_max,
        "critical_path_steps": critical_path_steps,
        "critical_path_len": critical_path_len,
        "parallel_fraction": parallel_fraction,
    }


def compute_role_token_stats(steps: Dict[str, Dict]) -> list:
    """Compute per-role token statistics from step records.

    Args:
        steps: Mapping of step_id -> step dict with agent_role,
            prompt_tokens, completion_tokens fields.

    Returns:
        List of dicts with role, count, prompt_mean, prompt_std,
        output_mean, output_std.
    """
    import math

    role_data: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"prompt": [], "output": []}
    )
    for info in steps.values():
        role = info.get("agent_role", "unknown")
        role_data[role]["prompt"].append(float(info.get("prompt_tokens", 0)))
        role_data[role]["output"].append(float(info.get("completion_tokens", 0)))

    result = []
    for role, data in sorted(role_data.items()):
        n = len(data["prompt"])
        p_mean = sum(data["prompt"]) / n if n else 0.0
        o_mean = sum(data["output"]) / n if n else 0.0
        p_std = math.sqrt(sum((x - p_mean) ** 2 for x in data["prompt"]) / n) if n > 1 else 0.0
        o_std = math.sqrt(sum((x - o_mean) ** 2 for x in data["output"]) / n) if n > 1 else 0.0
        result.append({
            "role": role,
            "count": n,
            "prompt_mean": round(p_mean, 1),
            "prompt_std": round(p_std, 1),
            "output_mean": round(o_mean, 1),
            "output_std": round(o_std, 1),
        })
    return result
