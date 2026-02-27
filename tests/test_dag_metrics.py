"""Tests for DAG topology metrics computation."""

import pytest

from benchmark.core.dag_metrics import compute_dag_metrics, compute_role_token_stats


class TestComputeDagMetrics:
    """Test compute_dag_metrics with various DAG topologies."""

    def test_empty_graph(self):
        result = compute_dag_metrics({})
        assert result["depth"] == 0
        assert result["max_width"] == 0
        assert result["fanout_max"] == 0
        assert result["fanin_max"] == 0
        assert result["critical_path_steps"] == []
        assert result["critical_path_len"] == 0
        assert result["parallel_fraction"] == 0.0

    def test_single_node(self):
        steps = {"P": {"deps": []}}
        result = compute_dag_metrics(steps)
        assert result["depth"] == 1
        assert result["max_width"] == 1
        assert result["fanout_max"] == 0
        assert result["fanin_max"] == 0
        assert result["critical_path_steps"] == ["P"]
        assert result["critical_path_len"] == 1
        assert result["parallel_fraction"] == 0.0

    def test_sequential_chain(self):
        """P → E → A: depth=3, max_width=1, parallel_fraction=0."""
        steps = {
            "P": {"deps": []},
            "E": {"deps": ["P"]},
            "A": {"deps": ["E"]},
        }
        result = compute_dag_metrics(steps)
        assert result["depth"] == 3
        assert result["max_width"] == 1
        assert result["fanout_max"] == 1
        assert result["fanin_max"] == 1
        assert result["critical_path_len"] == 3
        assert result["parallel_fraction"] == pytest.approx(0.0)

    def test_diamond(self):
        """P → {E0, E1} → A: depth=3, max_width=2."""
        steps = {
            "P": {"deps": []},
            "E0": {"deps": ["P"]},
            "E1": {"deps": ["P"]},
            "A": {"deps": ["E0", "E1"]},
        }
        result = compute_dag_metrics(steps)
        assert result["depth"] == 3
        assert result["max_width"] == 2
        assert result["fanout_max"] == 2
        assert result["fanin_max"] == 2
        assert result["critical_path_len"] == 3
        # parallel_fraction = 1 - 3/4 = 0.25
        assert result["parallel_fraction"] == pytest.approx(0.25)

    def test_wide_fanout(self):
        """P → {E0..E7} → A: max_width=8, fanout_max=8."""
        steps = {"P": {"deps": []}}
        executor_ids = [f"E{i}" for i in range(8)]
        for eid in executor_ids:
            steps[eid] = {"deps": ["P"]}
        steps["A"] = {"deps": executor_ids}
        result = compute_dag_metrics(steps)
        assert result["depth"] == 3
        assert result["max_width"] == 8
        assert result["fanout_max"] == 8
        assert result["fanin_max"] == 8
        assert result["critical_path_len"] == 3
        # parallel_fraction = 1 - 3/10 = 0.7
        assert result["parallel_fraction"] == pytest.approx(0.7)

    def test_two_independent_chains(self):
        """A→B and C→D: depth=2, max_width=2, two disconnected chains."""
        steps = {
            "A": {"deps": []},
            "B": {"deps": ["A"]},
            "C": {"deps": []},
            "D": {"deps": ["C"]},
        }
        result = compute_dag_metrics(steps)
        assert result["depth"] == 2
        assert result["max_width"] == 2
        assert result["critical_path_len"] == 2
        assert result["parallel_fraction"] == pytest.approx(0.5)

    def test_critical_path_steps_ordering(self):
        """P → E0 → A should have critical path [P, E0, A]."""
        steps = {
            "P": {"deps": []},
            "E0": {"deps": ["P"]},
            "A": {"deps": ["E0"]},
        }
        result = compute_dag_metrics(steps)
        assert result["critical_path_steps"] == ["P", "E0", "A"]

    def test_cycle_raises_value_error(self):
        """Cyclic deps should raise ValueError, not infinite recurse."""
        steps = {
            "A": {"deps": ["C"]},
            "B": {"deps": ["A"]},
            "C": {"deps": ["B"]},
        }
        with pytest.raises(ValueError, match="Cycle detected"):
            compute_dag_metrics(steps)

    def test_self_cycle_raises(self):
        """Self-referencing dep should raise ValueError."""
        steps = {"A": {"deps": ["A"]}}
        with pytest.raises(ValueError, match="Cycle detected"):
            compute_dag_metrics(steps)


class TestComputeRoleTokenStats:
    """Test compute_role_token_stats."""

    def test_empty(self):
        assert compute_role_token_stats({}) == []

    def test_single_role(self):
        steps = {
            "P": {"agent_role": "planner", "prompt_tokens": 100, "completion_tokens": 50},
        }
        result = compute_role_token_stats(steps)
        assert len(result) == 1
        assert result[0]["role"] == "planner"
        assert result[0]["count"] == 1
        assert result[0]["prompt_mean"] == 100.0
        assert result[0]["output_mean"] == 50.0

    def test_multiple_roles(self):
        steps = {
            "P": {"agent_role": "planner", "prompt_tokens": 100, "completion_tokens": 50},
            "E0": {"agent_role": "executor", "prompt_tokens": 200, "completion_tokens": 80},
            "E1": {"agent_role": "executor", "prompt_tokens": 300, "completion_tokens": 120},
            "A": {"agent_role": "aggregator", "prompt_tokens": 400, "completion_tokens": 60},
        }
        result = compute_role_token_stats(steps)
        assert len(result) == 3
        roles = {r["role"] for r in result}
        assert roles == {"planner", "executor", "aggregator"}

        # Check executor stats (mean of 200 and 300)
        executor_stat = [r for r in result if r["role"] == "executor"][0]
        assert executor_stat["count"] == 2
        assert executor_stat["prompt_mean"] == 250.0
        assert executor_stat["output_mean"] == 100.0
