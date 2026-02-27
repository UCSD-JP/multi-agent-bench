#!/usr/bin/env python3
"""Smoke test for memory-profile modules.

Tests workload generation and CSV format without requiring a GPU or vLLM server.
Run: python scripts/memory_profile/smoke_test.py
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from workload_gen import generate_workload, make_prompt
from gpu_mem_sampler import GpuMemSampler
from request_runner import AsyncRequestRunner


def test_workload_gen():
    """Test workload generation with small grid."""
    print("[1/4] Workload generation...", end=" ")
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = Path(f.name)

    n = generate_workload(
        path,
        input_lens=[512, 1024],
        output_lens=[32],
        concurrencies=[4, 8],
        prefix_modes=["low", "high"],
        n_per_condition=1,
    )

    # Expected: 2 × 1 × 2 × 2 × 1 = 8
    assert n == 8, f"Expected 8, got {n}"

    # Validate JSONL
    with open(path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 8

    for rec in lines:
        assert "req_id" in rec
        assert "prompt" in rec
        assert "input_len" in rec
        assert "output_len" in rec
        assert "concurrency_group" in rec
        assert len(rec["prompt"]) > 100  # Non-trivial prompt

    # Check concurrency groups
    conc_groups = set(r["concurrency_group"] for r in lines)
    assert conc_groups == {4, 8}, f"Expected {{4, 8}}, got {conc_groups}"

    path.unlink()
    print("OK (8 requests, JSONL valid)")


def test_prompt_construction():
    """Test prompt length targeting and prefix reuse."""
    print("[2/4] Prompt construction...", end=" ")
    cache = {}

    # Low prefix reuse: unique prompts
    p1 = make_prompt(1024, "low", "req_001", cache)
    p2 = make_prompt(1024, "low", "req_002", cache)
    assert p1 != p2, "Low-reuse prompts should be unique"

    # High prefix reuse: shared prefix
    p3 = make_prompt(1024, "high", "req_003", cache)
    p4 = make_prompt(1024, "high", "req_004", cache)
    # First 80% should be identical (shared prefix)
    prefix_len = int(len(p3) * 0.6)  # Conservative check
    assert p3[:prefix_len] == p4[:prefix_len], "High-reuse should share prefix"
    assert p3 != p4, "Full prompts should still differ"

    # Target length roughly correct (8.2 chars/token ± 50%)
    target_chars = int(1024 * 8.2)
    assert len(p1) > target_chars * 0.5, f"Prompt too short: {len(p1)}"
    assert len(p1) < target_chars * 1.5, f"Prompt too long: {len(p1)}"

    print("OK (unique low, shared high, length targets)")


def test_sampler_init():
    """Test sampler can be instantiated (no GPU needed)."""
    print("[3/4] Sampler instantiation...", end=" ")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = Path(f.name)

    sampler = GpuMemSampler(path, interval_ms=200, gpu_indices=[0, 1])
    assert not sampler.is_running
    assert sampler.CSV_HEADER == [
        "timestamp", "gpu_idx", "used_mib", "total_mib", "util_pct"
    ]

    path.unlink(missing_ok=True)
    print("OK (init without GPU)")


def test_runner_init():
    """Test runner can be instantiated."""
    print("[4/4] Runner instantiation...", end=" ")
    runner = AsyncRequestRunner(
        base_url="http://127.0.0.1:8000/v1",
        model="test-model",
        timeout_s=60,
    )
    assert runner.base_url == "http://127.0.0.1:8000/v1"
    assert runner.model == "test-model"
    assert runner.CSV_HEADER[0] == "req_id"

    print("OK (init without server)")


def main():
    print("=== Memory Profile Smoke Test ===\n")
    failures = 0

    for test_fn in [
        test_workload_gen,
        test_prompt_construction,
        test_sampler_init,
        test_runner_init,
    ]:
        try:
            test_fn()
        except Exception as e:
            print(f"FAIL: {e}")
            failures += 1

    print(f"\n{'='*40}")
    if failures == 0:
        print("All tests passed!")
        return 0
    else:
        print(f"{failures} test(s) failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
