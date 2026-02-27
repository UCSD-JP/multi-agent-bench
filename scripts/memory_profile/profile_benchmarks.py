#!/usr/bin/env python3
"""Profile token usage of 3 benchmarks: LiveCodeBench, SWE-bench Verified, Terminal Bench 2.

Outputs per-benchmark CSV with per-example token counts + summary statistics.
Uses tiktoken cl100k_base as reference tokenizer (GPT-4 family).
"""

import csv
import json
import statistics
import sys
from pathlib import Path

import tiktoken

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(enc.encode(text))


def pct(data, p):
    """Percentile helper."""
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


def print_stats(name: str, values: list[int]):
    if not values:
        print(f"  {name}: NO DATA")
        return
    print(f"  {name}:")
    print(f"    count  = {len(values)}")
    print(f"    min    = {min(values):,}")
    print(f"    P5     = {int(pct(values, 5)):,}")
    print(f"    P25    = {int(pct(values, 25)):,}")
    print(f"    median = {int(pct(values, 50)):,}")
    print(f"    mean   = {int(statistics.mean(values)):,}")
    print(f"    P75    = {int(pct(values, 75)):,}")
    print(f"    P95    = {int(pct(values, 95)):,}")
    print(f"    max    = {max(values):,}")
    print(f"    total  = {sum(values):,}")


# ---------------------------------------------------------------------------
# 1. LiveCodeBench v6
# ---------------------------------------------------------------------------
def profile_livecodebench(out_dir: Path):
    print("\n" + "=" * 60)
    print("1. LiveCodeBench — Code Generation")
    print("=" * 60)
    from datasets import load_dataset

    ds = load_dataset("livecodebench/code_generation_lite", split="test")
    print(f"  Loaded {len(ds)} problems")

    rows = []
    input_tokens = []
    # LiveCodeBench has question_content, starter_code, public_test_cases
    for i, ex in enumerate(ds):
        q = ex.get("question_content") or ""
        starter = ex.get("starter_code") or ""
        # Build a realistic prompt
        prompt_parts = []
        prompt_parts.append("You are an expert Python programmer. Solve the following problem.\n\n")
        prompt_parts.append(f"### Problem\n{q}\n\n")
        if starter:
            prompt_parts.append(f"### Starter Code\n```python\n{starter}\n```\n\n")
        prompt_parts.append("### Solution\n```python\n")

        full_prompt = "".join(prompt_parts)
        q_tokens = count_tokens(q)
        prompt_tokens = count_tokens(full_prompt)
        input_tokens.append(prompt_tokens)

        difficulty = ex.get("difficulty") or "unknown"
        platform = ex.get("platform") or "unknown"
        qid = ex.get("question_id") or str(i)

        rows.append({
            "benchmark": "LiveCodeBench",
            "task_id": qid,
            "difficulty": difficulty,
            "platform": platform,
            "question_tokens": q_tokens,
            "prompt_tokens": prompt_tokens,
            "starter_code_tokens": count_tokens(starter),
        })

        if (i + 1) % 200 == 0:
            print(f"    processed {i+1}/{len(ds)}...")

    # Write CSV
    csv_path = out_dir / "livecodebench_tokens.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print_stats("prompt_tokens (full input)", input_tokens)
    q_only = [r["question_tokens"] for r in rows]
    print_stats("question_tokens (problem only)", q_only)

    # By difficulty
    for diff in sorted(set(r["difficulty"] for r in rows)):
        subset = [r["prompt_tokens"] for r in rows if r["difficulty"] == diff]
        print_stats(f"prompt_tokens [{diff}]", subset)

    print(f"\n  CSV: {csv_path}")
    return rows


# ---------------------------------------------------------------------------
# 2. SWE-bench Verified
# ---------------------------------------------------------------------------
def profile_swebench(out_dir: Path):
    print("\n" + "=" * 60)
    print("2. SWE-bench Verified")
    print("=" * 60)
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    print(f"  Loaded {len(ds)} instances")

    rows = []
    issue_tokens = []
    patch_tokens = []
    hints_tokens = []

    for i, ex in enumerate(ds):
        instance_id = ex.get("instance_id", str(i))
        problem = ex.get("problem_statement") or ""
        patch = ex.get("patch") or ""
        hints = ex.get("hints_text") or ""

        p_tok = count_tokens(problem)
        pa_tok = count_tokens(patch)
        h_tok = count_tokens(hints)

        issue_tokens.append(p_tok)
        patch_tokens.append(pa_tok)
        if h_tok > 0:
            hints_tokens.append(h_tok)

        # Extract repo name
        repo = instance_id.rsplit("__", 1)[0] if "__" in instance_id else "unknown"

        rows.append({
            "benchmark": "SWE-bench",
            "instance_id": instance_id,
            "repo": repo,
            "issue_tokens": p_tok,
            "gold_patch_tokens": pa_tok,
            "hints_tokens": h_tok,
        })

        if (i + 1) % 100 == 0:
            print(f"    processed {i+1}/{len(ds)}...")

    csv_path = out_dir / "swebench_tokens.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print_stats("issue_tokens (problem_statement)", issue_tokens)
    print_stats("gold_patch_tokens (reference output)", patch_tokens)
    if hints_tokens:
        print_stats("hints_tokens (when available)", hints_tokens)

    # By repo
    repos = sorted(set(r["repo"] for r in rows))
    for repo in repos:
        subset = [r["issue_tokens"] for r in rows if r["repo"] == repo]
        if len(subset) >= 5:
            print_stats(f"issue_tokens [{repo}] (n={len(subset)})", subset)

    print(f"\n  CSV: {csv_path}")
    return rows


# ---------------------------------------------------------------------------
# 3. Terminal Bench 2
# ---------------------------------------------------------------------------
def profile_terminalbench(out_dir: Path):
    print("\n" + "=" * 60)
    print("3. Terminal Bench 2")
    print("=" * 60)
    from datasets import load_dataset

    # Try multiple dataset names
    ds = None
    for name in [
        "zai-org/terminal-bench-2-verified",
        "terminal-bench/terminal-bench-2",
        "laude-institute/terminal-bench-2",
    ]:
        try:
            ds = load_dataset(name, split="test")
            print(f"  Loaded from {name}: {len(ds)} tasks")
            break
        except Exception as e:
            try:
                ds = load_dataset(name)
                # Try to find the right split
                for split_name in ds:
                    ds = ds[split_name]
                    print(f"  Loaded from {name}/{split_name}: {len(ds)} tasks")
                    break
                break
            except Exception:
                print(f"  Failed to load {name}: {e}")
                continue

    if ds is None:
        print("  WARNING: Could not load Terminal Bench 2 dataset from HuggingFace.")
        print("  Trying GitHub-based approach...")
        return _profile_terminalbench_fallback(out_dir)

    rows = []
    instruction_tokens = []

    # Inspect columns
    print(f"  Columns: {ds.column_names}")

    for i, ex in enumerate(ds):
        task_id = ex.get("task_id") or ex.get("id") or ex.get("name") or str(i)
        instruction = ex.get("instruction") or ex.get("prompt") or ex.get("task") or ex.get("description") or ""
        category = ex.get("category") or ex.get("type") or "unknown"
        difficulty = ex.get("difficulty") or "unknown"

        inst_tok = count_tokens(instruction)
        instruction_tokens.append(inst_tok)

        rows.append({
            "benchmark": "TerminalBench2",
            "task_id": task_id,
            "category": category,
            "difficulty": difficulty,
            "instruction_tokens": inst_tok,
            "instruction_chars": len(instruction),
        })

    csv_path = out_dir / "terminalbench_tokens.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print_stats("instruction_tokens", instruction_tokens)

    # By category
    cats = sorted(set(r["category"] for r in rows))
    for cat in cats:
        subset = [r["instruction_tokens"] for r in rows if r["category"] == cat]
        if len(subset) >= 2:
            print_stats(f"instruction_tokens [{cat}]", subset)

    print(f"\n  CSV: {csv_path}")
    return rows


def _profile_terminalbench_fallback(out_dir: Path):
    """Fallback: try loading from GitHub or local."""
    print("  Terminal Bench 2 dataset not available on HuggingFace.")
    print("  Skipping — manual download required.")
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    out_dir = Path("results/memory_profile/benchmark_token_profiles")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Benchmark Token Profiler")
    print(f"Tokenizer: cl100k_base (GPT-4)")
    print(f"Output: {out_dir}/")

    lcb = profile_livecodebench(out_dir)
    swe = profile_swebench(out_dir)
    tb = profile_terminalbench(out_dir)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    summary = []
    if lcb:
        vals = [r["prompt_tokens"] for r in lcb]
        summary.append({
            "benchmark": "LiveCodeBench",
            "n_tasks": len(lcb),
            "input_min": min(vals),
            "input_p25": int(pct(vals, 25)),
            "input_median": int(pct(vals, 50)),
            "input_mean": int(statistics.mean(vals)),
            "input_p75": int(pct(vals, 75)),
            "input_p95": int(pct(vals, 95)),
            "input_max": max(vals),
            "output_est": "200-800 (code)",
        })
    if swe:
        vals = [r["issue_tokens"] for r in swe]
        pvals = [r["gold_patch_tokens"] for r in swe]
        summary.append({
            "benchmark": "SWE-bench Verified",
            "n_tasks": len(swe),
            "input_min": min(vals),
            "input_p25": int(pct(vals, 25)),
            "input_median": int(pct(vals, 50)),
            "input_mean": int(statistics.mean(vals)),
            "input_p75": int(pct(vals, 75)),
            "input_p95": int(pct(vals, 95)),
            "input_max": max(vals),
            "output_est": f"{int(pct(pvals, 50))} median patch",
        })
    if tb:
        vals = [r["instruction_tokens"] for r in tb]
        summary.append({
            "benchmark": "Terminal Bench 2",
            "n_tasks": len(tb),
            "input_min": min(vals),
            "input_p25": int(pct(vals, 25)),
            "input_median": int(pct(vals, 50)),
            "input_mean": int(statistics.mean(vals)),
            "input_p75": int(pct(vals, 75)),
            "input_p95": int(pct(vals, 95)),
            "input_max": max(vals),
            "output_est": "2K-10K (agentic)",
        })

    # Print summary table
    if summary:
        print(f"\n{'Benchmark':<22} {'N':>5} {'Min':>7} {'P25':>7} {'Median':>7} {'Mean':>7} {'P75':>7} {'P95':>7} {'Max':>7}  Output")
        print("-" * 105)
        for s in summary:
            print(f"{s['benchmark']:<22} {s['n_tasks']:>5} {s['input_min']:>7,} {s['input_p25']:>7,} {s['input_median']:>7,} {s['input_mean']:>7,} {s['input_p75']:>7,} {s['input_p95']:>7,} {s['input_max']:>7,}  {s['output_est']}")

    # Write summary JSON
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary JSON: {summary_path}")


if __name__ == "__main__":
    main()
