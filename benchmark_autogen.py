#!/usr/bin/env python3
"""
Multi-agent benchmark runner (ShareGPT/ShareGPT4V prompt-only) with:
- End-to-End Task Completion Time (makespan): P50/P95/P99 + mean
- Critical Path Latency (DAG longest path by step durations)
- Messages / Agent Interactions per Task
- Tokens exchanged per task (prompt + completion) + bytes exchanged
- Idle / Synchronization time (ready->start wait per step; barrier waits)
- Per-agent latency / TTFT / TPOT distributions
- Task throughput (tasks/sec), agent-step throughput (steps/sec)
- Tail & fairness stats (P95/P99 makespan, per-agent variance)

This script DOES NOT use any fallback prompt. Dataset is REQUIRED.
It uses OpenAI-compatible vLLM endpoint: http://localhost:8000/v1

Example:
  python bench_multiagent.py --model Qwen/Qwen2.5-7B-Instruct \
    --dataset_path /path/to/sharegpt4v.json \
    --tasks 128 --concurrency 32 --agents 3 --executors 2
"""

import asyncio
import time
import argparse
import os
import json
import random
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import httpx


# ----------------------------
# Dataset: ShareGPT / ShareGPT4V (text prompts only)
# ----------------------------

def load_sharegpt_dataset(dataset_path: str) -> List[str]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts: List[str] = []
    for item in data:
        if "conversations" in item:
            for msg in item["conversations"]:
                if msg.get("from") == "human" or msg.get("role") == "user":
                    prompts.append(msg.get("value", "") or "")
                    break
        elif "messages" in item:
            for msg in item["messages"]:
                if msg.get("role") == "user":
                    prompts.append(msg.get("content", "") or "")
                    break

    # hard filter empties (still dataset-only; we just skip invalid samples)
    prompts = [p.strip() for p in prompts if isinstance(p, str) and p.strip()]
    return prompts


def get_prompt_from_dataset(dataset: List[str], task_id: int) -> str:
    if not dataset:
        raise ValueError("Dataset is empty. ShareGPT4V prompts are required.")
    return dataset[task_id % len(dataset)]


# ----------------------------
# OpenAI-compatible async client (streaming for TTFT/TPOT)
# ----------------------------

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


class OpenAIStreamingClient:
    def __init__(self, base_url: str, api_key: str, timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = httpx.Timeout(timeout_s)

    async def chat_completions_stream(
        self,
        client: httpx.AsyncClient,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> LLMCallMetrics:
        """
        Uses /chat/completions streaming to estimate:
        - TTFT: first streamed token delta
        - TPOT: average inter-token time (from 1st token to end / (tokens-1))
        Also tries to read usage if server provides it via stream_options.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            # vLLM supports this; if unsupported, you just won't get usage
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        start_ts = time.time()
        first_token_ts: Optional[float] = None
        end_ts: float = start_ts
        token_timestamps: List[float] = []
        out_chunks: List[str] = []

        prompt_tokens = None
        completion_tokens = None
        total_tokens = None

        try:
            async with client.stream("POST", url, headers=headers, json=payload, timeout=self.timeout) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[len("data: "):].strip()
                    else:
                        continue

                    if data == "[DONE]":
                        break

                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    # streaming delta tokens
                    choices = obj.get("choices", [])
                    delta = (
                        choices[0].get("delta", {}).get("content", None)
                        if choices
                        else None
                    )
                    if delta:
                        now = time.time()
                        if first_token_ts is None:
                            first_token_ts = now
                        token_timestamps.append(now)
                        out_chunks.append(delta)

                    # usage may appear in streamed chunks (vLLM include_usage)
                    usage = obj.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        completion_tokens = usage.get("completion_tokens", completion_tokens)
                        total_tokens = usage.get("total_tokens", total_tokens)

            end_ts = time.time()
            out_text = "".join(out_chunks)

            # If usage missing, estimate tokens very roughly (fallback estimator)
            # NOTE: still dataset-only; token estimation isn't a prompt fallback.
            if completion_tokens is None:
                # ~4 chars/token heuristic (very rough)
                completion_tokens = max(1, len(out_text) // 4) if out_text else 0
            if prompt_tokens is None:
                # heuristic on prompt+system sizes
                joined = " ".join(m.get("content", "") for m in messages if "content" in m)
                prompt_tokens = max(1, len(joined) // 4) if joined else 0
            if total_tokens is None:
                total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

            ttft_ms = (first_token_ts - start_ts) * 1000 if first_token_ts else None

            # TPOT: average time per output token after first token
            tpot_ms = None
            if first_token_ts and completion_tokens and completion_tokens > 1:
                tpot_ms = ((end_ts - first_token_ts) * 1000) / (completion_tokens - 1)

            return LLMCallMetrics(
                ok=True,
                start_ts=start_ts,
                first_token_ts=first_token_ts,
                end_ts=end_ts,
                ttft_ms=ttft_ms,
                tpot_ms=tpot_ms,
                out_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
                out_text=out_text,
                error=None,
            )
        except Exception as e:
            end_ts = time.time()
            partial_text = "".join(out_chunks)
            return LLMCallMetrics(
                ok=False,
                start_ts=start_ts,
                first_token_ts=first_token_ts,
                end_ts=end_ts,
                ttft_ms=(first_token_ts - start_ts) * 1000 if first_token_ts else None,
                tpot_ms=None,
                out_tokens=None,
                prompt_tokens=None,
                total_tokens=None,
                out_text=partial_text,
                error=str(e),
            )


# ----------------------------
# Multi-agent workflow + tracing
# ----------------------------

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


def build_workflow(agent_count: int, executors: int) -> List[StepSpec]:
    """
    Simple (but realistic) multi-agent DAG:
      planner -> parallel executors -> aggregator

    Step IDs:
      P  (planner)
      E0..E{k-1}  (executors)
      A  (aggregator)

    You can expand this later (debate/critic/tool) by adding nodes & deps.
    """
    specs: List[StepSpec] = []
    specs.append(StepSpec(step_id="P", agent_role="planner", deps=[]))
    for i in range(executors):
        specs.append(StepSpec(step_id=f"E{i}", agent_role="executor", deps=["P"]))
    specs.append(StepSpec(step_id="A", agent_role="aggregator", deps=[f"E{i}" for i in range(executors)]))
    return specs


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    idx = int(len(s) * p / 100)
    idx = min(idx, len(s) - 1)
    return s[idx]


def critical_path_dag_ms(step_durations_ms: Dict[str, float], deps: Dict[str, List[str]]) -> float:
    """
    Longest path in DAG using step durations as weights.
    """
    # Topo by repeated relaxation (DAG assumed small)
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


# ----------------------------
# Task runner
# ----------------------------

async def run_task(
    task_id: int,
    prompt: str,
    model: str,
    llm: OpenAIStreamingClient,
    http_client: httpx.AsyncClient,
    llm_semaphore: asyncio.Semaphore,
    workflow: List[StepSpec],
) -> TaskRecord:
    task_start = time.time()

    # Step scheduling bookkeeping
    specs_by_id = {s.step_id: s for s in workflow}
    done: Dict[str, StepRecord] = {}
    deps = {s.step_id: list(s.deps) for s in workflow}

    # For idle/wait: ready_ts is when all deps are done
    # For the initial step(s), ready_ts = task_start
    ready_ts: Dict[str, float] = {}
    for s in workflow:
        ready_ts[s.step_id] = task_start if not s.deps else 0.0

    # Simple shared “conversation state”
    # (You can replace with richer memory / tool outputs.)
    state: Dict[str, Any] = {"user_prompt": prompt, "planner": "", "executions": {}}

    async def run_one_step(step: StepSpec) -> StepRecord:
        # Wait until deps finished (synchronization barrier per step)
        while True:
            if all(d in done for d in step.deps):
                break
            await asyncio.sleep(0.001)

        # Mark ready time
        rts = time.time()
        # If deps exist, ready time is when the LAST dependency finished
        if step.deps:
            rts = max(done[d].end_ts for d in step.deps)
        else:
            rts = task_start
        ready_ts[step.step_id] = rts

        # Compose step prompt (agent role differences)
        if step.agent_role == "planner":
            content = (
                "You are the Planner agent. Produce a short plan with numbered steps.\n\n"
                f"User request:\n{state['user_prompt']}\n"
            )
        elif step.agent_role == "executor":
            planner = state.get("planner", "")
            content = (
                "You are an Executor agent. Execute one part of the plan. Be concise but correct.\n\n"
                f"Plan:\n{planner}\n\n"
                f"User request:\n{state['user_prompt']}\n\n"
                f"Executor focus: {step.step_id}\n"
            )
        elif step.agent_role == "aggregator":
            planner = state.get("planner", "")
            execs = "\n\n".join([f"{k}: {v}" for k, v in state["executions"].items()])
            content = (
                "You are the Aggregator agent. Combine planner + executors into one final answer.\n\n"
                f"User request:\n{state['user_prompt']}\n\n"
                f"Plan:\n{planner}\n\n"
                f"Executor outputs:\n{execs}\n\n"
                "Return the final response."
            )
        else:
            content = f"Role={step.agent_role}\n\nUser:\n{state['user_prompt']}\n"

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ]

        # Count bytes in (roughly)
        bytes_in = sum(
            len((m.get("content", "") or "").encode("utf-8")) for m in messages
        )

        # Concurrency control at LLM level
        async with llm_semaphore:
            start_ts = time.time()
            wait_ms = (start_ts - rts) * 1000

            call = await llm.chat_completions_stream(
                client=http_client,
                model=model,
                messages=messages,
                temperature=0.2,
            )
            end_ts = call.end_ts

        bytes_out = len(call.out_text.encode("utf-8"))

        rec = StepRecord(
            step_id=step.step_id,
            agent_role=step.agent_role,
            deps=list(step.deps),
            ready_ts=rts,
            start_ts=start_ts,
            end_ts=end_ts,
            wait_ms=max(0.0, wait_ms),
            latency_ms=(end_ts - start_ts) * 1000,
            ttft_ms=call.ttft_ms,
            tpot_ms=call.tpot_ms,
            prompt_tokens=int(call.prompt_tokens or 0),
            completion_tokens=int(call.out_tokens or 0),
            total_tokens=int(call.total_tokens or 0),
            bytes_in=bytes_in,
            bytes_out=bytes_out,
            ok=call.ok,
            error=call.error,
        )

        # Update shared state
        if call.ok:
            if step.agent_role == "planner":
                state["planner"] = call.out_text
            elif step.agent_role == "executor":
                state["executions"][step.step_id] = call.out_text
            elif step.agent_role == "aggregator":
                state["final"] = call.out_text

        return rec

    # Launch steps when their deps are satisfied; allow parallel executors
    # We’ll create tasks for all steps, but each waits for deps internally.
    step_tasks = {s.step_id: asyncio.create_task(run_one_step(s)) for s in workflow}

    # Collect in completion order
    for sid, t in step_tasks.items():
        rec = await t
        done[sid] = rec

    task_end = time.time()
    makespan_ms = (task_end - task_start) * 1000

    # Derive metrics
    messages_count = len(workflow)  # each step is one LLM message exchange here
    tokens_exchanged = sum(r.total_tokens for r in done.values())
    bytes_exchanged = sum(r.bytes_in + r.bytes_out for r in done.values())
    total_idle_wait_ms = sum(r.wait_ms for r in done.values())

    step_durations = {sid: r.latency_ms for sid, r in done.items()}
    deps_map = {sid: r.deps for sid, r in done.items()}
    cp_ms = critical_path_dag_ms(step_durations, deps_map)

    return TaskRecord(
        task_id=task_id,
        task_start_ts=task_start,
        task_end_ts=task_end,
        makespan_ms=makespan_ms,
        steps=done,
        messages_count=messages_count,
        tokens_exchanged=tokens_exchanged,
        bytes_exchanged=bytes_exchanged,
        critical_path_ms=cp_ms,
        total_idle_wait_ms=total_idle_wait_ms,
    )


# ----------------------------
# Main + aggregation
# ----------------------------

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--dataset_path", type=str, required=True, help="REQUIRED: ShareGPT/ShareGPT4V JSON")
    ap.add_argument("--tasks", type=int, default=64, help="Number of tasks (dataset prompts)")
    ap.add_argument("--concurrency", type=int, default=32, help="LLM call concurrency (global)")
    ap.add_argument("--executors", type=int, default=2, help="How many executor agents (parallel)")
    ap.add_argument("--output_dir", type=str, default="results_multiagent")
    ap.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    ap.add_argument("--api_key", type=str, default="vllm-key")
    args = ap.parse_args()

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")

    dataset = load_sharegpt_dataset(args.dataset_path)
    if not dataset:
        raise ValueError(f"Loaded dataset is empty after filtering: {args.dataset_path}")

    random.shuffle(dataset)

    os.makedirs(args.output_dir, exist_ok=True)

    workflow = build_workflow(agent_count=2 + args.executors, executors=args.executors)

    llm = OpenAIStreamingClient(base_url=args.base_url, api_key=args.api_key)
    llm_semaphore = asyncio.Semaphore(args.concurrency)

    # Warmup MUST be dataset-only
    warmup_prompt = get_prompt_from_dataset(dataset, 0)

    async with httpx.AsyncClient() as http_client:
        print(f"[warmup] running 1 task (dataset-only)")
        _ = await run_task(
            task_id=-1,
            prompt=warmup_prompt,
            model=args.model,
            llm=llm,
            http_client=http_client,
            llm_semaphore=asyncio.Semaphore(1),
            workflow=workflow,
        )
        await asyncio.sleep(1)

        print(f"[run] tasks={args.tasks}  llm_concurrency={args.concurrency}  executors={args.executors}")
        t0 = time.time()

        # Run tasks concurrently (task-level); LLM-level concurrency is globally bounded by llm_semaphore
        task_futs = []
        for i in range(args.tasks):
            prompt = get_prompt_from_dataset(dataset, i)  # dataset-only
            task_futs.append(
                asyncio.create_task(
                    run_task(
                        task_id=i,
                        prompt=prompt,
                        model=args.model,
                        llm=llm,
                        http_client=http_client,
                        llm_semaphore=llm_semaphore,
                        workflow=workflow,
                    )
                )
            )

        records: List[TaskRecord] = await asyncio.gather(*task_futs)
        t1 = time.time()

    total_s = t1 - t0
    task_throughput = len(records) / total_s if total_s > 0 else 0.0

    makespans = [r.makespan_ms for r in records]
    cps = [r.critical_path_ms for r in records]
    msg_counts = [r.messages_count for r in records]
    idle_waits = [r.total_idle_wait_ms for r in records]
    toks = [r.tokens_exchanged for r in records]
    bytes_x = [r.bytes_exchanged for r in records]

    # Per-agent aggregates
    per_role_lat: Dict[str, List[float]] = {}
    per_role_ttft: Dict[str, List[float]] = {}
    per_role_tpot: Dict[str, List[float]] = {}
    per_role_wait: Dict[str, List[float]] = {}

    total_steps = 0
    for tr in records:
        for s in tr.steps.values():
            total_steps += 1
            per_role_lat.setdefault(s.agent_role, []).append(s.latency_ms)
            per_role_wait.setdefault(s.agent_role, []).append(s.wait_ms)
            if s.ttft_ms is not None:
                per_role_ttft.setdefault(s.agent_role, []).append(s.ttft_ms)
            if s.tpot_ms is not None:
                per_role_tpot.setdefault(s.agent_role, []).append(s.tpot_ms)

    step_throughput = total_steps / total_s if total_s > 0 else 0.0

    def summarize(name: str, vals: List[float]) -> None:
        print(f"{name}: mean={statistics.mean(vals):.2f}  p50={percentile(vals,50):.2f}  p95={percentile(vals,95):.2f}  p99={percentile(vals,99):.2f}")

    print("\n==== Multi-Agent Benchmark Results ====")
    print(f"Tasks: {len(records)}  Total time: {total_s:.2f}s  Task throughput: {task_throughput:.2f} tasks/s")
    print(f"Total agent steps: {total_steps}  Step throughput: {step_throughput:.2f} steps/s\n")

    summarize("E2E makespan (ms)", makespans)
    summarize("Critical path (ms)", cps)
    summarize("Total idle/sync wait per task (ms)", idle_waits)
    summarize("Messages per task", [float(x) for x in msg_counts])
    summarize("Tokens exchanged per task", [float(x) for x in toks])
    summarize("Bytes exchanged per task", [float(x) for x in bytes_x])

    print("\n-- Per-agent-role breakdown --")
    for role in sorted(per_role_lat.keys()):
        summarize(f"[{role}] latency (ms)", per_role_lat[role])
        if role in per_role_wait:
            summarize(f"[{role}] wait/idle (ms)", per_role_wait[role])
        if role in per_role_ttft and per_role_ttft[role]:
            summarize(f"[{role}] TTFT (ms)", per_role_ttft[role])
        if role in per_role_tpot and per_role_tpot[role]:
            summarize(f"[{role}] TPOT (ms)", per_role_tpot[role])
        print()

    # Write a JSONL trace for offline critical-path & visualization
    out_path = os.path.join(args.output_dir, f"trace_tasks_{args.tasks}_exec{args.executors}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for tr in records:
            row = {
                "task_id": tr.task_id,
                "task_start_ts": tr.task_start_ts,
                "task_end_ts": tr.task_end_ts,
                "makespan_ms": tr.makespan_ms,
                "critical_path_ms": tr.critical_path_ms,
                "messages_count": tr.messages_count,
                "tokens_exchanged": tr.tokens_exchanged,
                "bytes_exchanged": tr.bytes_exchanged,
                "total_idle_wait_ms": tr.total_idle_wait_ms,
                "steps": {
                    sid: {
                        "agent_role": s.agent_role,
                        "deps": s.deps,
                        "ready_ts": s.ready_ts,
                        "start_ts": s.start_ts,
                        "end_ts": s.end_ts,
                        "wait_ms": s.wait_ms,
                        "latency_ms": s.latency_ms,
                        "ttft_ms": s.ttft_ms,
                        "tpot_ms": s.tpot_ms,
                        "prompt_tokens": s.prompt_tokens,
                        "completion_tokens": s.completion_tokens,
                        "total_tokens": s.total_tokens,
                        "bytes_in": s.bytes_in,
                        "bytes_out": s.bytes_out,
                        "ok": s.ok,
                        "error": s.error,
                    }
                    for sid, s in tr.steps.items()
                },
            }
            f.write(json.dumps(row) + "\n")

    print(f"\nSaved task traces to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
