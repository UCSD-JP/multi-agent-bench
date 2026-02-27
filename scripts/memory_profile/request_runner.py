#!/usr/bin/env python3
"""Async request runner for memory-profile experiments.

Sends chat completion requests to an OpenAI-compatible endpoint (vLLM)
with controlled concurrency. Records per-request trace (start/end times,
status, token counts) but does NOT compute TTFT/TPOT/TPS — this mode
is purely for memory pressure characterization.

Output format (request_trace.csv):
  req_id,input_len,output_len,concurrency_group,start_ts,end_ts,status,prompt_tokens,completion_tokens
"""

import asyncio
import csv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx


@dataclass
class RequestSpec:
    """Single request specification from workload JSONL."""
    req_id: str
    prompt: str
    input_len: int
    output_len: int
    concurrency_group: int
    extra: dict = field(default_factory=dict)


@dataclass
class RequestResult:
    """Result of a single completed request."""
    req_id: str
    input_len: int
    output_len: int
    concurrency_group: int
    start_ts: str
    end_ts: str
    status: str  # "ok" or "error:<reason>"
    prompt_tokens: int
    completion_tokens: int


class AsyncRequestRunner:
    """Sends requests with controlled concurrency, writes trace CSV."""

    CSV_HEADER = [
        "req_id", "input_len", "output_len", "concurrency_group",
        "start_ts", "end_ts", "status", "prompt_tokens", "completion_tokens",
    ]

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "default",
        timeout_s: float = 600,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    async def _send_one(
        self,
        client: httpx.AsyncClient,
        spec: RequestSpec,
        semaphore: asyncio.Semaphore,
    ) -> RequestResult:
        """Send a single chat completion request."""
        async with semaphore:
            start = datetime.now(timezone.utc).isoformat()
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": spec.prompt}],
                "max_tokens": spec.output_len,
                "temperature": 0.0,
            }
            try:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=self.timeout_s,
                )
                end = datetime.now(timezone.utc).isoformat()
                if resp.status_code == 200:
                    data = resp.json()
                    usage = data.get("usage", {})
                    return RequestResult(
                        req_id=spec.req_id,
                        input_len=spec.input_len,
                        output_len=spec.output_len,
                        concurrency_group=spec.concurrency_group,
                        start_ts=start,
                        end_ts=end,
                        status="ok",
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                    )
                else:
                    return RequestResult(
                        req_id=spec.req_id,
                        input_len=spec.input_len,
                        output_len=spec.output_len,
                        concurrency_group=spec.concurrency_group,
                        start_ts=start,
                        end_ts=end,
                        status=f"error:http_{resp.status_code}",
                        prompt_tokens=0,
                        completion_tokens=0,
                    )
            except Exception as e:
                end = datetime.now(timezone.utc).isoformat()
                return RequestResult(
                    req_id=spec.req_id,
                    input_len=spec.input_len,
                    output_len=spec.output_len,
                    concurrency_group=spec.concurrency_group,
                    start_ts=start,
                    end_ts=end,
                    status=f"error:{type(e).__name__}",
                    prompt_tokens=0,
                    completion_tokens=0,
                )

    async def run_group(
        self,
        specs: list[RequestSpec],
        concurrency: int,
    ) -> list[RequestResult]:
        """Run a group of requests with given concurrency limit."""
        semaphore = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient() as client:
            tasks = [
                self._send_one(client, spec, semaphore)
                for spec in specs
            ]
            return await asyncio.gather(*tasks)

    async def run_workload(
        self,
        workload_path: Path,
        output_path: Path,
    ) -> list[RequestResult]:
        """Load workload JSONL, run by concurrency group, write trace CSV.

        Workload JSONL format per line:
          {"req_id": "r001", "prompt": "...", "input_len": 8192,
           "output_len": 64, "concurrency_group": 8}

        Requests are grouped by concurrency_group and run sequentially
        per group (all requests within a group run concurrently at that
        concurrency level, then the next group starts).
        """
        # Load specs
        specs: list[RequestSpec] = []
        with open(workload_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                specs.append(RequestSpec(
                    req_id=d["req_id"],
                    prompt=d["prompt"],
                    input_len=d["input_len"],
                    output_len=d["output_len"],
                    concurrency_group=d["concurrency_group"],
                    extra=d.get("extra", {}),
                ))

        # Group by concurrency
        groups: dict[int, list[RequestSpec]] = {}
        for s in specs:
            groups.setdefault(s.concurrency_group, []).append(s)

        # Run groups in ascending concurrency order
        all_results: list[RequestResult] = []
        for conc in sorted(groups.keys()):
            group_specs = groups[conc]
            print(f"  [runner] concurrency={conc}, requests={len(group_specs)}")
            results = await self.run_group(group_specs, conc)
            all_results.extend(results)

            # Brief pause between groups to let server stabilize
            await asyncio.sleep(2.0)

        # Write trace CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADER)
            writer.writeheader()
            for r in all_results:
                writer.writerow({
                    "req_id": r.req_id,
                    "input_len": r.input_len,
                    "output_len": r.output_len,
                    "concurrency_group": r.concurrency_group,
                    "start_ts": r.start_ts,
                    "end_ts": r.end_ts,
                    "status": r.status,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                })

        ok = sum(1 for r in all_results if r.status == "ok")
        print(f"  [runner] done: {ok}/{len(all_results)} ok")
        return all_results


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Async request runner")
    parser.add_argument(
        "workload", type=Path, help="Workload JSONL file",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("request_trace.csv"),
        help="Output trace CSV (default: request_trace.csv)",
    )
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000/v1",
        help="vLLM API base URL",
    )
    parser.add_argument(
        "--model", default="default",
        help="Model name for API (default: 'default')",
    )
    parser.add_argument(
        "--timeout", type=float, default=600,
        help="Per-request timeout in seconds (default: 600)",
    )
    args = parser.parse_args()

    runner = AsyncRequestRunner(args.base_url, args.model, args.timeout)
    await runner.run_workload(args.workload, args.output)


if __name__ == "__main__":
    asyncio.run(main())
