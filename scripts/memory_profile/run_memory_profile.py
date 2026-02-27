#!/usr/bin/env python3
"""Memory-profile orchestrator.

Coordinates the three components:
  1. Workload generator → JSONL
  2. GPU memory sampler → background CSV
  3. Async request runner → trace CSV

Produces output directory:
  results/memory_profile/<run_id>/
    ├── workload.jsonl
    ├── request_trace.csv
    ├── gpu_mem_trace.csv
    ├── summary.json
    └── manifest.json
"""

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running from repo root or scripts/memory_profile/
sys.path.insert(0, str(Path(__file__).parent))

from gpu_mem_sampler import GpuMemSampler
from request_runner import AsyncRequestRunner
from workload_gen import generate_workload


def make_run_id(tag: str = "") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if tag:
        return f"{ts}_{tag}"
    return ts


def build_manifest(
    run_id: str,
    args,
    n_requests: int,
    n_samples: int,
    n_ok: int,
    duration_s: float,
) -> dict:
    return {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "input_lens": [int(x) for x in args.input_lens.split(",")],
            "output_lens": [int(x) for x in args.output_lens.split(",")],
            "concurrencies": [int(x) for x in args.concurrencies.split(",")],
            "prefix_modes": args.prefix_modes.split(","),
            "n_per_condition": args.n_per_condition,
            "gpu_sample_interval_ms": args.gpu_interval_ms,
            "gpu_indices": args.gpu_indices,
        },
        "results": {
            "total_requests": n_requests,
            "ok_requests": n_ok,
            "gpu_samples": n_samples,
            "duration_s": round(duration_s, 2),
        },
        "files": {
            "workload": "workload.jsonl",
            "request_trace": "request_trace.csv",
            "gpu_mem_trace": "gpu_mem_trace.csv",
            "summary": "summary.json",
        },
    }


def build_summary(results, n_samples: int) -> dict:
    """Build a quick summary from request results."""
    ok = [r for r in results if r.status == "ok"]
    err = [r for r in results if r.status != "ok"]

    # Group by concurrency
    by_conc: dict[int, list] = {}
    for r in ok:
        by_conc.setdefault(r.concurrency_group, []).append(r)

    conc_summary = {}
    for conc in sorted(by_conc.keys()):
        group = by_conc[conc]
        conc_summary[str(conc)] = {
            "n_requests": len(group),
            "n_ok": len(group),
            "avg_completion_tokens": (
                sum(r.completion_tokens for r in group) / len(group)
                if group else 0
            ),
        }

    return {
        "total_requests": len(results),
        "ok_requests": len(ok),
        "error_requests": len(err),
        "gpu_samples": n_samples,
        "by_concurrency": conc_summary,
        "errors": [
            {"req_id": r.req_id, "status": r.status}
            for r in err[:20]  # Cap at 20 errors in summary
        ],
    }


async def run(args):
    run_id = make_run_id(args.tag)
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Memory Profile: {run_id} ===")
    print(f"Output: {out_dir}")

    # Step 1: Generate workload
    workload_path = out_dir / "workload.jsonl"
    print("\n[1/3] Generating workload...")
    n_requests = generate_workload(
        workload_path,
        input_lens=[int(x) for x in args.input_lens.split(",")],
        output_lens=[int(x) for x in args.output_lens.split(",")],
        concurrencies=[int(x) for x in args.concurrencies.split(",")],
        prefix_modes=args.prefix_modes.split(","),
        n_per_condition=args.n_per_condition,
    )
    print(f"  {n_requests} requests → {workload_path}")

    # Step 2: Start GPU sampler
    gpu_indices = None
    if args.gpu_indices:
        gpu_indices = [int(g) for g in args.gpu_indices.split(",")]

    gpu_trace_path = out_dir / "gpu_mem_trace.csv"
    sampler = GpuMemSampler(gpu_trace_path, args.gpu_interval_ms, gpu_indices)
    print(f"\n[2/3] Starting GPU sampler (interval={args.gpu_interval_ms}ms)...")
    sampler.start()

    # Step 3: Run requests
    print(f"\n[3/3] Running requests → {args.base_url}")
    t0 = time.monotonic()
    runner = AsyncRequestRunner(args.base_url, args.model, args.timeout)
    request_trace_path = out_dir / "request_trace.csv"

    try:
        results = await runner.run_workload(workload_path, request_trace_path)
    except KeyboardInterrupt:
        print("\n  Interrupted!")
        results = []
    finally:
        duration_s = time.monotonic() - t0
        n_samples = sampler.stop()

    # Write summary
    n_ok = sum(1 for r in results if r.status == "ok")
    summary = build_summary(results, n_samples)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Write manifest
    manifest = build_manifest(
        run_id, args, n_requests, n_samples, n_ok, duration_s,
    )
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Final report
    print(f"\n=== Done ===")
    print(f"  Requests: {n_ok}/{len(results)} ok")
    print(f"  GPU samples: {n_samples}")
    print(f"  Duration: {duration_s:.1f}s")
    print(f"  Output: {out_dir}")

    return 0 if n_ok == len(results) else 1


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Memory-profile experiment orchestrator",
    )
    # Server
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000/v1",
        help="vLLM API base URL (default: http://127.0.0.1:8000/v1)",
    )
    parser.add_argument(
        "--model", default="default",
        help="Model name for API requests",
    )
    parser.add_argument(
        "--timeout", type=float, default=600,
        help="Per-request timeout in seconds (default: 600)",
    )

    # Workload grid
    parser.add_argument(
        "--input-lens", default="8192,16384,32768",
        help="Comma-separated input lengths",
    )
    parser.add_argument(
        "--output-lens", default="64,128",
        help="Comma-separated output lengths",
    )
    parser.add_argument(
        "--concurrencies", default="8,16,32,64",
        help="Comma-separated concurrency levels",
    )
    parser.add_argument(
        "--prefix-modes", default="low,high",
        help="Comma-separated prefix modes (low,high)",
    )
    parser.add_argument(
        "--n-per-condition", type=int, default=1,
        help="Requests per condition (default: 1)",
    )

    # GPU sampler
    parser.add_argument(
        "--gpu-interval-ms", type=int, default=200,
        help="GPU polling interval in ms (default: 200)",
    )
    parser.add_argument(
        "--gpu-indices", default=None,
        help="Comma-separated GPU indices (default: all)",
    )

    # Output
    parser.add_argument(
        "--output-dir", default="results/memory_profile",
        help="Base output directory",
    )
    parser.add_argument(
        "--tag", default="",
        help="Optional tag appended to run_id",
    )

    args = parser.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
