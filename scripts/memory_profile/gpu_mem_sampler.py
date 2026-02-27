#!/usr/bin/env python3
"""GPU memory sampler using nvidia-smi polling.

Runs nvidia-smi in a subprocess at a configurable interval (default 200ms)
and writes timestamped GPU memory usage to a CSV file.

Output format:
  timestamp,gpu_idx,used_mib,total_mib,util_pct
  2026-02-25T10:00:00.123456+00:00,0,1234,81920,1.5
"""

import csv
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class GpuMemSampler:
    """Polls nvidia-smi at fixed intervals, writes CSV trace."""

    CSV_HEADER = ["timestamp", "gpu_idx", "used_mib", "total_mib", "util_pct"]

    def __init__(
        self,
        output_path: Path,
        interval_ms: int = 200,
        gpu_indices: Optional[list[int]] = None,
    ):
        self.output_path = Path(output_path)
        self.interval_s = interval_ms / 1000.0
        self.gpu_indices = gpu_indices  # None = all GPUs
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples_written = 0

    def _query_nvidia_smi(self) -> list[dict]:
        """Run nvidia-smi and parse memory usage per GPU."""
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return []
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

        rows = []
        ts = datetime.now(timezone.utc).isoformat()
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            gpu_idx = int(parts[0])
            if self.gpu_indices and gpu_idx not in self.gpu_indices:
                continue
            used_mib = float(parts[1])
            total_mib = float(parts[2])
            util_pct = float(parts[3])
            rows.append({
                "timestamp": ts,
                "gpu_idx": gpu_idx,
                "used_mib": used_mib,
                "total_mib": total_mib,
                "util_pct": util_pct,
            })
        return rows

    def _poll_loop(self):
        """Main polling loop running in background thread."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADER)
            writer.writeheader()
            while not self._stop_event.is_set():
                rows = self._query_nvidia_smi()
                for row in rows:
                    writer.writerow(row)
                    self._samples_written += 1
                f.flush()
                self._stop_event.wait(self.interval_s)

    def start(self):
        """Start background polling thread."""
        if self._thread is not None:
            raise RuntimeError("Sampler already running")
        self._stop_event.clear()
        self._samples_written = 0
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        """Stop polling and return number of samples written."""
        if self._thread is None:
            return 0
        self._stop_event.set()
        self._thread.join(timeout=5)
        self._thread = None
        return self._samples_written

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


def main():
    """CLI entry point for standalone sampling."""
    import argparse

    parser = argparse.ArgumentParser(description="GPU memory sampler")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("gpu_mem_trace.csv"),
        help="Output CSV path (default: gpu_mem_trace.csv)",
    )
    parser.add_argument(
        "--interval-ms", type=int, default=200,
        help="Polling interval in ms (default: 200)",
    )
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated GPU indices (default: all)",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Run for N seconds then stop (default: run until Ctrl+C)",
    )
    args = parser.parse_args()

    gpu_indices = None
    if args.gpus:
        gpu_indices = [int(g) for g in args.gpus.split(",")]

    sampler = GpuMemSampler(args.output, args.interval_ms, gpu_indices)
    print(f"Sampling GPU memory every {args.interval_ms}ms → {args.output}")
    sampler.start()

    try:
        if args.duration:
            time.sleep(args.duration)
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass

    n = sampler.stop()
    print(f"Stopped. {n} samples written to {args.output}")


if __name__ == "__main__":
    main()
