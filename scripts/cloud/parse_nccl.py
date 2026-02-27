#!/usr/bin/env python3
"""Parse NCCL benchmark output into JSON summary."""
import json
import os
import re


def parse_nccl_output(filepath):
    results = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            try:
                size_bytes = int(parts[0])
                time_us = float(parts[-3])
                algbw = float(parts[-2])
                busbw = float(parts[-1])
                results.append({
                    "size_bytes": size_bytes,
                    "time_us": time_us,
                    "algbw_gbps": algbw,
                    "busbw_gbps": busbw,
                })
            except (ValueError, IndexError):
                pass
    return results


def main():
    outdir = os.environ.get("OUTDIR", ".")
    summary = {}

    for ngpu in [2, 4]:
        fpath = os.path.join(outdir, f"nccl_allreduce_{ngpu}gpu.txt")
        if not os.path.exists(fpath):
            continue
        data = parse_nccl_output(fpath)
        if not data:
            continue

        peak = max(data, key=lambda x: x["busbw_gbps"])
        mid = [d for d in data if 1e6 <= d["size_bytes"] <= 16e6]
        small = [d for d in data if d["size_bytes"] <= 65536]
        avg_mid = sum(d["busbw_gbps"] for d in mid) / len(mid) if mid else 0
        avg_lat = sum(d["time_us"] for d in small) / len(small) if small else 0

        summary[f"{ngpu}gpu"] = {
            "peak_busbw_gbps": round(peak["busbw_gbps"], 2),
            "mid_range_avg_busbw_gbps": round(avg_mid, 2),
            "small_msg_avg_latency_us": round(avg_lat, 1),
        }

    if "2gpu" in summary and "4gpu" in summary:
        s2, s4 = summary["2gpu"], summary["4gpu"]
        summary["tp4_vs_tp2"] = {
            "peak_bw_ratio": round(s4["peak_busbw_gbps"] / s2["peak_busbw_gbps"], 3),
            "mid_bw_ratio": round(s4["mid_range_avg_busbw_gbps"] / s2["mid_range_avg_busbw_gbps"], 3) if s2["mid_range_avg_busbw_gbps"] else None,
        }

    out = os.path.join(outdir, "nccl_allreduce_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\n[OK] Saved to {out}")


if __name__ == "__main__":
    main()
