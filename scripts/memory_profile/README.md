# Memory Profile Experiment Mode

GPU memory usage profiling for vLLM inference under controlled workloads.
Measures **memory pressure** (not latency metrics like TTFT/TPOT/TPS).

## Components

| File | Purpose |
|------|---------|
| `gpu_mem_sampler.py` | nvidia-smi polling → `gpu_mem_trace.csv` |
| `request_runner.py` | Async httpx → `request_trace.csv` |
| `workload_gen.py` | Grid scan → `workload.jsonl` |
| `run_memory_profile.py` | Orchestrator (runs all three) |
| `run_memory_profile.sh` | Shell wrapper with env-var config |
| `smoke_test.py` | Quick validation (no GPU needed) |

## Quick Start

```bash
# 1. Start vLLM server (on Paladin or target machine)
export HF_HOME=/mnt/raid0_ssd/huggingface
export TMPDIR=/mnt/raid0_ssd/jinpyo/tmp
bash /home/jinpyo/llm_serving/run_server.sh tp2-fp16

# 2. Run memory profile (from multi-agent-bench repo root)
VLLM_URL=http://127.0.0.1:8000/v1 \
MODEL_NAME=Qwen/Qwen3-Next-80B-A3B \
GPU_INDICES=0,1 \
RUN_TAG=tp2_fp16 \
bash scripts/memory_profile/run_memory_profile.sh
```

## Output Structure

```
results/memory_profile/<run_id>/
├── workload.jsonl          # Generated request specs
├── request_trace.csv       # Per-request timing + status
├── gpu_mem_trace.csv       # GPU memory samples (200ms interval)
├── summary.json            # Quick stats
└── manifest.json           # Full config + metadata
```

## Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_URL` | `http://127.0.0.1:8000/v1` | vLLM API endpoint |
| `MODEL_NAME` | `default` | Model name for API |
| `INPUT_LENS` | `8192,16384,32768` | Input lengths (tokens) |
| `OUTPUT_LENS` | `64,128` | Output lengths (tokens) |
| `CONCURRENCIES` | `8,16,32,64` | Concurrency levels |
| `PREFIX_MODES` | `low,high` | Prefix reuse modes |
| `N_PER_CONDITION` | `1` | Requests per grid point |
| `GPU_INTERVAL_MS` | `200` | nvidia-smi poll interval |
| `GPU_INDICES` | (all) | GPU indices to monitor |
| `OUTPUT_DIR` | `results/memory_profile` | Output base dir |
| `RUN_TAG` | (none) | Tag appended to run_id |
| `REQUEST_TIMEOUT` | `600` | Per-request timeout (s) |

## Workload Grid

Default grid: 3×2×4×2 = 48 requests per run.

| Axis | Values |
|------|--------|
| input_len | 8192, 16384, 32768 |
| output_len | 64, 128 |
| concurrency | 8, 16, 32, 64 |
| prefix_reuse | low, high |

## Standalone Usage

Each component can run independently:

```bash
# Just generate workload
python scripts/memory_profile/workload_gen.py -o /tmp/test.jsonl

# Just sample GPU memory
python scripts/memory_profile/gpu_mem_sampler.py -o /tmp/gpu.csv --duration 10

# Just run requests
python scripts/memory_profile/request_runner.py /tmp/test.jsonl -o /tmp/trace.csv
```

## Smoke Test

```bash
python scripts/memory_profile/smoke_test.py
```

Validates workload generation + CSV format without needing a GPU or vLLM server.
