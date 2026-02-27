#!/bin/bash
# =============================================================================
# Multi-Agent-Bench — Cloud Experiment Configuration
# =============================================================================
# Centralized config for Vast.ai 4GPU experiments (H100 SXM / H200 NVLink).
# All paths point to /data (volume) for persistence across stop/restart.
# =============================================================================

# --- Run identification ---
export RUN_ID="${RUN_ID:-run_$(date +%Y%m%d_%H%M%S)}"

# --- Volume paths (persistent) ---
export DATA_ROOT="/data"
export HF_HOME="${DATA_ROOT}/models"
export RESULT_DIR="${DATA_ROOT}/results/${RUN_ID}"
export LOG_DIR="${DATA_ROOT}/logs/${RUN_ID}"
export DATASET_DIR="${DATA_ROOT}/datasets"

# --- Dataset ---
export DATASET_PATH="${DATASET_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json"
export DATASET_URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

# --- Server settings ---
export VLLM_PORT="${VLLM_PORT:-18000}"
export BASE_URL="http://localhost:${VLLM_PORT}/v1"
export OPENAI_API_KEY="${OPEN_BUTTON_TOKEN:-${OPENAI_API_KEY:-EMPTY}}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# --- Model presets ---
set_preset() {
  local preset="${1:-tp2-fp16}"
  export PRESET="$preset"
  case "$preset" in
    tp2-fp16)
      export MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct"
      export TP_SIZE=2; export GPU_UTIL=0.90; export MAX_NUM_SEQS=64
      export CUDA_DEVICES="0,1" ;;
    tp2-fp8)
      export MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
      export TP_SIZE=2; export GPU_UTIL=0.90; export MAX_NUM_SEQS=64
      export CUDA_DEVICES="0,1" ;;
    tp4-fp16)
      export MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct"
      export TP_SIZE=4; export GPU_UTIL=0.90; export MAX_NUM_SEQS=128
      export CUDA_DEVICES="0,1,2,3" ;;
    tp4-fp8)
      export MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
      export TP_SIZE=4; export GPU_UTIL=0.90; export MAX_NUM_SEQS=128
      export CUDA_DEVICES="0,1,2,3" ;;
    *)
      echo "[config] Unknown preset: $preset"; return 1 ;;
  esac
  echo "[config] Preset=${PRESET} Model=${MODEL} TP=${TP_SIZE} GPUs=${CUDA_DEVICES}"
}

# --- Benchmark settings ---
export BENCH_FRAMEWORK="${BENCH_FRAMEWORK:-autogen}"
export BENCH_TASKS="${BENCH_TASKS:-48}"
export BENCH_EXECUTORS="${BENCH_EXECUTORS:-2}"
export CONC_LEVELS="${CONC_LEVELS:-1 8 32 64}"
export REPEAT="${REPEAT:-3}"

# --- Batch sweep grid ---
export BATCH_INPUT_LENS="${BATCH_INPUT_LENS:-128 512 2048}"
export BATCH_SIZES="${BATCH_SIZES:-1 8 16 32 64}"
export BATCH_OUTPUT_LEN="${BATCH_OUTPUT_LEN:-128}"

# --- Locate MAB repo (this repo) ---
export MAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export BENCH_SCRIPT="${MAB_ROOT}/benchmark_agentic.py"
