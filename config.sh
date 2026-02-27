#!/bin/bash
# Multi-Agent Bench — client-side configuration
# GPU server runs vLLM separately; this config is for the benchmark client only.
#
# Server presets (matches run_server.sh):
#   tp1-fp8   — single GPU, FP8 quantized      (max c≈32)
#   tp2-fp8   — 2 GPU tensor parallel, FP8     (max c≈64)
#   tp2-fp16  — 2 GPU tensor parallel, FP16    (max c≈64)
#   tp1-fp16  — single GPU, FP16/BF16          (max c≈32)

# Server preset — determines model name automatically
export PRESET="${PRESET:-tp2-fp8}"

# Benchmark framework: "autogen" | "langgraph" | "a2a"
export BENCH_FRAMEWORK="${BENCH_FRAMEWORK:-autogen}"

# Derive model from preset
case "$PRESET" in
  tp1-fp8|tp2-fp8|dp2-ep2-fp8)
    export MODEL="${MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct-FP8}" ;;
  tp1-fp16|tp2-fp16|dp2-ep2-fp16)
    export MODEL="${MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct}" ;;
  *)
    export MODEL="${MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct-FP8}" ;;
esac

# GPU server connection
export SERVER_IP="${SERVER_IP:-paladin.ucsd.edu}"
export SERVER_PORT="${SERVER_PORT:-8000}"
export BASE_URL="${BASE_URL:-http://${SERVER_IP}:${SERVER_PORT}/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# Dataset (ShareGPT V3 text-only)
export DATASET_PATH="${DATASET_PATH:-/home/jp/CXL_project/old_heimdall/benchmark/llm_bench/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"

# Benchmark parameters
export BENCH_TASKS="${BENCH_TASKS:-48}"
export BENCH_CONCURRENCY="${BENCH_CONCURRENCY:-8}"
export BENCH_TASK_CONCURRENCY="${BENCH_TASK_CONCURRENCY:-8}"
export BENCH_EXECUTORS="${BENCH_EXECUTORS:-2}"
export BENCH_OUTPUT_DIR="${BENCH_OUTPUT_DIR:-results_multiagent}"
