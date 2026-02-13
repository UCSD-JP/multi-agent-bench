#!/bin/bash
# Multi-Agent Bench — client-side configuration
# GPU server runs vLLM separately; this config is for the benchmark client only.

# Benchmark framework: "autogen" | "langgraph" | "a2a"
export BENCH_FRAMEWORK="${BENCH_FRAMEWORK:-autogen}"

# Model served on GPU server
export MODEL="${MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct}"

# GPU server connection
# wolverine (TP1 single GPU):
export SERVER_IP="${SERVER_IP:-wolverine.ucsd.edu}"
# paladin (TP2 multi-GPU):
# export SERVER_IP="${SERVER_IP:-paladin.ucsd.edu}"
export SERVER_PORT="${SERVER_PORT:-8000}"
export BASE_URL="${BASE_URL:-http://${SERVER_IP}:${SERVER_PORT}/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# Dataset (ShareGPT V3 text-only)
export DATASET_PATH="${DATASET_PATH:-/mnt/raid0_ssd/huggingface/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"

# Benchmark parameters
export BENCH_TASKS="${BENCH_TASKS:-64}"
export BENCH_CONCURRENCY="${BENCH_CONCURRENCY:-32}"
export BENCH_EXECUTORS="${BENCH_EXECUTORS:-2}"
export BENCH_OUTPUT_DIR="${BENCH_OUTPUT_DIR:-results_multiagent}"
