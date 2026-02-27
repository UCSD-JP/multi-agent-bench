#!/usr/bin/env bash
# Memory-profile experiment shell wrapper.
#
# Usage:
#   bash scripts/memory_profile/run_memory_profile.sh
#
# Environment variables (all optional, defaults shown):
#   VLLM_URL=http://127.0.0.1:8000/v1
#   MODEL_NAME=default
#   INPUT_LENS=8192,16384,32768
#   OUTPUT_LENS=64,128
#   CONCURRENCIES=8,16,32,64
#   PREFIX_MODES=low,high
#   N_PER_CONDITION=1
#   GPU_INTERVAL_MS=200
#   GPU_INDICES=          # e.g., "0,1" for first two GPUs
#   OUTPUT_DIR=results/memory_profile
#   RUN_TAG=              # optional tag, e.g., "tp2_fp16"
#   REQUEST_TIMEOUT=600
#
# Example (Paladin 2×H100):
#   VLLM_URL=http://127.0.0.1:8000/v1 \
#   MODEL_NAME=Qwen/Qwen3-Next-80B-A3B \
#   GPU_INDICES=0,1 \
#   RUN_TAG=tp2_fp16 \
#   bash scripts/memory_profile/run_memory_profile.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

# Defaults
VLLM_URL="${VLLM_URL:-http://127.0.0.1:8000/v1}"
MODEL_NAME="${MODEL_NAME:-default}"
INPUT_LENS="${INPUT_LENS:-8192,16384,32768}"
OUTPUT_LENS="${OUTPUT_LENS:-64,128}"
CONCURRENCIES="${CONCURRENCIES:-8,16,32,64}"
PREFIX_MODES="${PREFIX_MODES:-low,high}"
N_PER_CONDITION="${N_PER_CONDITION:-1}"
GPU_INTERVAL_MS="${GPU_INTERVAL_MS:-200}"
GPU_INDICES="${GPU_INDICES:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/memory_profile}"
RUN_TAG="${RUN_TAG:-}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"

echo "=== Memory Profile Experiment ==="
echo "  URL:           ${VLLM_URL}"
echo "  Model:         ${MODEL_NAME}"
echo "  Input lens:    ${INPUT_LENS}"
echo "  Output lens:   ${OUTPUT_LENS}"
echo "  Concurrencies: ${CONCURRENCIES}"
echo "  Prefix modes:  ${PREFIX_MODES}"
echo "  GPU interval:  ${GPU_INTERVAL_MS}ms"
echo "  Output dir:    ${OUTPUT_DIR}"
echo ""

# Build args
ARGS=(
    --base-url "${VLLM_URL}"
    --model "${MODEL_NAME}"
    --input-lens "${INPUT_LENS}"
    --output-lens "${OUTPUT_LENS}"
    --concurrencies "${CONCURRENCIES}"
    --prefix-modes "${PREFIX_MODES}"
    --n-per-condition "${N_PER_CONDITION}"
    --gpu-interval-ms "${GPU_INTERVAL_MS}"
    --output-dir "${OUTPUT_DIR}"
    --timeout "${REQUEST_TIMEOUT}"
)

if [ -n "${GPU_INDICES}" ]; then
    ARGS+=(--gpu-indices "${GPU_INDICES}")
fi

if [ -n "${RUN_TAG}" ]; then
    ARGS+=(--tag "${RUN_TAG}")
fi

python "${SCRIPT_DIR}/run_memory_profile.py" "${ARGS[@]}"
