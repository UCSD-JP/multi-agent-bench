#!/bin/bash
# Multi-Agent Bench — run benchmark from workstation against remote GPU server
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "=== Multi-Agent Benchmark ==="
echo "  Framework:   ${BENCH_FRAMEWORK}"
echo "  Model:       ${MODEL}"
echo "  Server:      ${BASE_URL}"
echo "  Tasks:       ${BENCH_TASKS}"
echo "  Concurrency: ${BENCH_CONCURRENCY}"
echo "  Executors:   ${BENCH_EXECUTORS}"
echo ""

# Health check
echo "[check] Testing server connectivity..."
if ! curl -sf "${BASE_URL}/models" > /dev/null 2>&1; then
  echo "ERROR: Cannot reach vLLM server at ${BASE_URL}/models"
  echo "Make sure the server is running on ${SERVER_IP}:${SERVER_PORT}"
  exit 1
fi
echo "[check] Server is up."
echo ""

python3 "${SCRIPT_DIR}/benchmark_agentic.py" \
  --framework "${BENCH_FRAMEWORK}" \
  --model "${MODEL}" \
  --dataset_path "${DATASET_PATH}" \
  --base_url "${BASE_URL}" \
  --api_key "${OPENAI_API_KEY}" \
  --tasks "${BENCH_TASKS}" \
  --concurrency "${BENCH_CONCURRENCY}" \
  --task_concurrency "${BENCH_TASK_CONCURRENCY}" \
  --executors "${BENCH_EXECUTORS}" \
  --output_dir "${BENCH_OUTPUT_DIR}"
