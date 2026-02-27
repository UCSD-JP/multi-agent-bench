#!/usr/bin/env bash
# Run agentic multi-turn memory profile experiment.
# Monitors vLLM KV cache usage while simulating agent sessions.
#
# Usage:
#   bash run_agentic_memory_profile.sh
#   CONCURRENCIES=1,2,4 bash run_agentic_memory_profile.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configurable via environment variables
VLLM_URL="${VLLM_URL:-http://127.0.0.1:8000/v1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Next-80B-A3B-Instruct}"
CONCURRENCIES="${CONCURRENCIES:-1,2,4,8}"
BENCHMARKS="${BENCHMARKS:-swebench,terminalbench,livecodebench}"
MAX_TURNS="${MAX_TURNS:-7}"
MAX_TOKENS_PER_TURN="${MAX_TOKENS_PER_TURN:-256}"
MAX_CONTEXT="${MAX_CONTEXT:-30000}"
TOOL_BASE="${TOOL_BASE:-2000}"
TOOL_GROWTH="${TOOL_GROWTH:-1500}"

RUN_TAG="${RUN_TAG:-agentic_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="../../results/memory_profile/${RUN_TAG}"

echo "=== Agentic Memory Profile ==="
echo "  URL:          $VLLM_URL"
echo "  Model:        $MODEL_NAME"
echo "  Concurrency:  $CONCURRENCIES"
echo "  Benchmarks:   $BENCHMARKS"
echo "  Max turns:    $MAX_TURNS"
echo "  Max context:  $MAX_CONTEXT"
echo "  Tool base:    $TOOL_BASE tokens"
echo "  Tool growth:  $TOOL_GROWTH tokens/turn"
echo "  Output:       $OUT_DIR"
echo ""

# Quick health check
echo "--- Server health check ---"
HEALTH_URL="${VLLM_URL%/v1}/health"
if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
    echo "  Server is healthy"
else
    echo "  WARNING: Server health check failed at $HEALTH_URL"
    echo "  Proceeding anyway..."
fi

# Run experiment
python3 run_agentic_memory_profile.py \
    --base-url "$VLLM_URL" \
    --model "$MODEL_NAME" \
    --concurrencies "$CONCURRENCIES" \
    --benchmarks "$BENCHMARKS" \
    --max-turns "$MAX_TURNS" \
    --max-tokens-per-turn "$MAX_TOKENS_PER_TURN" \
    --max-context "$MAX_CONTEXT" \
    --tool-base-tokens "$TOOL_BASE" \
    --tool-growth-tokens "$TOOL_GROWTH" \
    --output-dir "$OUT_DIR"

echo ""
echo "=== Done ==="
echo "Results: $OUT_DIR/"
ls -la "$OUT_DIR/"
