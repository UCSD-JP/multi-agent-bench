#!/bin/bash
set -euo pipefail

PRESET="${1:-tp2-fp8}"
PORT="${PORT:-8000}"

# Engine: vllm (default) or sglang
ENGINE="${ENGINE:-vllm}"

# Defaults
DP="${DP:-1}"
EP="${EP:-false}"
EP_SIZE="${EP_SIZE:-0}"

case "$PRESET" in
  tp1-fp8)
    MODEL="${MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct-FP8}"
    TP=1; GPU_UTIL="${GPU_UTIL:-0.95}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}" ;;
  tp2-fp8)
    MODEL="${MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct-FP8}"
    TP=2; GPU_UTIL="${GPU_UTIL:-0.90}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}" ;;
  tp2-fp16)
    MODEL="${MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct}"
    TP=2; GPU_UTIL="${GPU_UTIL:-0.90}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}" ;;
  tp1-fp16)
    MODEL="${MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct}"
    TP=1; GPU_UTIL="${GPU_UTIL:-0.95}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}" ;;
  dp2-ep2-fp8)
    MODEL="${MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct-FP8}"
    TP=2; DP=1; EP=true; EP_SIZE=2; GPU_UTIL="${GPU_UTIL:-0.90}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}" ;;
  dp2-ep2-fp16)
    MODEL="${MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct}"
    TP=2; DP=1; EP=true; EP_SIZE=2; GPU_UTIL="${GPU_UTIL:-0.90}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}" ;;
  *)
    echo "Unknown preset: $PRESET"
    echo "Usage: [ENGINE=sglang] $0 <preset>"
    echo ""
    echo "Presets: tp1-fp8 | tp2-fp8 | tp2-fp16 | tp1-fp16 | dp2-ep2-fp8 | dp2-ep2-fp16"
    echo "Engines: vllm (default) | sglang"
    exit 1 ;;
esac

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MEM_FRAC="${MEM_FRAC:-0.88}"

echo "=== Starting ${ENGINE} server ==="
echo "  Preset:        ${PRESET}"
echo "  Engine:        ${ENGINE}"
echo "  Model:         ${MODEL}"
echo "  TP:            ${TP}"
echo "  DP:            ${DP}"
echo "  EP:            ${EP}"
echo "  Port:          ${PORT}"
echo "  Max model len: ${MAX_MODEL_LEN}"
echo ""

# ─────────────────────────────────────────────
# vLLM
# ─────────────────────────────────────────────
if [ "${ENGINE}" = "vllm" ]; then
  echo "  GPU util:      ${GPU_UTIL}"
  echo "  Max num seqs:  ${MAX_NUM_SEQS}"
  echo "  Metrics:       http://localhost:${PORT}/metrics"
  echo ""

  EP_FLAGS=""
  if [ "${EP}" = "true" ]; then
    EP_FLAGS="--enable-expert-parallel"
    if [ "${EPLB:-false}" = "true" ]; then
      EP_FLAGS="${EP_FLAGS} --enable-eplb"
    fi
    echo "  EP flags:      ${EP_FLAGS}"
    echo ""
  fi

  python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --enable-log-requests \
    ${EP_FLAGS}

# ─────────────────────────────────────────────
# SGLang
# ─────────────────────────────────────────────
elif [ "${ENGINE}" = "sglang" ]; then
  echo "  Mem fraction:  ${MEM_FRAC}"
  echo "  Metrics:       http://localhost:${PORT}/metrics"
  echo ""

  SGLANG_ARGS=(
    --model-path "${MODEL}"
    --host 0.0.0.0
    --port "${PORT}"
    --tp-size "${TP}"
    --mem-fraction-static "${MEM_FRAC}"
    --context-length "${MAX_MODEL_LEN}"
    --enable-metrics
    --log-requests
  )

  if [ "${EP}" = "true" ]; then
    SGLANG_ARGS+=(--dp-size "${DP}" --ep-size "${EP_SIZE}")
    echo "  SGLang EP:     dp=${DP} ep=${EP_SIZE}"
    echo ""
  fi

  python3 -m sglang.launch_server "${SGLANG_ARGS[@]}"

else
  echo "Unknown engine: ${ENGINE} (use vllm or sglang)"
  exit 1
fi
