#!/bin/bash
# =============================================================================
# LMCache / KV-Offload Experiment — Paladin Configuration
# =============================================================================
# Source this before running experiments:
#   source scripts/lmcache/paladin_config.sh
# =============================================================================

# --- Paladin environment ---
export HF_HOME="/mnt/raid0_ssd/huggingface"
export TMPDIR="/mnt/raid0_ssd/jinpyo/tmp"
export VLLM_CACHE_ROOT="/mnt/raid0_ssd/jinpyo/.cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="/mnt/raid0_ssd/jinpyo/.cache/torch"

# --- Model ---
export MODEL="${MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct}"
export TP_SIZE="${TP_SIZE:-2}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export GPU_UTIL="${GPU_UTIL:-0.90}"

# --- Server ---
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8000}"
export BASE_URL="http://${HOST}:${PORT}/v1"

# --- Experiment ---
export RESULT_BASE="/mnt/raid0_ssd/jinpyo/lmcache_results"
export RUN_ID="${RUN_ID:-lmcache_$(date +%Y%m%d_%H%M%S)}"
export RESULT_DIR="${RESULT_BASE}/${RUN_ID}"
export LOG_DIR="${RESULT_DIR}/logs"

# --- Benchmark ---
export CONC_LEVELS="${CONC_LEVELS:-1 8 32}"
export NUM_TURNS="${NUM_TURNS:-8}"
export NUM_SESSIONS="${NUM_SESSIONS:-32}"
export SYSTEM_PROMPT_TOKENS="${SYSTEM_PROMPT_TOKENS:-2000}"

# --- KV Offload Configs ---
# Each "mode" is a server configuration to test
# Format: MODE_NAME:FLAGS
declare -A KV_MODES
KV_MODES=(
  [baseline]=""
  [prefix]="--enable-prefix-caching"
  [prefix_offload]="--enable-prefix-caching --kv-offloading-backend native --kv-offloading-size 20"
)

# LMCache mode (optional — requires lmcache pip package)
# Uncomment after installing lmcache:
# KV_MODES[prefix_lmcache]="--enable-prefix-caching --kv-transfer-config {\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}"

# --- Python ---
export VENV_PY="${VENV_PY:-python3}"
# If using conda:
# export VENV_PY="/mnt/raid0_ssd/jinpyo/miniconda3/envs/vllm/bin/python"

# --- Helpers ---
kill_server() {
  local pid_file="${LOG_DIR}/server_${PORT}.pid"
  if [ -f "${pid_file}" ]; then
    local pid=$(cat "${pid_file}")
    echo "[server] Killing PID ${pid}..."
    kill -TERM "${pid}" 2>/dev/null || true
    sleep 2
    kill -9 "${pid}" 2>/dev/null || true
    rm -f "${pid_file}"
  fi
  # Also kill any vllm on our port
  lsof -ti:${PORT} 2>/dev/null | xargs kill -9 2>/dev/null || true
  # Wait for GPU release
  echo -n "[server] Waiting for GPU release..."
  for _ in $(seq 1 30); do
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "${GPU_PROCS}" -eq 0 ]; then
      echo " clear."
      return 0
    fi
    sleep 2
    echo -n "."
  done
  echo " timeout (force killing)."
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | while read pid; do
    kill -9 "$pid" 2>/dev/null || true
  done
  sleep 3
}

start_server() {
  local mode_name="$1"
  local extra_flags="$2"
  local server_log="${LOG_DIR}/server_${mode_name}.log"
  local pid_file="${LOG_DIR}/server_${PORT}.pid"

  mkdir -p "${LOG_DIR}"

  echo "[server] Starting mode=${mode_name}"
  echo "[server]   Model: ${MODEL}"
  echo "[server]   TP: ${TP_SIZE}, GPU util: ${GPU_UTIL}"
  echo "[server]   Extra flags: ${extra_flags}"

  local cmd_args=(
    -m vllm.entrypoints.openai.api_server
    --model "${MODEL}"
    --host "${HOST}"
    --port "${PORT}"
    --tensor-parallel-size "${TP_SIZE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --gpu-memory-utilization "${GPU_UTIL}"
    --trust-remote-code
    --download-dir "${HF_HOME}/hub"
  )

  # Append extra flags (prefix caching, KV offload, etc.)
  if [ -n "${extra_flags}" ]; then
    # shellcheck disable=SC2206
    cmd_args+=(${extra_flags})
  fi

  export VLLM_USE_V1=1
  export VLLM_TORCH_COMPILE=0

  setsid env \
    VLLM_USE_V1=1 \
    VLLM_TORCH_COMPILE=0 \
    HF_HOME="${HF_HOME}" \
    TMPDIR="${TMPDIR}" \
    ${VENV_PY} "${cmd_args[@]}" > "${server_log}" 2>&1 &

  echo $! > "${pid_file}"
  disown $! 2>/dev/null || true
  echo "[server] PID=$(cat "${pid_file}"), log=${server_log}"

  # Health check
  echo -n "[server] Waiting for /health..."
  for i in $(seq 1 600); do
    if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
      echo " ready (${i}s)."
      return 0
    fi
    sleep 1
  done
  echo " TIMEOUT."
  tail -20 "${server_log}"
  return 1
}

scrape_metrics() {
  local output_file="$1"
  curl -sf "http://${HOST}:${PORT}/metrics" > "${output_file}" 2>/dev/null || echo "(metrics unavailable)"
}
