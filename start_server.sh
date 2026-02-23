#!/bin/bash
set -euo pipefail

# --- LOAD CENTRALIZED CONFIG ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/config.sh" ]]; then
  source "${SCRIPT_DIR}/config.sh"
fi

# --- PYTHON ENVIRONMENT ---
VENV_PY="${VENV_PY:-/mnt/raid0_ssd/mingi/venvs/vllm/bin/python}"

# --- CACHE REDIRECTION ---
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/mnt/raid0_ssd/mingi/.cache/torch_compile}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/mnt/raid0_ssd/mingi/.cache/vllm}"
export HF_HOME="${HF_HOME:-/mnt/raid0_ssd/mingi/.cache/huggingface}"

# --- ENGINE SELECTION ---
# VLLM_USE_V1=1 (default): V1 engine with CUDA graphs (production mode)
# VLLM_USE_V1=0: V0 engine, use with --enforce-eager for profiling/debugging
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_TORCH_COMPILE=0

echo "[server] VLLM_USE_V1=${VLLM_USE_V1}"
echo "[server] VLLM_TORCH_COMPILE=${VLLM_TORCH_COMPILE}"

# --- NSYS PROFILING SUPPORT (disabled by default) ---
export NSYS_ENABLE=0
unset NSYS_OUTPUT_PREFIX 2>/dev/null || true

# Create the directories to be safe
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$HF_HOME"

# --- CONFIGURATION ---
MODEL="${MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-2}"
MAX_CTX="${MAX_CTX:-8192}"

# GPU utilization from centralized config (config.sh)
# Can be overridden by environment variable GPU_UTIL
if [[ "${NSYS_ENABLE}" == "1" ]]; then
  GPU_UTIL="${GPU_UTIL:-${GPU_UTIL_NSYS:-0.85}}"  # Use config value or default
  echo "[server] Nsight Systems profiling ENABLED - using reduced GPU utilization: ${GPU_UTIL}"
else
  GPU_UTIL="${GPU_UTIL:-${GPU_UTIL_DEFAULT:-0.90}}"  # Use config value or default
fi

# Reduce max-num-seqs to save memory when profiling
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"

# --- SERVER COMMAND ARGS ---
CMD_ARGS=(
  -m vllm.entrypoints.openai.api_server
  --model "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TP_SIZE}"
  --max-model-len "${MAX_CTX}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --gpu-memory-utilization "${GPU_UTIL}"
  --trust-remote-code
  --download-dir /mnt/raid0_ssd/huggingface/hub
)

# Legacy profiling mode: V0 requires --enforce-eager
if [[ "${VLLM_USE_V1}" == "0" ]]; then
  CMD_ARGS+=(--enforce-eager)
  echo "[server] V0 mode: --enforce-eager enabled (no CUDA graphs)"
else
  echo "[server] V1 mode: CUDA graphs enabled"
fi

# Optional metrics/logging flags (from config.sh)
if [[ "${ENABLE_LOG_REQUESTS:-0}" == "1" ]]; then
  CMD_ARGS+=(--enable-log-requests)
fi
if [[ "${ENABLE_LOG_OUTPUTS:-0}" == "1" ]]; then
  CMD_ARGS+=(--enable-log-outputs)
fi
if [[ "${ENABLE_PROMPT_TOKENS_DETAILS:-0}" == "1" ]]; then
  CMD_ARGS+=(--enable-prompt-tokens-details)
fi
if [[ "${ENABLE_FORCE_INCLUDE_USAGE:-0}" == "1" ]]; then
  CMD_ARGS+=(--enable-force-include-usage)
fi
if [[ "${ENABLE_SERVER_LOAD_TRACKING:-0}" == "1" ]]; then
  CMD_ARGS+=(--enable-server-load-tracking)
fi

# --- PID FILE FOR GRACEFUL SHUTDOWN ---
LOG_DIR="${LOG_DIR:-./}"
PID_FILE="${LOG_DIR}/server_${PORT}.pid"
SERVER_LOG="${SERVER_LOG:-${LOG_DIR}/server_${PORT}_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$LOG_DIR"

# --- LAUNCH WITH OR WITHOUT NSYS ---
if [[ "${NSYS_ENABLE}" == "1" ]]; then
  NSYS_OUTPUT_PREFIX="${NSYS_OUTPUT_PREFIX%.nsys-rep}"
  if [[ -z "${NSYS_OUTPUT_PREFIX}" ]]; then
    NSYS_OUTPUT_PREFIX="./nsys_profile_$(date +%Y%m%d_%H%M%S)"
  fi
  echo "[server] Nsight Systems output: ${NSYS_OUTPUT_PREFIX}.nsys-rep"
  
  # Wrap with nsys profile (delay allows model to load before profiling starts)
  # Note: Environment variables (VLLM_USE_V1, etc.) are inherited by the Python process
  # Use setsid to create a new session so the server survives when this script exits
  setsid env VLLM_USE_V1="${VLLM_USE_V1}" VLLM_TORCH_COMPILE=0 "${VENV_PY}" "${CMD_ARGS[@]}" > "${SERVER_LOG}" 2>&1 &
else
  echo "[server] GPU memory utilization: ${GPU_UTIL}"
  # Use setsid to create a new session so the server survives when this script exits
  # (nohup/disown alone is insufficient -- vLLM workers detect parent PID changes and self-terminate)
  setsid env VLLM_USE_V1="${VLLM_USE_V1}" VLLM_TORCH_COMPILE=0 "${VENV_PY}" "${CMD_ARGS[@]}" > "${SERVER_LOG}" 2>&1 &
fi

# Save the PID of the background process
echo $! > "${PID_FILE}"
disown $! 2>/dev/null || true
echo "[server] Started with PID $(cat "${PID_FILE}")"
echo "[server] Log file: ${SERVER_LOG}"

# Health Check
echo -n "[server] Waiting for /health ..."
for _ in $(seq 1 300); do
  if curl -sf "http://${HOST}:${PORT}/health" >/dev/null; then
    echo " ready."
    if [[ "${ENABLE_METRICS_SNAPSHOT:-0}" == "1" ]]; then
      METRICS_URL="${METRICS_URL:-http://${HOST}:${PORT}/metrics}"
      METRICS_OUTPUT_DIR="${METRICS_OUTPUT_DIR:-${LOG_DIR}}"
      SNAPSHOT_INTERVAL="${METRICS_SNAPSHOT_INTERVAL_SEC:-10}"
      mkdir -p "${METRICS_OUTPUT_DIR}"
      METRICS_PID_FILE="${METRICS_OUTPUT_DIR}/metrics_${PORT}.pid"
      SERVER_PID="$(cat "${PID_FILE}")"
      echo "[metrics] Snapshotting ${METRICS_URL} every ${SNAPSHOT_INTERVAL}s"
      setsid bash -c "while kill -0 ${SERVER_PID} 2>/dev/null; do ts=\$(date +%Y%m%d_%H%M%S); curl -sf \"${METRICS_URL}\" > \"${METRICS_OUTPUT_DIR}/metrics_${PORT}_\${ts}.prom\" 2>/dev/null || true; sleep ${SNAPSHOT_INTERVAL}; done" >/dev/null 2>&1 &
      echo $! > "${METRICS_PID_FILE}"
      disown $! 2>/dev/null || true
    fi
    exit 0
  fi
  sleep 1
done

echo "[server] ERROR: Health check timeout"
exit 1