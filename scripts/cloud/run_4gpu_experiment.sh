#!/bin/bash
# =============================================================================
# 4GPU NVLink Experiment — Full Pipeline
# =============================================================================
# Runs TP2 + TP4 experiments on a 4×H200/H100 NVLink machine.
# Collects: NCCL bench → TP2 (batch + agentic) → TP4 (batch + agentic)
#
# Usage:
#   bash scripts/cloud/run_4gpu_experiment.sh              # full run
#   SKIP_NCCL=1 bash scripts/cloud/run_4gpu_experiment.sh  # skip NCCL
#   SKIP_BATCH=1 bash scripts/cloud/run_4gpu_experiment.sh # skip batch sweep
#   PRESETS="tp2-fp16" bash scripts/cloud/run_4gpu_experiment.sh  # TP2 only
#
# Prerequisites:
#   - vLLM installed, model downloaded
#   - Dataset at ${DATASET_PATH}
#   - MAB repo with benchmark_agentic.py
#
# Output:
#   /data/results/${RUN_ID}/
#     nccl_bench/
#     tp2-fp16/batch_sweep_v4/   + sweep_tp2-fp16_autogen/
#     tp4-fp16/batch_sweep_v4/   + sweep_tp4-fp16_autogen/
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cloud_config.sh"

PRESETS="${PRESETS:-tp2-fp16 tp4-fp16}"
SKIP_NCCL="${SKIP_NCCL:-0}"
SKIP_BATCH="${SKIP_BATCH:-0}"
SKIP_AGENTIC="${SKIP_AGENTIC:-0}"

START_TIME=$(date +%s)

echo "================================================================"
echo " 4GPU NVLink Experiment"
echo " RUN_ID:   ${RUN_ID}"
echo " Presets:  ${PRESETS}"
echo " Results:  ${RESULT_DIR}"
echo " Skip:     NCCL=${SKIP_NCCL} BATCH=${SKIP_BATCH} AGENTIC=${SKIP_AGENTIC}"
echo "================================================================"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

# ─────────────────────────────────────────────────────────────────────
# Phase 0: Topology verification
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "########################################"
echo "# Phase 0: Topology Check"
echo "########################################"
nvidia-smi topo -m 2>&1 | tee "${RESULT_DIR}/topology.txt"
nvidia-smi -L 2>&1 | tee -a "${RESULT_DIR}/topology.txt"
echo ""

# Verify NVLink
if nvidia-smi topo -m 2>/dev/null | grep -q "NV"; then
  echo "[OK] NVLink detected"
else
  echo "[WARN] No NVLink detected — this may be a PCIe machine!"
  echo "       Continuing anyway, but results may not match expectations."
fi

# ─────────────────────────────────────────────────────────────────────
# Phase 1: NCCL AllReduce benchmark
# ─────────────────────────────────────────────────────────────────────
if [ "${SKIP_NCCL}" = "0" ]; then
  echo ""
  echo "########################################"
  echo "# Phase 1: NCCL Benchmark"
  echo "########################################"
  bash "${SCRIPT_DIR}/run_nccl_bench.sh"
else
  echo ""
  echo "[SKIP] NCCL benchmark"
fi

# ─────────────────────────────────────────────────────────────────────
# Helper: start vLLM server for a preset
# ─────────────────────────────────────────────────────────────────────
start_server() {
  local preset="$1"
  set_preset "$preset"

  echo "[server] Starting vLLM for ${PRESET} (TP=${TP_SIZE}, GPUs=${CUDA_DEVICES})"
  local log="${LOG_DIR}/vllm_${PRESET}.log"

  # Kill existing
  pkill -f "vllm.entrypoints" 2>/dev/null || true
  sleep 3
  echo -n "  Waiting for GPUs to free..."
  for i in $(seq 1 30); do
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    [ "${GPU_PROCS}" -eq 0 ] && echo " clear" && break
    sleep 2 && echo -n "."
  done

  CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host 0.0.0.0 --port ${VLLM_PORT} \
    --tensor-parallel-size ${TP_SIZE} \
    --gpu-memory-utilization ${GPU_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --download-dir "${HF_HOME}" \
    --trust-remote-code \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    > "${log}" 2>&1 &

  SERVER_PID=$!
  echo "  PID: ${SERVER_PID}, log: ${log}"

  # Wait for health
  echo -n "  Waiting for health..."
  for i in $(seq 1 120); do
    if curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
      echo " ready ($((i * 5))s)"
      return 0
    fi
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
      echo " FAILED (process exited)"
      tail -30 "${log}" 2>/dev/null
      return 1
    fi
    sleep 5 && echo -n "."
  done
  echo " TIMEOUT"
  tail -30 "${log}" 2>/dev/null
  return 1
}

kill_server() {
  echo "[server] Stopping vLLM..."
  pkill -f "vllm.entrypoints" 2>/dev/null || true
  sleep 5
  # Force kill if needed
  GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "${GPU_PROCS}" -gt 0 ]; then
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | while read pid; do
      kill -9 "$pid" 2>/dev/null || true
    done
    sleep 3
  fi
  echo "[server] Stopped"
}

# ─────────────────────────────────────────────────────────────────────
# Phase 2+3: Per-preset experiments
# ─────────────────────────────────────────────────────────────────────
for PRESET_NAME in ${PRESETS}; do
  echo ""
  echo "########################################"
  echo "# Preset: ${PRESET_NAME}"
  echo "########################################"

  set_preset "${PRESET_NAME}"
  PRESET_DIR="${RESULT_DIR}/${PRESET_NAME}"
  mkdir -p "${PRESET_DIR}"

  # Start server
  if ! start_server "${PRESET_NAME}"; then
    echo "[ERROR] Server failed for ${PRESET_NAME}, skipping"
    kill_server
    continue
  fi

  # Quick inference test
  echo ""
  echo "=== Inference Test ==="
  curl -s "http://localhost:${VLLM_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}],\"max_tokens\":10,\"temperature\":0}" \
    2>/dev/null | python3 -c "
import sys, json
try:
  r = json.loads(sys.stdin.read())
  print(f'  Response: {r[\"choices\"][0][\"message\"][\"content\"]}')
except: print('  [WARN] Inference test issue')
" || echo "  [WARN] Inference test failed"

  # --- Batch sweep ---
  if [ "${SKIP_BATCH}" = "0" ]; then
    echo ""
    echo "=== Batch Sweep: ${PRESET_NAME} ==="
    python3 "${SCRIPT_DIR}/run_batch_sweep.py" \
      --base-url "${BASE_URL}" \
      --api-key "${OPENAI_API_KEY}" \
      --model "${MODEL}" \
      --preset "${PRESET_NAME}" \
      --input-lens ${BATCH_INPUT_LENS} \
      --batch-sizes ${BATCH_SIZES} \
      --output-len ${BATCH_OUTPUT_LEN} \
      --output-dir "${PRESET_DIR}/batch_sweep_v4" \
      2>&1 | tee "${PRESET_DIR}/batch_sweep.log"
  else
    echo "[SKIP] Batch sweep"
  fi

  # --- Agentic sweep ---
  if [ "${SKIP_AGENTIC}" = "0" ]; then
    echo ""
    echo "=== Agentic Sweep: ${PRESET_NAME} ==="
    FW="${BENCH_FRAMEWORK}"
    for REP in $(seq 1 ${REPEAT}); do
      for CONC in ${CONC_LEVELS}; do
        OUTDIR="${PRESET_DIR}/sweep_${PRESET_NAME}_${FW}/r${REP}/${FW}_c${CONC}"
        mkdir -p "${OUTDIR}"

        echo ""
        echo "--- ${PRESET_NAME} / ${FW} / r${REP} / c=${CONC} ---"

        # Skip if done
        if [ -f "${OUTDIR}/server_metrics_${FW}_c${CONC}.json" ]; then
          echo "  [SKIP] Already completed"
          continue
        fi

        set +e
        python3 "${BENCH_SCRIPT}" \
          --framework "${FW}" \
          --model "${MODEL}" \
          --dataset_path "${DATASET_PATH}" \
          --base_url "${BASE_URL}" \
          --api_key "${OPENAI_API_KEY}" \
          --tasks "${BENCH_TASKS}" \
          --concurrency "${CONC}" \
          --task_concurrency "${CONC}" \
          --executors "${BENCH_EXECUTORS}" \
          --max_model_len "${MAX_MODEL_LEN}" \
          --output_dir "${OUTDIR}" \
          2>&1 | tee "${OUTDIR}/bench.log"
        EXIT_CODE=${PIPESTATUS[0]}
        set -e

        if [ ${EXIT_CODE} -ne 0 ]; then
          echo "  [WARN] Exit code ${EXIT_CODE}"
        fi
        echo "  Cooling 5s..."
        sleep 5
      done
    done
  else
    echo "[SKIP] Agentic sweep"
  fi

  # Kill server before next preset
  kill_server
done

# ─────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "================================================================"
echo " Experiment Complete"
echo " Duration: ${ELAPSED} minutes"
echo " Results:  ${RESULT_DIR}"
echo "================================================================"

echo ""
echo "=== Batch Sweep Summary ==="
for PRESET_NAME in ${PRESETS}; do
  SUMMARY="${RESULT_DIR}/${PRESET_NAME}/batch_sweep_v4/summary.json"
  if [ -f "${SUMMARY}" ]; then
    echo ""
    echo "  ${PRESET_NAME}:"
    python3 -c "
import json
data = json.load(open('${SUMMARY}'))
print(f'    {\"input\":>8} {\"batch\":>6} {\"TPOT\":>8} {\"TPS\":>8}')
for r in data:
    print(f'    {r[\"input_len\"]:>8} {r[\"batch_size\"]:>6} {r[\"tpot_mean_ms\"]:>7.2f}ms {r[\"gen_tps\"]:>7.1f}')
" 2>/dev/null || echo "    (parse error)"
  fi
done

echo ""
echo "=== Agentic Sweep Summary ==="
for PRESET_NAME in ${PRESETS}; do
  echo ""
  echo "  ${PRESET_NAME}:"
  find "${RESULT_DIR}/${PRESET_NAME}" -name "server_metrics_*.json" -exec python3 -c "
import json, sys, os
f = sys.argv[1]
d = json.load(open(f))
rel = os.path.relpath(f, '${RESULT_DIR}/${PRESET_NAME}')
parts = []
if d.get('tpot_mean_ms'): parts.append(f'TPOT={d[\"tpot_mean_ms\"]:.1f}ms')
if d.get('gen_tps'): parts.append(f'TPS={d[\"gen_tps\"]:.1f}')
print(f'    {rel}: {\"  \".join(parts)}')
" {} \; 2>/dev/null | sort || echo "    (no agentic data)"
done

echo ""
echo "================================================================"
echo " Copy results to local machine:"
echo "   rsync -avz root@<IP>:${RESULT_DIR}/ results_from_real_H100/"
echo "================================================================"
