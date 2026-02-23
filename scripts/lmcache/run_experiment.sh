#!/bin/bash
# =============================================================================
# LMCache / KV-Offload Experiment Runner — Paladin
# =============================================================================
# Runs benchmark across server configs (baseline, prefix, prefix+offload)
# and concurrency levels. Restarts server between configs.
#
# Usage:
#   bash scripts/lmcache/run_experiment.sh                # all modes, all concurrencies
#   MODES="baseline prefix" bash scripts/lmcache/run_experiment.sh  # subset
#   CONC_LEVELS="1 8" NUM_TURNS=5 bash scripts/lmcache/run_experiment.sh
#
# Prerequisites:
#   - conda activate vllm  (on Paladin)
#   - pip install httpx     (for benchmark client)
#   - pip install lmcache   (optional, for lmcache mode)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/paladin_config.sh"

# --- Parse modes to run ---
# Default: baseline, prefix, prefix_offload
MODES="${MODES:-baseline prefix prefix_offload}"

echo "================================================================"
echo " LMCache / KV-Offload Experiment"
echo " RUN_ID:       ${RUN_ID}"
echo " Modes:        ${MODES}"
echo " Concurrency:  ${CONC_LEVELS}"
echo " Sessions:     ${NUM_SESSIONS}"
echo " Turns:        ${NUM_TURNS}"
echo " Sys prompt:   ~${SYSTEM_PROMPT_TOKENS} tokens"
echo " Model:        ${MODEL}"
echo " Results:      ${RESULT_DIR}"
echo "================================================================"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

# Save experiment config
python3 -c "
import json, datetime
config = {
    'run_id': '${RUN_ID}',
    'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
    'modes': '${MODES}'.split(),
    'conc_levels': [int(x) for x in '${CONC_LEVELS}'.split()],
    'num_sessions': ${NUM_SESSIONS},
    'num_turns': ${NUM_TURNS},
    'system_prompt_tokens': ${SYSTEM_PROMPT_TOKENS},
    'model': '${MODEL}',
    'tp_size': ${TP_SIZE},
    'max_model_len': ${MAX_MODEL_LEN},
    'gpu_util': ${GPU_UTIL},
}
with open('${RESULT_DIR}/experiment_config.json', 'w') as f:
    json.dump(config, f, indent=2)
print(json.dumps(config, indent=2))
"

# --- Main loop: mode × concurrency ---
for MODE in ${MODES}; do
  echo ""
  echo "###############################################"
  echo "# Mode: ${MODE}"
  echo "###############################################"

  # Get server flags for this mode
  case "${MODE}" in
    baseline)
      EXTRA_FLAGS="" ;;
    prefix)
      EXTRA_FLAGS="--enable-prefix-caching" ;;
    prefix_offload)
      EXTRA_FLAGS="--enable-prefix-caching --kv-offloading-backend native --kv-offloading-size 20" ;;
    prefix_lmcache)
      if ! python3 -c "import lmcache" 2>/dev/null; then
        echo "[SKIP] lmcache not installed. Run: pip install lmcache"
        continue
      fi
      EXTRA_FLAGS="--enable-prefix-caching --kv-transfer-config {\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}" ;;
    *)
      echo "[ERROR] Unknown mode: ${MODE}"
      continue ;;
  esac

  # Kill existing server
  kill_server

  # Start server with this mode's config
  if ! start_server "${MODE}" "${EXTRA_FLAGS}"; then
    echo "[ERROR] Server failed to start for mode=${MODE}. Skipping."
    continue
  fi

  # Warmup: single short request to fill CUDA graph caches
  echo "[bench] Warmup request..."
  python3 -c "
import httpx, json
resp = httpx.post('${BASE_URL}/chat/completions', json={
    'model': '${MODEL}',
    'messages': [{'role': 'user', 'content': 'Hello'}],
    'max_tokens': 5,
}, timeout=120)
print(f'  Warmup: {resp.status_code}')
" 2>/dev/null || echo "  Warmup failed (non-critical)"
  sleep 2

  # Run benchmark at each concurrency level
  for CONC in ${CONC_LEVELS}; do
    OUTDIR="${RESULT_DIR}/${MODE}/c${CONC}"
    mkdir -p "${OUTDIR}"

    # Skip if already completed
    if [ -f "${OUTDIR}/results_${MODE}_c${CONC}.json" ]; then
      echo "[SKIP] ${MODE}/c${CONC} already done."
      continue
    fi

    echo ""
    echo "--- ${MODE} / c=${CONC} ---"

    # Scrape metrics before
    scrape_metrics "${OUTDIR}/metrics_before.prom"

    python3 "${SCRIPT_DIR}/benchmark_prefix_workload.py" \
      --base_url "${BASE_URL}" \
      --model "${MODEL}" \
      --concurrency "${CONC}" \
      --num_sessions "${NUM_SESSIONS}" \
      --num_turns "${NUM_TURNS}" \
      --system_prompt_tokens "${SYSTEM_PROMPT_TOKENS}" \
      --context_growth_tokens 1500 \
      --max_output_tokens 64 \
      --mode "${MODE}" \
      --output_dir "${OUTDIR}" \
      2>&1 | tee "${OUTDIR}/bench.log"

    # Scrape metrics after
    scrape_metrics "${OUTDIR}/metrics_after.prom"

    echo "[bench] Cooling 5s..."
    sleep 5
  done

  echo ""
  echo "[${MODE}] Done."
done

# Kill final server
kill_server

echo ""
echo "================================================================"
echo " Experiment Complete"
echo " Results: ${RESULT_DIR}"
echo "================================================================"

# Quick summary
echo ""
echo "=== Quick Summary ==="
python3 "${SCRIPT_DIR}/analyze_results.py" --result_dir "${RESULT_DIR}" --summary_only 2>/dev/null || echo "(analysis script not available)"
