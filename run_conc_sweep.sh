#!/bin/bash
# Multi-Agent Bench — Concurrency sweep
# Runs benchmark at multiple concurrency levels to characterize server scaling.
#
# Usage:
#   # TP2 FP8 (default)
#   PRESET=tp2-fp8 ./run_conc_sweep.sh
#
#   # TP2 FP16
#   PRESET=tp2-fp16 ./run_conc_sweep.sh
#
#   # TP1 FP8 (lower concurrency range)
#   PRESET=tp1-fp8 CONC_LEVELS="1 4 8 16 32" ./run_conc_sweep.sh
#
#   # Custom everything
#   PRESET=tp2-fp8 CONC_LEVELS="1 8 32" BENCH_TASKS=96 ./run_conc_sweep.sh
#
# Output structure:
#   results_multiagent/sweep_{PRESET}_{FRAMEWORK}/
#     autogen_c1/   trace_*.jsonl + server_metrics_*.json
#     autogen_c8/   ...
#     autogen_c32/  ...
#
# Server must be running with matching preset (./run_server.sh $PRESET)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

FRAMEWORK="${BENCH_FRAMEWORK:-autogen}"
TASKS="${BENCH_TASKS:-48}"
EXECUTORS="${BENCH_EXECUTORS:-2}"

# Output dir includes preset for separation
OUTPUT_BASE="${BENCH_OUTPUT_DIR:-results_multiagent}/sweep_${PRESET}_${FRAMEWORK}"

# Default concurrency levels based on preset
case "$PRESET" in
  tp1-fp8|tp1-fp16)
    CONC_LEVELS="${CONC_LEVELS:-1 4 8 16 32}" ;;
  tp2-fp8|tp2-fp16)
    CONC_LEVELS="${CONC_LEVELS:-1 8 32}" ;;
  dp2-ep2-fp8|dp2-ep2-fp16)
    CONC_LEVELS="${CONC_LEVELS:-1 8 32 64}" ;;
  *)
    CONC_LEVELS="${CONC_LEVELS:-1 8 32}" ;;
esac

echo "=== Concurrency Sweep ==="
echo "  Preset:      ${PRESET}"
echo "  Framework:   ${FRAMEWORK}"
echo "  Model:       ${MODEL}"
echo "  Server:      ${BASE_URL}"
echo "  Tasks:       ${TASKS}"
echo "  Executors:   ${EXECUTORS}"
echo "  Levels:      ${CONC_LEVELS}"
echo "  Output:      ${OUTPUT_BASE}/"
echo ""

# Health check
echo "[check] Testing server connectivity..."
if ! curl -sf "${BASE_URL}/models" > /dev/null 2>&1; then
  echo "ERROR: Cannot reach vLLM server at ${BASE_URL}/models"
  echo "Start server: ./run_server.sh ${PRESET}"
  exit 1
fi

# Verify model matches preset
SERVED_MODEL=$(curl -sf "${BASE_URL}/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")
echo "[check] Server is up. Model: ${SERVED_MODEL}"
if [ "${SERVED_MODEL}" != "${MODEL}" ]; then
  echo "WARNING: Server model '${SERVED_MODEL}' != expected '${MODEL}'"
  echo "         Server may be running a different preset."
  read -p "Continue anyway? [y/N] " -r
  [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi
echo ""

for CONC in ${CONC_LEVELS}; do
  OUTDIR="${OUTPUT_BASE}/${FRAMEWORK}_c${CONC}"
  mkdir -p "${OUTDIR}"

  echo ""
  echo "=== ${PRESET} / ${FRAMEWORK} concurrency=${CONC} ==="

  python3 "${SCRIPT_DIR}/benchmark_agentic.py" \
    --framework "${FRAMEWORK}" \
    --model "${MODEL}" \
    --dataset_path "${DATASET_PATH}" \
    --base_url "${BASE_URL}" \
    --api_key "${OPENAI_API_KEY}" \
    --tasks "${TASKS}" \
    --concurrency "${CONC}" \
    --task_concurrency "${CONC}" \
    --executors "${EXECUTORS}" \
    --output_dir "${OUTDIR}"

  echo "[sweep] Finished c=${CONC}, sleeping 5s to let server cool..."
  sleep 5
done

echo ""
echo "=== Sweep complete ==="
echo "Preset: ${PRESET}  Framework: ${FRAMEWORK}"
echo "Results in: ${OUTPUT_BASE}/"
echo ""
echo "Summary per level:"
for CONC in ${CONC_LEVELS}; do
  OUTDIR="${OUTPUT_BASE}/${FRAMEWORK}_c${CONC}"
  METRICS_FILE="${OUTDIR}/server_metrics_${FRAMEWORK}_c${CONC}.json"
  if [ -f "${METRICS_FILE}" ]; then
    echo "  c=${CONC}: $(python3 -c "
import sys, json
d = json.load(open('${METRICS_FILE}'))
parts = []
if d.get('tpot_mean_ms'): parts.append(f\"TPOT={d['tpot_mean_ms']:.1f}ms\")
if d.get('gen_tps'): parts.append(f\"TPS={d['gen_tps']:.1f}\")
if d.get('ttft_mean_ms'): parts.append(f\"TTFT={d['ttft_mean_ms']:.1f}ms\")
print('  '.join(parts) if parts else 'no metrics')
" 2>/dev/null || echo "parse error")"
  else
    echo "  c=${CONC}: no server metrics file"
  fi
done

echo ""
echo "Analyze: python3 scripts/analyze_conc_sweep.py --sweep_dir ${OUTPUT_BASE}"
