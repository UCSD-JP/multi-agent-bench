#!/bin/bash
# Scrape vLLM server-side metrics (Prometheus format)
# Usage: ./scrape_metrics.sh [before|after] [output_dir]
#
# Server-side metrics (no network latency):
#   vllm:time_to_first_token_seconds  — TTFT histogram
#   vllm:inter_token_latency_seconds  — TPOT histogram
#   vllm:e2e_request_latency_seconds  — E2E latency histogram
#   vllm:generation_tokens_total      — Total generated tokens (TPS = delta/time)
#   vllm:prompt_tokens_total          — Total prompt tokens
#   vllm:num_requests_running         — Concurrent requests
#   vllm:kv_cache_usage_perc          — KV cache utilization

set -euo pipefail

source "$(dirname "$0")/config.sh" 2>/dev/null || true

LABEL="${1:-snapshot}"
OUTPUT_DIR="${2:-results_multiagent}"
METRICS_URL="${BASE_URL%/v1}/metrics"

mkdir -p "${OUTPUT_DIR}"
OUTFILE="${OUTPUT_DIR}/vllm_metrics_${LABEL}_$(date +%Y%m%d_%H%M%S).txt"

echo "[metrics] Scraping ${METRICS_URL} → ${OUTFILE}"
curl -sf "${METRICS_URL}" > "${OUTFILE}" 2>&1

if [ $? -eq 0 ]; then
  echo "[metrics] Saved. Key stats:"
  echo "  TTFT (count):  $(grep -c 'time_to_first_token_seconds_bucket' "${OUTFILE}" 2>/dev/null || echo 0)"
  echo "  TPOT (count):  $(grep -c 'inter_token_latency_seconds_bucket' "${OUTFILE}" 2>/dev/null || echo 0)"
  echo "  Gen tokens:    $(grep 'generation_tokens_total' "${OUTFILE}" | tail -1)"
  echo "  Requests now:  $(grep 'num_requests_running' "${OUTFILE}" | tail -1)"
  echo "  KV cache:      $(grep 'kv_cache_usage_perc' "${OUTFILE}" | tail -1)"
else
  echo "[metrics] ERROR: Could not reach ${METRICS_URL}"
fi
