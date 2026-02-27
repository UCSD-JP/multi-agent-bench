#!/bin/bash
# =============================================================================
# NCCL AllReduce Microbenchmark (2GPU + 4GPU)
# =============================================================================
# Adapted from gpusim scripts/cloud/run_nccl_bench.sh for MAB cloud setup.
# Measures raw AllReduce bandwidth/latency to calibrate simulator comm model.
#
# Usage:
#   bash scripts/cloud/run_nccl_bench.sh
#
# Output:
#   ${RESULT_DIR}/nccl_bench/
#     nccl_allreduce_2gpu.txt
#     nccl_allreduce_4gpu.txt
#     nccl_allreduce_summary.json
#     topology.txt
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cloud_config.sh"

OUTDIR="${RESULT_DIR}/nccl_bench"
mkdir -p "${OUTDIR}"

echo "================================================================"
echo " NCCL AllReduce Microbenchmark"
echo " Output: ${OUTDIR}"
echo "================================================================"

# --- 0) Capture topology ---
echo ""
echo "=== Step 0: GPU Topology ==="
{
  echo "=== nvidia-smi topo -m ==="
  nvidia-smi topo -m 2>&1
  echo ""
  echo "=== NVLink status ==="
  nvidia-smi nvlink --status 2>&1 || echo "(no NVLink or unavailable)"
  echo ""
  echo "=== GPU list ==="
  nvidia-smi -L 2>&1
} | tee "${OUTDIR}/topology.txt"

# --- 1) Build nccl-tests if needed ---
echo ""
echo "=== Step 1: Build nccl-tests ==="
NCCL_DIR="/tmp/nccl-tests"
NCCL_BIN="${NCCL_DIR}/build/all_reduce_perf"

if [ -x "${NCCL_BIN}" ]; then
  echo "[OK] nccl-tests already built"
else
  echo "[INFO] Building nccl-tests..."
  rm -rf "${NCCL_DIR}"
  git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git "${NCCL_DIR}" 2>&1
  cd "${NCCL_DIR}" && make -j$(nproc) MPI=0 2>&1 | tail -3 && cd -
  [ -x "${NCCL_BIN}" ] || { echo "[ERROR] Build failed"; exit 1; }
  echo "[OK] Built"
fi

# --- 2) AllReduce — 2 GPU ---
echo ""
echo "=== Step 2: AllReduce — 2 GPU (0,1) ==="
CUDA_VISIBLE_DEVICES=0,1 "${NCCL_BIN}" \
  -b 1K -e 128M -f 2 -g 2 -n 100 -w 20 \
  2>&1 | tee "${OUTDIR}/nccl_allreduce_2gpu.txt"

# --- 3) AllReduce — 4 GPU ---
echo ""
echo "=== Step 3: AllReduce — 4 GPU (0,1,2,3) ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 "${NCCL_BIN}" \
  -b 1K -e 128M -f 2 -g 4 -n 100 -w 20 \
  2>&1 | tee "${OUTDIR}/nccl_allreduce_4gpu.txt"

# --- 4) Parse to JSON ---
echo ""
echo "=== Step 4: Summary ==="
OUTDIR="${OUTDIR}" python3 "${SCRIPT_DIR}/parse_nccl.py"

echo ""
echo "================================================================"
echo " NCCL Complete — ${OUTDIR}"
echo "================================================================"
