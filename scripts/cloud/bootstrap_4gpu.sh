#!/bin/bash
# =============================================================================
# Bootstrap for 4GPU Vast.ai Instance
# =============================================================================
# Run this first after SSH into a new instance.
# Sets up: dirs, dataset, model, MAB deps.
#
# Usage:
#   # On Vast.ai instance (vLLM template):
#   git clone https://github.com/UCSD-JP/multi-agent-bench.git /data/multi-agent-bench
#   bash /data/multi-agent-bench/scripts/cloud/bootstrap_4gpu.sh
#
#   # Or if /data is too small, clone to /root:
#   git clone https://github.com/UCSD-JP/multi-agent-bench.git ~/multi-agent-bench
#   bash ~/multi-agent-bench/scripts/cloud/bootstrap_4gpu.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# --- Determine data root ---
# Vast.ai templates: /data may be small (10GB) or large depending on config.
# Try /data first, fallback to /root/data
if [ -d "/data" ] && [ "$(df /data --output=avail 2>/dev/null | tail -1)" -gt 20000000 ] 2>/dev/null; then
  DATA_ROOT="/data"
else
  DATA_ROOT="/root/data"
  mkdir -p "${DATA_ROOT}"
  echo "[INFO] /data too small, using ${DATA_ROOT}"
fi

echo "================================================================"
echo " Bootstrap — 4GPU NVLink Instance"
echo " Data root: ${DATA_ROOT}"
echo " MAB repo:  ${MAB_ROOT}"
echo "================================================================"

# --- 1) Directory structure ---
echo ""
echo "=== Step 1: Create directories ==="
mkdir -p "${DATA_ROOT}"/{models,datasets,results,logs,cache}
echo "[OK] Directories created"

# --- 2) GPU topology snapshot ---
echo ""
echo "=== Step 2: GPU Topology ==="
nvidia-smi -L
echo ""
nvidia-smi topo -m
echo ""
nvidia-smi nvlink --status 2>/dev/null || echo "(NVLink status unavailable)"

# Save snapshot
{
  date
  nvidia-smi -L
  nvidia-smi topo -m
  nvidia-smi nvlink --status 2>/dev/null || true
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
} > "${DATA_ROOT}/gpu_topology.txt"
echo "[OK] Saved to ${DATA_ROOT}/gpu_topology.txt"

# --- 3) Dataset ---
echo ""
echo "=== Step 3: Dataset ==="
DATASET_PATH="${DATA_ROOT}/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
DATASET_URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

if [ -f "${DATASET_PATH}" ]; then
  echo "[OK] Dataset exists ($(du -h "${DATASET_PATH}" | cut -f1))"
else
  echo "[INFO] Downloading ShareGPT V3..."
  wget -q --show-progress -O "${DATASET_PATH}" "${DATASET_URL}" || \
    curl -L -o "${DATASET_PATH}" "${DATASET_URL}"
  echo "[OK] Downloaded ($(du -h "${DATASET_PATH}" | cut -f1))"
fi

# --- 4) Model download ---
echo ""
echo "=== Step 4: Model ==="
export HF_HOME="${DATA_ROOT}/models"

# Check if model already available (Vast.ai template may pre-download)
MODEL_NAME="Qwen/Qwen3-Next-80B-A3B-Instruct"
MODEL_DIR="${HF_HOME}/hub/models--Qwen--Qwen3-Next-80B-A3B-Instruct"

if [ -d "${MODEL_DIR}" ] && [ "$(find "${MODEL_DIR}" -name "*.safetensors" 2>/dev/null | wc -l)" -gt 0 ]; then
  echo "[OK] Model already downloaded"
elif [ -d "/root/models/hub/models--Qwen--Qwen3-Next-80B-A3B-Instruct" ]; then
  echo "[OK] Model found in /root/models (Vast.ai cache)"
  export HF_HOME="/root/models"
else
  echo "[INFO] Model not found. It will be downloaded when vLLM starts."
  echo "       This may take 30-45 minutes on first run."
  echo "       Set HF_HOME=${HF_HOME} when starting vLLM."
fi

# --- 5) MAB dependencies ---
echo ""
echo "=== Step 5: MAB Dependencies ==="
cd "${MAB_ROOT}"
if pip list 2>/dev/null | grep -q autogen-agentchat; then
  echo "[OK] autogen already installed"
else
  echo "[INFO] Installing MAB requirements..."
  pip install -q -r requirements.txt 2>&1 | tail -3
  echo "[OK] Installed"
fi
# aiohttp for batch sweep
pip install -q aiohttp 2>/dev/null || true

# --- 6) Verify vLLM ---
echo ""
echo "=== Step 6: Verify vLLM ==="
python3 -c "import vllm; print(f'vLLM {vllm.__version__}')" 2>/dev/null || \
  echo "[WARN] vLLM not importable — install it or use Vast.ai vLLM template"

# --- 7) Summary ---
echo ""
echo "================================================================"
echo " Bootstrap Complete"
echo ""
echo " Next steps:"
echo "   1. Verify model is available (or let vLLM download on first start)"
echo "   2. Run experiment:"
echo "      bash ${MAB_ROOT}/scripts/cloud/run_4gpu_experiment.sh"
echo ""
echo " Environment:"
echo "   DATA_ROOT=${DATA_ROOT}"
echo "   HF_HOME=${HF_HOME}"
echo "   DATASET_PATH=${DATASET_PATH}"
echo "================================================================"
