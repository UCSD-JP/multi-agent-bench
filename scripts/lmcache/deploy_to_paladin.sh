#!/bin/bash
# =============================================================================
# Deploy LMCache experiment scripts to Paladin
# =============================================================================
# Run from LOCAL machine. Copies scripts from this repo (multi-agent-bench)
# to Paladin and optionally installs dependencies.
#
# Usage:
#   bash scripts/lmcache/deploy_to_paladin.sh          # deploy scripts only
#   bash scripts/lmcache/deploy_to_paladin.sh --setup   # deploy + install deps
# =============================================================================
set -euo pipefail

PALADIN="jinpyo@paladin.ucsd.edu"
REMOTE_DIR="/mnt/raid0_ssd/jinpyo/lmcache_experiment"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Deploying LMCache experiment to Paladin ==="
echo "  Local:  ${SCRIPT_DIR}"
echo "  Remote: ${PALADIN}:${REMOTE_DIR}"

# Create remote directory
ssh "${PALADIN}" "mkdir -p ${REMOTE_DIR}/scripts"

# Copy experiment scripts
scp "${SCRIPT_DIR}/paladin_config.sh" "${PALADIN}:${REMOTE_DIR}/scripts/"
scp "${SCRIPT_DIR}/run_experiment.sh" "${PALADIN}:${REMOTE_DIR}/scripts/"
scp "${SCRIPT_DIR}/benchmark_prefix_workload.py" "${PALADIN}:${REMOTE_DIR}/scripts/"
scp "${SCRIPT_DIR}/analyze_results.py" "${PALADIN}:${REMOTE_DIR}/scripts/"
scp "${SCRIPT_DIR}/analyze_traces.py" "${PALADIN}:${REMOTE_DIR}/scripts/"

echo "[OK] Scripts deployed."

# Optional: install dependencies and clone traces
if [[ "${1:-}" == "--setup" ]]; then
  echo ""
  echo "=== Setting up Paladin environment ==="

  ssh "${PALADIN}" << 'REMOTE_EOF'
    set -e

    source /home/jinpyo/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate vllm 2>/dev/null || true

    echo "=== Python/vLLM versions ==="
    python3 --version
    python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')" 2>/dev/null || echo "vLLM: not found"

    echo ""
    echo "=== Installing httpx ==="
    pip install httpx 2>/dev/null || pip3 install httpx 2>/dev/null || echo "[WARN] httpx install failed"

    echo ""
    echo "=== Checking lmcache ==="
    if python3 -c "import lmcache; print(f'lmcache: {lmcache.__version__}')" 2>/dev/null; then
      echo "[OK] lmcache already installed"
    else
      echo "[INFO] lmcache not installed. Install with: pip install lmcache"
      echo "       (Optional — native offloading works without it)"
    fi

    echo ""
    echo "=== Checking --kv-offloading-backend support ==="
    python3 -c "
from vllm.config import CacheConfig
import inspect
sig = inspect.signature(CacheConfig.__init__)
params = list(sig.parameters.keys())
if 'kv_offloading_backend' in params or 'kv_offloading_size' in params:
    print('[OK] kv_offloading_backend/size supported')
else:
    from vllm.config import KVTransferConfig
    print('[OK] KVTransferConfig available (use --kv-transfer-config)')
" 2>/dev/null || echo "[WARN] Could not verify KV offloading support. Check vLLM version."

    echo ""
    echo "=== Cloning LMCache agent traces ==="
    TRACE_DIR="/mnt/raid0_ssd/jinpyo/lmcache-agent-trace"
    if [ -d "${TRACE_DIR}/.git" ]; then
      echo "[OK] Already cloned at ${TRACE_DIR}"
      git -C "${TRACE_DIR}" pull --ff-only 2>/dev/null || true
    else
      git clone --depth 1 https://github.com/LMCache/lmcache-agent-trace.git "${TRACE_DIR}"
      echo "[OK] Cloned to ${TRACE_DIR}"
    fi

    echo ""
    echo "=== Setup complete ==="
    echo "  Scripts: /mnt/raid0_ssd/jinpyo/lmcache_experiment/scripts/"
    echo "  Traces:  /mnt/raid0_ssd/jinpyo/lmcache-agent-trace/"
    echo ""
    echo "  To run experiment:"
    echo "    conda activate vllm"
    echo "    cd /mnt/raid0_ssd/jinpyo/lmcache_experiment"
    echo "    bash scripts/run_experiment.sh"
REMOTE_EOF

fi

echo ""
echo "=== Deploy complete ==="
echo ""
echo "To run on Paladin:"
echo "  ssh ${PALADIN}"
echo "  conda activate vllm"
echo "  cd ${REMOTE_DIR}"
echo "  bash scripts/run_experiment.sh"
echo ""
echo "To collect results:"
echo "  scp -r ${PALADIN}:${REMOTE_DIR}/results/ results/lmcache_paladin/"
