#!/bin/bash
# Run the 4 remaining TP2 agentic sweeps (LangGraph + A2A for FP8 and FP16)
#
# Prerequisites:
#   1. vLLM server running with matching preset on paladin
#   2. For FP8 sweeps:  ./run_server.sh tp2-fp8
#      For FP16 sweeps: ./run_server.sh tp2-fp16
#
# Usage:
#   # Run all 4 sweeps (requires server restart between FP8 and FP16)
#   ./run_remaining_tp2_sweeps.sh
#
#   # Run only FP8 sweeps (2 sweeps, same server)
#   ./run_remaining_tp2_sweeps.sh fp8
#
#   # Run only FP16 sweeps (2 sweeps, same server)
#   ./run_remaining_tp2_sweeps.sh fp16
#
# Already completed:
#   TP1-FP8:  autogen/langgraph/a2a  (c=1,4,8,16,32)
#   TP2-FP8:  autogen                (c=1,8,32)
#   TP2-FP16: autogen                (c=1,8,32)
#
# Missing (this script):
#   TP2-FP8:  langgraph, a2a         (c=1,8,32)
#   TP2-FP16: langgraph, a2a         (c=1,8,32)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${1:-all}"  # all | fp8 | fp16

run_sweep() {
    local preset="$1"
    local framework="$2"

    echo ""
    echo "================================================================"
    echo "  SWEEP: ${preset} / ${framework}"
    echo "================================================================"

    PRESET="${preset}" BENCH_FRAMEWORK="${framework}" \
        "${SCRIPT_DIR}/run_conc_sweep.sh"

    echo "[done] ${preset} / ${framework} complete"
    echo ""
}

if [[ "$MODE" == "all" || "$MODE" == "fp8" ]]; then
    echo "=========================================="
    echo "  TP2-FP8 sweeps (server must be tp2-fp8)"
    echo "=========================================="
    echo ""
    echo "Verify: server is running with ./run_server.sh tp2-fp8"
    echo "Press Enter to continue (or Ctrl+C to abort)..."
    read -r

    run_sweep tp2-fp8 langgraph
    run_sweep tp2-fp8 a2a
fi

if [[ "$MODE" == "all" ]]; then
    echo ""
    echo "=========================================="
    echo "  FP8 sweeps done. Now need FP16 server."
    echo "=========================================="
    echo ""
    echo "Please restart server with: ./run_server.sh tp2-fp16"
    echo "Press Enter when server is ready (or Ctrl+C to abort)..."
    read -r
fi

if [[ "$MODE" == "all" || "$MODE" == "fp16" ]]; then
    echo "============================================"
    echo "  TP2-FP16 sweeps (server must be tp2-fp16)"
    echo "============================================"
    echo ""
    echo "Verify: server is running with ./run_server.sh tp2-fp16"
    echo "Press Enter to continue (or Ctrl+C to abort)..."
    read -r

    run_sweep tp2-fp16 langgraph
    run_sweep tp2-fp16 a2a
fi

echo ""
echo "=========================================="
echo "  All remaining sweeps complete!"
echo "=========================================="
echo ""
echo "Results:"
for p in tp2-fp8 tp2-fp16; do
    for f in langgraph a2a; do
        d="results_multiagent/sweep_${p}_${f}"
        if [ -d "$d" ]; then
            echo "  [OK] ${d}/"
        else
            echo "  [--] ${d}/ (not found)"
        fi
    done
done
