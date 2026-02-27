#!/bin/bash
# =============================================================================
# Experiment Matrix: 2-GPU Server Configurations
# =============================================================================
# Models:
#   - Qwen3-Next-80B-A3B (MoE, 128 experts)
#   - Llama-3.1-70B      (Dense)
#
# Engines: vLLM, SGLang
# Configs: TP=2, DP=2+EP=2, DP=2 (dense only)
#
# Usage:
#   ./run_experiment_matrix.sh <experiment_id>
#
#   experiment_id:
#     # --- vLLM ---
#     1   vllm-qwen-tp2-fp8          (baseline, already have data)
#     2   vllm-qwen-tp2-fp16         (baseline, already have data)
#     3   vllm-qwen-dp2-ep2-fp8      (NEW: expert parallel)
#     4   vllm-qwen-dp2-ep2-fp16     (NEW: expert parallel)
#     5   vllm-llama70b-tp2-fp16     (NEW: dense model, TP=2)
#     6   vllm-llama70b-tp2-fp8      (NEW: dense model, TP=2, FP8)
#     # --- SGLang ---
#     7   sglang-qwen-tp2-fp8        (NEW: SGLang comparison)
#     8   sglang-qwen-tp2-fp16       (NEW: SGLang comparison)
#     9   sglang-qwen-ep2-fp8        (NEW: SGLang EP)
#     10  sglang-llama70b-tp2-fp16   (NEW: SGLang dense)
#
#   Or use preset groups:
#     vllm-all    Run experiments 1-6 (interactive, prompts between each)
#     sglang-all  Run experiments 7-10
#     all         Run everything
#
# Prerequisites:
#   - 2x H100 GPUs on paladin
#   - vLLM installed (for experiments 1-6)
#   - SGLang installed (for experiments 7-10): pip install sglang[all]
#   - Model weights downloaded
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# Model IDs
QWEN_FP8="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
QWEN_FP16="Qwen/Qwen3-Next-80B-A3B-Instruct"
LLAMA70B_FP16="meta-llama/Meta-Llama-3.1-70B-Instruct"
LLAMA70B_FP8="neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8"

# ─────────────────────────────────────────────────
# vLLM launch helper
# ─────────────────────────────────────────────────
launch_vllm() {
    local model="$1"
    local label="$2"
    shift 2
    local extra_flags="$@"

    echo ""
    echo "================================================================"
    echo "  vLLM: ${label}"
    echo "  Model: ${model}"
    echo "  Extra: ${extra_flags:-none}"
    echo "================================================================"
    echo ""

    python -m vllm.entrypoints.openai.api_server \
        --model "${model}" \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --gpu-memory-utilization 0.90 \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs 64 \
        --enable-prefix-caching \
        --enable-prompt-tokens-details \
        --enable-log-requests \
        ${extra_flags}
}

# ─────────────────────────────────────────────────
# SGLang launch helper
# ─────────────────────────────────────────────────
launch_sglang() {
    local model="$1"
    local label="$2"
    shift 2
    local extra_flags="$@"

    echo ""
    echo "================================================================"
    echo "  SGLang: ${label}"
    echo "  Model: ${model}"
    echo "  Extra: ${extra_flags:-none}"
    echo "================================================================"
    echo ""

    python3 -m sglang.launch_server \
        --model-path "${model}" \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --mem-fraction-static 0.88 \
        --context-length "${MAX_MODEL_LEN}" \
        --enable-metrics \
        ${extra_flags}
}

# ─────────────────────────────────────────────────
# Experiment definitions
# ─────────────────────────────────────────────────
run_experiment() {
    local exp_id="$1"

    case "$exp_id" in
        # =================== vLLM ===================
        1|vllm-qwen-tp2-fp8)
            launch_vllm "${QWEN_FP8}" "Qwen3-Next MoE TP=2 FP8" \
                --tensor-parallel-size 2
            ;;

        2|vllm-qwen-tp2-fp16)
            launch_vllm "${QWEN_FP16}" "Qwen3-Next MoE TP=2 FP16" \
                --tensor-parallel-size 2
            ;;

        3|vllm-qwen-dp2-ep2-fp8)
            launch_vllm "${QWEN_FP8}" "Qwen3-Next MoE DP=2 EP=2 FP8" \
                --tensor-parallel-size 1 \
                --data-parallel-size 2 \
                --data-parallel-size-local 2 \
                --enable-expert-parallel \
                --enable-eplb
            ;;

        4|vllm-qwen-dp2-ep2-fp16)
            launch_vllm "${QWEN_FP16}" "Qwen3-Next MoE DP=2 EP=2 FP16" \
                --tensor-parallel-size 1 \
                --data-parallel-size 2 \
                --data-parallel-size-local 2 \
                --enable-expert-parallel \
                --enable-eplb
            ;;

        5|vllm-llama70b-tp2-fp16)
            launch_vllm "${LLAMA70B_FP16}" "Llama-3.1-70B Dense TP=2 FP16" \
                --tensor-parallel-size 2
            ;;

        6|vllm-llama70b-tp2-fp8)
            launch_vllm "${LLAMA70B_FP8}" "Llama-3.1-70B Dense TP=2 FP8" \
                --tensor-parallel-size 2
            ;;

        # =================== SGLang ===================
        7|sglang-qwen-tp2-fp8)
            launch_sglang "${QWEN_FP8}" "Qwen3-Next MoE TP=2 FP8" \
                --tp 2
            ;;

        8|sglang-qwen-tp2-fp16)
            launch_sglang "${QWEN_FP16}" "Qwen3-Next MoE TP=2 FP16" \
                --tp 2
            ;;

        9|sglang-qwen-ep2-fp8)
            launch_sglang "${QWEN_FP8}" "Qwen3-Next MoE EP=2 FP8" \
                --tp 2 \
                --ep 2 \
                --enable-ep-moe
            ;;

        10|sglang-llama70b-tp2-fp16)
            launch_sglang "${LLAMA70B_FP16}" "Llama-3.1-70B Dense TP=2 FP16" \
                --tp 2
            ;;

        *)
            echo "Unknown experiment: $exp_id"
            echo ""
            echo "Available experiments:"
            echo "  1  vllm-qwen-tp2-fp8        Qwen3-Next MoE, TP=2, FP8 (baseline)"
            echo "  2  vllm-qwen-tp2-fp16       Qwen3-Next MoE, TP=2, FP16 (baseline)"
            echo "  3  vllm-qwen-dp2-ep2-fp8    Qwen3-Next MoE, DP=2+EP=2, FP8"
            echo "  4  vllm-qwen-dp2-ep2-fp16   Qwen3-Next MoE, DP=2+EP=2, FP16"
            echo "  5  vllm-llama70b-tp2-fp16   Llama-3.1-70B Dense, TP=2, FP16"
            echo "  6  vllm-llama70b-tp2-fp8    Llama-3.1-70B Dense, TP=2, FP8"
            echo "  7  sglang-qwen-tp2-fp8      SGLang Qwen3-Next, TP=2, FP8"
            echo "  8  sglang-qwen-tp2-fp16     SGLang Qwen3-Next, TP=2, FP16"
            echo "  9  sglang-qwen-ep2-fp8      SGLang Qwen3-Next, EP=2, FP8"
            echo "  10 sglang-llama70b-tp2-fp16  SGLang Llama-3.1-70B, TP=2, FP16"
            exit 1
            ;;
    esac
}

# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────
EXP="${1:?Usage: $0 <experiment_id|name>}"
run_experiment "$EXP"
