#!/bin/bash
# =============================================================================
# Nsight Profiling Automation for Paladin
# Runs ENTIRELY on paladin (same shell context for nsys launch/start/stop)
#
# Usage (on paladin, or via SSH tmux):
#   /mnt/raid0_ssd/jinpyo/scripts/nsys_profiling_paladin.sh <scenario> <engine>
#
# Scenarios:
#   scenario1  - EP2 vs TP2 comm pattern (2 server starts, 2 profiles)
#   scenario4  - Framework burst pattern (1 server start, 2 profiles)
#   scenario2  - Batch scaling (1 server start, 4 profiles)
#
# Engine: vllm or sglang (default: vllm)
#
# Synchronization with jpserver:
#   After nsys start, creates /mnt/raid0_ssd/jinpyo/nsys_profiles/READY
#   Waits for /mnt/raid0_ssd/jinpyo/nsys_profiles/WORKLOAD_DONE
#   After nsys stop, removes both signal files
# =============================================================================

set -euo pipefail

SCENARIO="${1:-scenario1}"
ENGINE="${2:-vllm}"

# --- Setup ---
source /home/jinpyo/miniconda3/etc/profile.d/conda.sh
conda activate vllm
export TMPDIR=/mnt/raid0_ssd/jinpyo/tmp
export HF_HOME=/mnt/raid0_ssd/huggingface

OUTDIR=/mnt/raid0_ssd/jinpyo/nsys_profiles
SIGNAL_DIR="$OUTDIR/signals"
mkdir -p "$OUTDIR" "$SIGNAL_DIR"

SESSION_NAME="prof_sess"
SERVER_DIR="/home/jinpyo/llm_serving"
SEQ=0
# Set ENFORCE_EAGER=1 to skip CUDA graph capture (faster startup but less realistic)
# Default 0: use CUDA graphs (production-representative profiling)
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

cleanup_nsys() {
    # Kill any stale nsys sessions from previous failed runs
    nsys shutdown --session="$SESSION_NAME" 2>/dev/null || true
    # Also kill any leftover GPU processes
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 2
}

cleanup_signals() {
    rm -f "$SIGNAL_DIR"/*
}

start_server_with_nsys() {
    local preset="$1"
    log "Starting server: ENGINE=$ENGINE preset=$preset ENFORCE_EAGER=$ENFORCE_EAGER with nsys session=$SESSION_NAME"
    nsys launch --session="$SESSION_NAME" --trace=cuda,nvtx,osrt \
        bash -c "cd $SERVER_DIR && ENGINE=$ENGINE ENFORCE_EAGER=$ENFORCE_EAGER ./run_server.sh $preset" &
    NSYS_PID=$!
    log "nsys launch PID: $NSYS_PID"

    # Wait for server ready (up to 30min for nsys + CUDA graph capture)
    log "Waiting for server..."
    for i in $(seq 1 360); do
        if curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
            log "Server ready at attempt $i (${i}*5=${((i*5))}s)"
            return 0
        fi
        sleep 5
    done
    log "ERROR: Server did not start in 1800s"
    return 1
}

stop_server() {
    log "Stopping server..."
    # Kill the nsys-launched process tree
    kill "$NSYS_PID" 2>/dev/null || true
    sleep 3
    # Force kill any remaining GPU processes
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 3
    log "Server stopped"
}

run_profiled_workload() {
    local profile_name="$1"
    local config_info="$2"

    SEQ=$((SEQ + 1))
    log "=== Profiling: $profile_name (seq=$SEQ) ==="

    # Clean any stale signals
    cleanup_signals

    # Start collection
    nsys start --session="$SESSION_NAME" -o "$OUTDIR/$profile_name" 2>&1
    log "nsys collection started"

    # Signal jpserver that we're ready for workload (seq-tagged)
    echo "$config_info" > "$SIGNAL_DIR/CONFIG_$SEQ"
    echo "$SEQ" > "$SIGNAL_DIR/READY_$SEQ"
    log "Waiting for workload to complete (signal: DONE_$SEQ) ..."

    # Wait for matching DONE signal
    while [ ! -f "$SIGNAL_DIR/DONE_$SEQ" ]; do
        sleep 2
    done
    log "Workload done signal received (seq=$SEQ)"

    # Stop collection
    nsys stop --session="$SESSION_NAME" 2>&1
    log "nsys collection stopped"

    # Verify file
    if [ -f "$OUTDIR/${profile_name}.nsys-rep" ]; then
        local size=$(du -h "$OUTDIR/${profile_name}.nsys-rep" | cut -f1)
        log "Profile saved: ${profile_name}.nsys-rep ($size)"
    else
        log "WARNING: Profile file not found!"
    fi

    # Cleanup signals
    cleanup_signals
    sleep 3
}

# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================

run_scenario1() {
    log "=========================================="
    log "SCENARIO 1: EP2 vs TP2 Communication"
    log "Engine: $ENGINE"
    log "=========================================="

    # --- Config A: TP2-FP16 ---
    start_server_with_nsys "tp2-fp16"
    run_profiled_workload "s1_${ENGINE}_tp2_fp16_c64" "engine=$ENGINE preset=tp2-fp16 framework=autogen concurrency=64"
    stop_server
    sleep 10

    # --- Config B: EP2-FP16 ---
    start_server_with_nsys "dp2-ep2-fp16"
    run_profiled_workload "s1_${ENGINE}_ep2_fp16_c64" "engine=$ENGINE preset=dp2-ep2-fp16 framework=autogen concurrency=64"
    stop_server

    log "Scenario 1 complete. Profiles:"
    ls -lh "$OUTDIR"/s1_*.nsys-rep 2>/dev/null
}

run_scenario4() {
    log "=========================================="
    log "SCENARIO 4: Framework Burst Pattern"
    log "Engine: $ENGINE, Preset: tp2-fp16"
    log "=========================================="

    start_server_with_nsys "tp2-fp16"

    # --- AutoGen (sequential) ---
    run_profiled_workload "s4_${ENGINE}_tp2fp16_autogen_c32" "engine=$ENGINE preset=tp2-fp16 framework=autogen concurrency=32"

    # --- A2A (parallel) ---
    run_profiled_workload "s4_${ENGINE}_tp2fp16_a2a_c32" "engine=$ENGINE preset=tp2-fp16 framework=a2a concurrency=32"

    stop_server

    log "Scenario 4 complete. Profiles:"
    ls -lh "$OUTDIR"/s4_*.nsys-rep 2>/dev/null
}

run_scenario2() {
    log "=========================================="
    log "SCENARIO 2: Batch Scaling"
    log "Engine: $ENGINE, Preset: tp2-fp16"
    log "=========================================="

    start_server_with_nsys "tp2-fp16"

    for conc in 1 8 32 64; do
        run_profiled_workload "s2_${ENGINE}_tp2fp16_autogen_c${conc}" "engine=$ENGINE preset=tp2-fp16 framework=autogen concurrency=$conc"
    done

    stop_server

    log "Scenario 2 complete. Profiles:"
    ls -lh "$OUTDIR"/s2_*.nsys-rep 2>/dev/null
}

# --- HELPER: Send simple batch requests (no agentic benchmark needed) ---
send_simple_requests() {
    local batch="$1"
    local input_len="$2"
    local output_len="${3:-32}"
    local rounds="${4:-4}"
    local port=8000

    # Determine model name from running server
    local model
    model=$(curl -sf http://localhost:$port/v1/models 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")

    # Generate prompt of approximately input_len tokens
    local word_count=$((input_len * 4 / 5))
    local prompt
    prompt=$(python3 -c "print(' '.join(['hello'] * ${word_count}))")

    log "Sending simple workload: batch=$batch input≈${input_len} output=$output_len rounds=$rounds model=$model"

    # Warmup
    for i in 1 2; do
        curl -s "http://localhost:$port/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"$model\", \"prompt\": \"$prompt\", \"max_tokens\": $output_len, \"temperature\": 0}" \
            > /dev/null 2>&1 &
    done
    wait
    log "Warmup done"

    # Profiling rounds
    for round in $(seq 1 $rounds); do
        local pids=()
        for i in $(seq 1 $batch); do
            curl -s "http://localhost:$port/v1/completions" \
                -H "Content-Type: application/json" \
                -d "{\"model\": \"$model\", \"prompt\": \"$prompt\", \"max_tokens\": $output_len, \"temperature\": 0}" \
                > /dev/null 2>&1 &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
        log "  Round $round/$rounds done"
    done
    log "Simple workload complete"
}

# --- HELPER: Self-contained profiled workload (no signal coordination needed) ---
run_self_profiled_workload() {
    local profile_name="$1"
    local batch="$2"
    local input_len="$3"
    local output_len="${4:-32}"

    SEQ=$((SEQ + 1))
    log "=== Self-profiling: $profile_name (batch=$batch i=$input_len o=$output_len) ==="

    nsys start --session="$SESSION_NAME" -o "$OUTDIR/$profile_name" 2>&1
    log "nsys collection started"

    send_simple_requests "$batch" "$input_len" "$output_len" 4

    nsys stop --session="$SESSION_NAME" 2>&1
    log "nsys collection stopped"

    if [ -f "$OUTDIR/${profile_name}.nsys-rep" ]; then
        local size=$(du -h "$OUTDIR/${profile_name}.nsys-rep" | cut -f1)
        log "Profile saved: ${profile_name}.nsys-rep ($size)"
    else
        log "WARNING: Profile file not found!"
    fi
    sleep 3
}

run_scenario_b8() {
    log "=========================================="
    log "SCENARIO B8: Batch=8 Kernel Decomposition"
    log "Engine: $ENGINE"
    log "=========================================="

    # --- FP8 TP2, batch=8, input=128 (agentic typical) ---
    start_server_with_nsys "tp2-fp8"
    run_self_profiled_workload "b8_${ENGINE}_tp2fp8_b8_i128" 8 128
    # --- FP8 TP2, batch=8, input=2048 (longer context) ---
    run_self_profiled_workload "b8_${ENGINE}_tp2fp8_b8_i2048" 8 2048
    # --- FP8 TP2, batch=1, input=128 (baseline) ---
    run_self_profiled_workload "b8_${ENGINE}_tp2fp8_b1_i128" 1 128
    stop_server
    sleep 10

    # --- FP16 TP2, batch=8, input=128 (comparison) ---
    start_server_with_nsys "tp2-fp16"
    run_self_profiled_workload "b8_${ENGINE}_tp2fp16_b8_i128" 8 128
    stop_server

    log "Scenario B8 complete. Profiles:"
    ls -lh "$OUTDIR"/b8_*.nsys-rep 2>/dev/null
}

run_scenario_ep2fp8() {
    log "=========================================="
    log "SCENARIO EP2-FP8: All-to-All Kernel Decomposition"
    log "Engine: $ENGINE"
    log "=========================================="

    # --- TP2 FP8 ---
    start_server_with_nsys "tp2-fp8"
    run_profiled_workload "ep2fp8_${ENGINE}_tp2_fp8_c64" "engine=$ENGINE preset=tp2-fp8 framework=autogen concurrency=64"
    stop_server
    sleep 10

    # --- EP2 FP8 ---
    start_server_with_nsys "dp2-ep2-fp8"
    run_profiled_workload "ep2fp8_${ENGINE}_ep2_fp8_c64" "engine=$ENGINE preset=dp2-ep2-fp8 framework=autogen concurrency=64"
    stop_server

    log "Scenario EP2-FP8 complete. Profiles:"
    ls -lh "$OUTDIR"/ep2fp8_*.nsys-rep 2>/dev/null
}

run_scenario_fp8_scaling() {
    log "=========================================="
    log "SCENARIO FP8-SCALE: FP8 Batch Scaling"
    log "Engine: $ENGINE, Preset: tp2-fp8"
    log "=========================================="

    start_server_with_nsys "tp2-fp8"

    for conc in 1 8 32 64; do
        run_profiled_workload "fp8s_${ENGINE}_tp2fp8_autogen_c${conc}" "engine=$ENGINE preset=tp2-fp8 framework=autogen concurrency=$conc"
    done

    stop_server

    log "Scenario FP8-SCALE complete. Profiles:"
    ls -lh "$OUTDIR"/fp8s_*.nsys-rep 2>/dev/null
}

# =============================================================================
# MAIN
# =============================================================================

cleanup_signals
cleanup_nsys
log "Starting scenario: $SCENARIO (engine: $ENGINE)"

case "$SCENARIO" in
    scenario1) run_scenario1 ;;
    scenario4) run_scenario4 ;;
    scenario2) run_scenario2 ;;
    b8)        run_scenario_b8 ;;
    ep2fp8)    run_scenario_ep2fp8 ;;
    fp8scale)  run_scenario_fp8_scaling ;;
    all)
        run_scenario1
        sleep 15
        run_scenario4
        sleep 15
        run_scenario2
        ;;
    fp8_self)
        # Self-contained FP8 profiling (no signal coordination needed)
        log "==========================================="
        log "FP8 SELF-CONTAINED: b=1/8/32/64 profiling"
        log "==========================================="
        start_server_with_nsys "tp2-fp8"
        for batch in 1 8 32 64; do
            run_self_profiled_workload "fp8self_${ENGINE}_tp2fp8_b${batch}_i128" "$batch" 128
        done
        stop_server
        log "FP8 self-contained complete."
        ls -lh "$OUTDIR"/fp8self_*.nsys-rep 2>/dev/null
        ;;
    all_fp8)
        run_scenario_b8
        sleep 15
        run_scenario_ep2fp8
        sleep 15
        run_scenario_fp8_scaling
        ;;
    *)
        echo "Unknown scenario: $SCENARIO"
        echo "Usage: $0 <scenario> [vllm|sglang]"
        echo ""
        echo "Existing (FP16):  scenario1 | scenario2 | scenario4 | all"
        echo "New (FP8/b8):     b8 | ep2fp8 | fp8scale | all_fp8"
        exit 1
        ;;
esac

log "All done!"
ls -lh "$OUTDIR"/*.nsys-rep 2>/dev/null
