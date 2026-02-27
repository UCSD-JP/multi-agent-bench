#!/bin/bash
# =============================================================================
# Workload Client — runs on jpserver, sends workloads to paladin
#
# Watches for READY signal from paladin, reads config, runs benchmark,
# then signals WORKLOAD_DONE.
#
# Usage:
#   ./scripts/nsys_workload_client.sh
# =============================================================================

set -euo pipefail

REMOTE_HOST="paladin.ucsd.edu"
REMOTE_USER="jinpyo"
REMOTE_OUTDIR="/mnt/raid0_ssd/jinpyo/nsys_profiles"
SIGNAL_DIR="$REMOTE_OUTDIR/signals"

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_RESULTS="$LOCAL_DIR/results_multiagent/profiling"
DATASET="/home/jp/CXL_project/old_heimdall/benchmark/llm_bench/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"

log() { echo "[$(date +%H:%M:%S)] $*"; }

ssh_cmd() {
    ssh "$REMOTE_USER@$REMOTE_HOST" "$@"
}

LAST_SEQ=0

wait_for_ready() {
    log "Waiting for paladin READY signal (seq > $LAST_SEQ)..."
    while true; do
        # Find any READY_N where N > LAST_SEQ
        local seq=$(ssh_cmd "ls $SIGNAL_DIR/READY_* 2>/dev/null | sed 's/.*READY_//' | sort -n | tail -1" 2>/dev/null)
        if [ -n "$seq" ] && [ "$seq" -gt "$LAST_SEQ" ] 2>/dev/null; then
            LAST_SEQ="$seq"
            return 0
        fi
        sleep 3
    done
}

signal_done() {
    ssh_cmd "touch $SIGNAL_DIR/DONE_$LAST_SEQ"
    log "Sent DONE_$LAST_SEQ signal"
}

read_config() {
    ssh_cmd "cat $SIGNAL_DIR/CONFIG_$LAST_SEQ 2>/dev/null"
}

run_benchmark() {
    local engine="$1"
    local preset="$2"
    local framework="$3"
    local concurrency="$4"
    local profile_name="$5"

    # Determine model from preset
    if [[ "$preset" == *fp8* ]]; then
        local model="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
    else
        local model="Qwen/Qwen3-Next-80B-A3B-Instruct"
    fi

    local output_dir="$LOCAL_RESULTS/$profile_name"
    mkdir -p "$output_dir"

    log "Running benchmark: $framework c=$concurrency → $output_dir"
    python "$LOCAL_DIR/benchmark_agentic.py" \
        --framework "$framework" \
        --model "$model" \
        --dataset_path "$DATASET" \
        --base_url "http://$REMOTE_HOST:8000/v1" \
        --api_key EMPTY \
        --tasks 48 \
        --concurrency "$concurrency" \
        --task_concurrency "$concurrency" \
        --executors 2 \
        --output_dir "$output_dir"
}

# =============================================================================
# MAIN LOOP
# =============================================================================

log "Workload client started. Watching for READY signals from paladin..."
log "Press Ctrl+C to stop."

PROFILE_COUNT=0

while true; do
    wait_for_ready

    # Read config
    CONFIG=$(read_config)
    log "Config received: $CONFIG"

    # Parse config: engine=X preset=Y framework=Z concurrency=N
    ENGINE=$(echo "$CONFIG" | grep -oP 'engine=\K\S+')
    PRESET=$(echo "$CONFIG" | grep -oP 'preset=\K\S+')
    FRAMEWORK=$(echo "$CONFIG" | grep -oP 'framework=\K\S+')
    CONCURRENCY=$(echo "$CONFIG" | grep -oP 'concurrency=\K\S+')

    # Derive profile name from config
    PROFILE_NAME="${ENGINE}_${PRESET}_${FRAMEWORK}_c${CONCURRENCY}"

    log "=== Workload $((++PROFILE_COUNT)): $PROFILE_NAME ==="
    run_benchmark "$ENGINE" "$PRESET" "$FRAMEWORK" "$CONCURRENCY" "$PROFILE_NAME"

    # Signal completion
    signal_done
    log "=== Workload $PROFILE_COUNT complete ==="
    sleep 2
done
