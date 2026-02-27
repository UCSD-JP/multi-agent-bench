#!/usr/bin/env python3
"""
Automated Experiment Runner — Local orchestration via SSH + tmux.

Manages the full lifecycle:
  1. SSH to paladin, start server in tmux session
  2. Wait for server health check
  3. Optionally attach nsys profiler to running server
  4. Run workload (benchmark) from local
  5. Stop profiler, collect results
  6. Kill server

Usage:
  # Basic sweep (no profiling)
  python scripts/run_experiment.py --preset tp2-fp8 --engine sglang --sweep

  # Single run with nsys profiling
  python scripts/run_experiment.py --preset tp2-fp8 --engine sglang \\
      --framework autogen --concurrency 8 --profile

  # Profile EP2 vs TP2 comparison
  python scripts/run_experiment.py --compare ep2-vs-tp2

  # Just start server (keep running for manual testing)
  python scripts/run_experiment.py --preset tp2-fp8 --engine vllm --start-only

  # Custom SSH target
  python scripts/run_experiment.py --host myserver --user jp --preset tp2-fp8 --sweep
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results_multiagent"

# ─── SSH / Remote Configuration ───

DEFAULT_HOST = "paladin.ucsd.edu"
DEFAULT_USER = os.environ.get("SSH_USER", "jinpyo")
DEFAULT_SSH_KEY = os.environ.get("SSH_KEY", "")
REMOTE_PROJECT_DIR = "/home/jinpyo/llm_serving"
REMOTE_PROFILE_DIR = "/mnt/raid0_ssd/jinpyo/nsys_profiles"
CONDA_INIT = "source /home/jinpyo/miniconda3/etc/profile.d/conda.sh"

TMUX_SERVER_SESSION = "exp_server"
TMUX_PROFILER_SESSION = "exp_profiler"


def ssh_cmd(host, user, command, key=None, timeout=30):
    """Execute a command on remote host via SSH."""
    ssh_args = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]
    if key:
        ssh_args += ["-i", key]
    ssh_args += [f"{user}@{host}", command]
    try:
        result = subprocess.run(ssh_args, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "SSH command timed out"


def ssh_cmd_fire_and_forget(host, user, command, key=None):
    """Execute SSH command without waiting for completion."""
    ssh_args = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]
    if key:
        ssh_args += ["-i", key]
    ssh_args += [f"{user}@{host}", command]
    subprocess.Popen(ssh_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def check_ssh(host, user, key=None):
    """Verify SSH connectivity."""
    rc, out, err = ssh_cmd(host, user, "echo ok", key=key)
    if rc != 0:
        print(f"[ERROR] Cannot SSH to {user}@{host}: {err}")
        sys.exit(1)
    print(f"[ssh] Connected to {user}@{host}")


# ─── tmux Helpers ───

def tmux_has_session(host, user, session_name, key=None):
    rc, out, _ = ssh_cmd(host, user, f"tmux has-session -t {session_name} 2>/dev/null && echo yes || echo no", key=key)
    return out.strip() == "yes"


def tmux_kill_session(host, user, session_name, key=None):
    if tmux_has_session(host, user, session_name, key=key):
        ssh_cmd(host, user, f"tmux kill-session -t {session_name}", key=key)
        print(f"[tmux] Killed session: {session_name}")


def tmux_start_session(host, user, session_name, command, key=None, env_vars=None):
    """Start a detached tmux session running the given command."""
    tmux_kill_session(host, user, session_name, key=key)
    env_prefix = ""
    if env_vars:
        env_prefix = " ".join(f"{k}={v}" for k, v in env_vars.items()) + " "
    full_cmd = f"tmux new-session -d -s {session_name} '{env_prefix}{command}'"
    rc, out, err = ssh_cmd(host, user, full_cmd, key=key)
    if rc != 0:
        print(f"[ERROR] Failed to start tmux session '{session_name}': {err}")
        return False
    print(f"[tmux] Started session '{session_name}': {command[:80]}...")
    return True


# ─── Server Management ───

def start_server(host, user, preset, engine, key=None, extra_env=None):
    """Start the serving engine in a tmux session on the remote host."""
    env_vars = {
        "ENGINE": engine,
        "TMPDIR": "/mnt/raid0_ssd/jinpyo/tmp",
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
    }
    if extra_env:
        env_vars.update(extra_env)

    # Conda env: sglang for sglang engine, vllm for vllm engine
    conda_env = "sglang" if engine == "sglang" else "vllm"
    server_cmd = f"{CONDA_INIT} && conda activate {conda_env} && cd {REMOTE_PROJECT_DIR} && ./run_server.sh {preset}"
    return tmux_start_session(host, user, TMUX_SERVER_SESSION, server_cmd, key=key, env_vars=env_vars)


def wait_for_server(host, port=8000, timeout=600, interval=10):
    """Poll server health endpoint until ready."""
    url = f"http://{host}:{port}/v1/models"
    print(f"[wait] Waiting for server at {url} (timeout={timeout}s)")
    start = time.time()
    while time.time() - start < timeout:
        try:
            result = subprocess.run(
                ["curl", "-sf", url], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                model = data.get("data", [{}])[0].get("id", "unknown")
                elapsed = time.time() - start
                print(f"[wait] Server ready in {elapsed:.0f}s. Model: {model}")
                return True, model
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
            pass
        remaining = timeout - (time.time() - start)
        print(f"[wait] Not ready yet... ({remaining:.0f}s remaining)")
        time.sleep(interval)
    print(f"[ERROR] Server did not start within {timeout}s")
    return False, None


def stop_server(host, user, key=None):
    """Kill the server tmux session."""
    tmux_kill_session(host, user, TMUX_SERVER_SESSION, key=key)


# ─── Nsight Profiler (attach mode) ───

def get_server_pid(host, user, engine, key=None):
    """Find the main server process PID on the remote host."""
    if engine == "vllm":
        grep_pattern = "vllm.entrypoints.openai.api_server"
    else:
        grep_pattern = "sglang.launch_server"
    # Get the parent process (not workers)
    cmd = f"pgrep -f '{grep_pattern}' | head -1"
    rc, out, _ = ssh_cmd(host, user, cmd, key=key)
    if rc == 0 and out.strip():
        return int(out.strip())
    return None


def start_profiler(host, user, engine, output_name, duration=120, key=None):
    """Attach nsys profiler to the running server process."""
    pid = get_server_pid(host, user, engine, key=key)
    if not pid:
        print("[WARN] Could not find server PID. Trying to profile all CUDA activity instead.")
        # Fallback: profile by command pattern
        nsys_cmd = (
            f"nsys profile "
            f"--sample=none --trace=cuda,nvtx,osrt "
            f"--duration={duration} "
            f"--output={REMOTE_PROFILE_DIR}/{output_name} "
            f"--force-overwrite=true "
            f"--delay=5 "
            f"sleep {duration + 10}"
        )
    else:
        print(f"[nsys] Found server PID: {pid}")
        nsys_cmd = (
            f"nsys profile "
            f"--sample=none --trace=cuda,nvtx,osrt "
            f"--duration={duration} "
            f"--output={REMOTE_PROFILE_DIR}/{output_name} "
            f"--force-overwrite=true "
            f"--attach-target={pid} "
            f"sleep {duration + 10}"
        )

    # Ensure output directory exists
    ssh_cmd(host, user, f"mkdir -p {REMOTE_PROFILE_DIR}", key=key)

    # Start profiler in tmux
    return tmux_start_session(host, user, TMUX_PROFILER_SESSION, nsys_cmd, key=key)


def stop_profiler(host, user, key=None):
    """Stop the nsys profiler gracefully."""
    # Send SIGINT to nsys to finalize the profile
    ssh_cmd(host, user, f"tmux send-keys -t {TMUX_PROFILER_SESSION} C-c", key=key)
    time.sleep(3)
    tmux_kill_session(host, user, TMUX_PROFILER_SESSION, key=key)
    print("[nsys] Profiler stopped.")


def export_nsys_sqlite(host, user, profile_name, key=None):
    """Export nsys-rep to sqlite on the remote host."""
    nsys_rep = f"{REMOTE_PROFILE_DIR}/{profile_name}.nsys-rep"
    sqlite_out = f"{REMOTE_PROFILE_DIR}/{profile_name}.sqlite"
    cmd = f"nsys export --type=sqlite --output={sqlite_out} {nsys_rep} 2>/dev/null || echo 'export failed'"
    rc, out, err = ssh_cmd(host, user, cmd, key=key, timeout=120)
    if rc == 0 and "failed" not in out:
        print(f"[nsys] Exported: {sqlite_out}")
        return sqlite_out
    else:
        print(f"[WARN] nsys export failed: {err}")
        return None


# ─── Workload Runner ───

def run_benchmark(host, port, preset, framework, concurrency, output_dir, tasks=48, executors=2):
    """Run the agentic benchmark locally against the remote server."""
    # Determine model from preset
    if "fp8" in preset:
        model = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
    else:
        model = "Qwen/Qwen3-Next-80B-A3B-Instruct"

    base_url = f"http://{host}:{port}/v1"
    dataset_path = str(PROJECT_DIR / "datasets" / "ShareGPT_V3_unfiltered_cleaned_split.json")
    # Fallback dataset path
    if not os.path.exists(dataset_path):
        dataset_path = "/home/jp/CXL_project/old_heimdall/benchmark/llm_bench/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, str(PROJECT_DIR / "benchmark_agentic.py"),
        "--framework", framework,
        "--model", model,
        "--dataset_path", dataset_path,
        "--base_url", base_url,
        "--api_key", "EMPTY",
        "--tasks", str(tasks),
        "--concurrency", str(concurrency),
        "--task_concurrency", str(concurrency),
        "--executors", str(executors),
        "--output_dir", output_dir,
    ]

    print(f"\n[bench] Running: {framework} c={concurrency}")
    print(f"[bench] Output: {output_dir}")
    result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    return result.returncode == 0


def run_sweep(host, port, preset, engine, frameworks, concurrencies, key=None):
    """Run full sweep across frameworks and concurrency levels."""
    results = {}
    for fw in frameworks:
        for conc in concurrencies:
            output_name = f"{engine}-{preset}"
            output_dir = str(RESULTS_DIR / f"sweep_{output_name}_{fw}" / f"{fw}_c{conc}")
            success = run_benchmark(host, port, preset, fw, conc, output_dir)
            results[(fw, conc)] = success
            print(f"[sweep] {fw} c={conc}: {'OK' if success else 'FAILED'}")
            time.sleep(5)  # cooldown
    return results


# ─── Profiled Run ───

def run_profiled(host, user, port, preset, engine, framework, concurrency,
                 profile_name, duration=120, key=None):
    """Run benchmark with nsys profiling attached."""
    # 1. Start profiler (attaches to running server)
    print(f"\n[profile] Starting nsys profiler: {profile_name}")
    start_profiler(host, user, engine, profile_name, duration=duration, key=key)
    time.sleep(3)  # let profiler attach

    # 2. Run benchmark
    output_dir = str(RESULTS_DIR / "profiling" / profile_name)
    success = run_benchmark(host, port, preset, framework, concurrency, output_dir)

    # 3. Stop profiler
    time.sleep(2)
    stop_profiler(host, user, key=key)

    # 4. Export to sqlite
    sqlite_path = export_nsys_sqlite(host, user, profile_name, key=key)

    print(f"\n[profile] Done. Success={success}")
    print(f"[profile] nsys-rep: {REMOTE_PROFILE_DIR}/{profile_name}.nsys-rep")
    if sqlite_path:
        print(f"[profile] sqlite:   {sqlite_path}")

    return success


# ─── Comparison Experiments ───

def run_comparison_ep2_vs_tp2(host, user, port, engine, key=None):
    """
    Scenario 1: EP2 vs TP2 communication pattern comparison.
    Requires running twice with different server configs.
    """
    print("\n" + "=" * 60)
    print("COMPARISON: EP2 vs TP2 Communication Pattern")
    print("=" * 60)

    configs = [
        ("tp2-fp16", "tp2_fp16_decode"),
        ("dp2-ep2-fp16", "ep2_fp16_decode"),
    ]

    for preset, profile_name in configs:
        print(f"\n{'─' * 40}")
        print(f"Config: {preset} (engine={engine})")
        print(f"{'─' * 40}")

        # Start server
        if not start_server(host, user, preset, engine, key=key):
            print(f"[ERROR] Failed to start server for {preset}")
            continue

        # Wait for server
        ready, model = wait_for_server(host, port=port)
        if not ready:
            stop_server(host, user, key=key)
            continue

        # Run profiled benchmark
        run_profiled(
            host, user, port, preset, engine,
            framework="autogen", concurrency=64,
            profile_name=profile_name, duration=120, key=key
        )

        # Stop server
        stop_server(host, user, key=key)
        time.sleep(10)  # cooldown between configs

    print("\n[compare] Both profiles collected. Run analysis:")
    print(f"  python scripts/profile_ep2_vs_tp2.py --analyze-scenario1")


def run_comparison_framework_burst(host, user, port, preset, engine, key=None):
    """
    Scenario 4: Framework burst pattern comparison.
    Same server, different workload patterns.
    """
    print("\n" + "=" * 60)
    print("COMPARISON: Framework Burst Pattern (AutoGen vs A2A)")
    print("=" * 60)

    # Start server once
    if not start_server(host, user, preset, engine, key=key):
        return
    ready, _ = wait_for_server(host, port=port)
    if not ready:
        stop_server(host, user, key=key)
        return

    for fw, profile_name in [("autogen", f"burst_{preset}_autogen"), ("a2a", f"burst_{preset}_a2a")]:
        print(f"\n[burst] Profiling framework: {fw}")
        run_profiled(
            host, user, port, preset, engine,
            framework=fw, concurrency=32,
            profile_name=profile_name, duration=120, key=key
        )
        time.sleep(10)

    stop_server(host, user, key=key)
    print("\n[compare] Both profiles collected. Run analysis:")
    print(f"  python scripts/profile_ep2_vs_tp2.py --analyze-scenario4")


def run_comparison_batch_scaling(host, user, port, preset, engine, key=None):
    """
    Scenario 2: Batch size scaling — compute/memory bound transition.
    Same server, sweep concurrency levels with profiling.
    """
    print("\n" + "=" * 60)
    print("COMPARISON: Batch Size Scaling")
    print("=" * 60)

    if not start_server(host, user, preset, engine, key=key):
        return
    ready, _ = wait_for_server(host, port=port)
    if not ready:
        stop_server(host, user, key=key)
        return

    for conc in [1, 8, 32, 64]:
        profile_name = f"batch_{preset}_c{conc}"
        print(f"\n[batch] Profiling concurrency={conc}")
        run_profiled(
            host, user, port, preset, engine,
            framework="autogen", concurrency=conc,
            profile_name=profile_name, duration=180, key=key
        )
        time.sleep(10)

    stop_server(host, user, key=key)
    print("\n[compare] All batch profiles collected. Run analysis:")
    print(f"  python scripts/profile_ep2_vs_tp2.py --analyze-scenario2")


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description="Automated Experiment Runner")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Remote server hostname")
    parser.add_argument("--user", default=DEFAULT_USER, help="SSH username")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--key", default=DEFAULT_SSH_KEY, help="SSH private key path")
    parser.add_argument("--engine", default="sglang", choices=["vllm", "sglang"])
    parser.add_argument("--preset", default="tp2-fp8",
                        help="Server preset (tp2-fp8, tp2-fp16, dp2-ep2-fp8, dp2-ep2-fp16, ...)")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sweep", action="store_true",
                      help="Run full sweep (3 frameworks × default concurrencies)")
    mode.add_argument("--run", action="store_true",
                      help="Single benchmark run (requires --framework, --concurrency)")
    mode.add_argument("--profile", action="store_true",
                      help="Single profiled run (requires --framework, --concurrency)")
    mode.add_argument("--start-only", action="store_true",
                      help="Just start the server and exit")
    mode.add_argument("--stop", action="store_true",
                      help="Stop server and profiler sessions")
    mode.add_argument("--compare", choices=["ep2-vs-tp2", "framework-burst", "batch-scaling"],
                      help="Run comparison experiment")

    parser.add_argument("--framework", default="autogen", choices=["autogen", "langgraph", "a2a"])
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--frameworks", nargs="+", default=["autogen", "langgraph", "a2a"])
    parser.add_argument("--concurrencies", nargs="+", type=int, default=None,
                        help="Override concurrency levels for sweep")
    parser.add_argument("--profile-duration", type=int, default=120,
                        help="Nsys profiling duration in seconds")
    parser.add_argument("--server-timeout", type=int, default=600,
                        help="Max seconds to wait for server startup")
    parser.add_argument("--no-server-manage", action="store_true",
                        help="Don't start/stop server (assume already running)")

    args = parser.parse_args()
    key = args.key if args.key else None

    # Determine default concurrency levels
    if args.concurrencies is None:
        if "ep2" in args.preset or "ep" in args.preset:
            args.concurrencies = [1, 8, 32, 64]
        elif "tp1" in args.preset:
            args.concurrencies = [1, 4, 8, 16, 32]
        else:
            args.concurrencies = [1, 8, 32]

    print(f"{'=' * 60}")
    print(f"Experiment Runner")
    print(f"  Host:   {args.host}")
    print(f"  User:   {args.user}")
    print(f"  Engine: {args.engine}")
    print(f"  Preset: {args.preset}")
    print(f"{'=' * 60}")

    # Check SSH
    check_ssh(args.host, args.user, key=key)

    # ─── Modes ───

    if args.stop:
        stop_profiler(args.host, args.user, key=key)
        stop_server(args.host, args.user, key=key)
        return

    if args.start_only:
        start_server(args.host, args.user, args.preset, args.engine, key=key)
        ready, model = wait_for_server(args.host, port=args.port, timeout=args.server_timeout)
        if ready:
            print(f"\n[ready] Server is running. Use Ctrl+C or --stop to shut down.")
        return

    if args.compare:
        if args.compare == "ep2-vs-tp2":
            run_comparison_ep2_vs_tp2(args.host, args.user, args.port, args.engine, key=key)
        elif args.compare == "framework-burst":
            run_comparison_framework_burst(args.host, args.user, args.port, args.preset, args.engine, key=key)
        elif args.compare == "batch-scaling":
            run_comparison_batch_scaling(args.host, args.user, args.port, args.preset, args.engine, key=key)
        return

    # For --sweep, --run, --profile: manage server lifecycle
    if not args.no_server_manage:
        start_server(args.host, args.user, args.preset, args.engine, key=key)
        ready, model = wait_for_server(args.host, port=args.port, timeout=args.server_timeout)
        if not ready:
            stop_server(args.host, args.user, key=key)
            sys.exit(1)

    try:
        if args.sweep:
            run_sweep(args.host, args.port, args.preset, args.engine,
                      args.frameworks, args.concurrencies, key=key)

        elif args.run:
            output_name = f"{args.engine}-{args.preset}"
            output_dir = str(RESULTS_DIR / f"sweep_{output_name}_{args.framework}" / f"{args.framework}_c{args.concurrency}")
            run_benchmark(args.host, args.port, args.preset, args.framework, args.concurrency, output_dir)

        elif args.profile:
            profile_name = f"profile_{args.engine}_{args.preset}_{args.framework}_c{args.concurrency}"
            run_profiled(
                args.host, args.user, args.port, args.preset, args.engine,
                args.framework, args.concurrency,
                profile_name, duration=args.profile_duration, key=key
            )
    finally:
        if not args.no_server_manage:
            stop_server(args.host, args.user, key=key)


if __name__ == "__main__":
    main()
