import os
from random import random
import subprocess
from pathlib import Path
import shlex

CACHEBENCH = "./opt/bin/cachebench"
GETDEPS_ENV_CMD = [
    "python3", "./build/fbcode_builder/getdeps.py", "env",
    "cachelib",
    "--scratch-path=./deps",
    "--install-dir=./opt"
]

def load_env():
    """Load the CacheLib environment (same as eval $(getdeps.py env ...))."""
    print("Loading build environment...")
    cmd = " ".join(shlex.quote(x) for x in GETDEPS_ENV_CMD)
    env_output = subprocess.check_output(cmd, shell=True, text=True)
    
    # Parse env lines like: export VAR=value
    env = os.environ.copy()
    for line in env_output.splitlines():
        if line.startswith("export "):
            line = line[len("export "):]
            if "=" in line:
                key, val = line.split("=", 1)
                env[key] = val
    return env


def run_configs(root_dir: str):
    root = Path(root_dir)
    env = load_env()

    configs = list(root.rglob("*/config_*.json"))

    # no s4fifo
    configs = [c for c in configs]

    import random
    random.shuffle(configs)

    if not configs:
        print("No config_*.json found.")
        return

    print(configs)
    print(f"Found {len(configs)} configs.")    
    new_config = []
    for config_path in configs:
        import json
        with open(config_path, "r") as f:
            loaded_json = json.load(f)
        num_ops = loaded_json.get("test_config", {}).get("numOps", None)
        if num_ops != 50000000:
            print(f"⚠ Skipping {config_path} due to numOps={num_ops} (expected 50000000)")
            exit(1)
        pool_rebalance_interval = loaded_json.get("cache_config", {}).get("poolRebalanceIntervalSec", None)
        if pool_rebalance_interval != 1:
            print(f"⚠ Skipping {config_path} due to poolRebalanceIntervalSec={pool_rebalance_interval} (expected 1)")
            exit(1)

    print("Execution order of configs:", configs)
    for i, config_path in enumerate(configs):
        curr_time_str = os.popen("date +%Y%m%d_%H%M%S").read().strip()
        
        print(f"[{i+1}/{len(configs)}] {curr_time_str} Processing {config_path} ...")

        # Only 1 from top output
        parent_name = str(config_path.parent.name)
        # 2 from top output
        # parent_name = str(config_path.parent.parent.name)

        results_dir = "results_new_s4fifo_splice_s3revamp_partial" + "/" + parent_name 
        results_dir = Path(results_dir)

        print("Results dir:", results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
    
        name = config_path.stem.replace("config_", "")  # s3fifo, tinylfu, lru, ...
        log_path = results_dir / f"log_{name}.txt"

        print(f"Running {config_path} → {log_path}")
        # If log_path exists, skip
        if log_path.exists():
            print(f"  ⚠ Skipped (log exists): {log_path}")
            continue

        cmd = [
            CACHEBENCH,
            f"--json-test-config={str(config_path)}",
            f"--progress_stats_file={log_path}",
            "--progress=15",
            "--report-api-latency=true",
            '--report_ac_memory_usage_stats="raw"'
        ]

        with open(log_path, "w") as logf:
            process = subprocess.Popen(cmd, stdout=logf, stderr=logf, env=env)
            process.wait()

        print(f"  ✔ Done: {log_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 run_configs.py <root_dir_of_configs>")
        exit(1)

    run_configs(sys.argv[1])
    # Example: python3 /home/cc/cachelib-fork/opt/test_configs/hit_ratio/runner.py /home/cc/cachelib-fork/opt_rel/test_configs/hit_ratio
    # python3 /home/cc/cachelib-fork/opt/test_configs/hit_ratio/runner.py /home/cc/cachelib-fork/opt/test_configs/hit_ratio