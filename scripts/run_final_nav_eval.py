"""Evaluate a disjoint shard of all 90 validation starts with the learned policy."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from prepare_final_nav_eval import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, choices=range(9), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = args.output / f"shard_{args.shard}"
    benchmark = output / "benchmark"
    prepare(benchmark)
    episodes = json.loads((benchmark / "benchmark.json").read_text())
    checkpoint = "/bigtemp/xna8aw/nav_door/train/final/trajectory/llm/step5000_unsharded"
    results = []
    for index in range(args.shard * 10, (args.shard + 1) * 10):
        episode = episodes[index]
        horizon = max(2000, 2 * episode["source"]["episode_length"])
        log = output / f"episode_{index:03d}.log"
        command = [sys.executable, "launch_scripts/run_eval.py", "--checkpoint_path", checkpoint,
                   "--benchmark_path", str(benchmark), "--eval_config_cls",
                   "lrl_molmospaces_extensions.configs.molmobot_eval:FinalTrajectoryNavDoorEvalConfig",
                   "--output_dir", str(output / f"episode_{index:03d}"), "--num_workers", "1",
                   "--idx", str(index), "--task_horizon", str(horizon), "--terminate_upon_success"]
        print(f"Evaluating held-out episode {index}/89; horizon={horizon}", flush=True)
        with log.open("w") as stream:
            run = subprocess.run(command, cwd=project / "MolmoBot/MolmoBot", stdout=stream, stderr=subprocess.STDOUT)
        outcomes = re.findall(r"^house_[^\n]+/ep\d+: (pass|fail)$", log.read_text(), flags=re.MULTILINE)
        row = {"index": index, "house": episode["house_index"], "source": episode["source"],
               "horizon": horizon, "process_exit": run.returncode,
               "success": outcomes[-1] == "pass" if outcomes and run.returncode == 0 else None}
        results.append(row)
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    if any(row["success"] is None for row in results):
        raise SystemExit("Incomplete evaluation: inspect per-episode errors")


if __name__ == "__main__":
    main()
