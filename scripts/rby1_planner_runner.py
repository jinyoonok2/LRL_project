#!/usr/bin/env python3
"""Run RBY1 MolmoSpaces planner policies on JSON benchmarks.

This mirrors scripts/rby1_runner.py, but intentionally avoids MolmoBot-specific
checkpoint/model assumptions. It is meant for reproducibility checks such as:

    same benchmark episode + same planner + same seed + N repeats

The runner saves one log per task/index/repeat and hashes any generated HDF5
trajectory files so repeated runs can be compared quickly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


TASKS: dict[str, dict[str, str]] = {
    "pick": {
        "benchmark_rel": "procthor-objaverse/rby1_benchmarks/pick_benchmark",
        "eval_config_cls": (
            "molmo_spaces.data_generation.config.object_manipulation_datagen_configs:"
            "RBY1PickDataGenConfig"
        ),
    },
    "pnp": {
        "benchmark_rel": "procthor-objaverse/rby1_benchmarks/pnp_benchmark",
        "eval_config_cls": (
            "molmo_spaces.data_generation.config.object_manipulation_datagen_configs:"
            "RBY1PickAndPlaceDataGenConfig"
        ),
    },
    "opening": {
        "benchmark_rel": "ithor/rby1_benchmarks/opening_benchmark",
        "eval_config_cls": (
            "molmo_spaces.data_generation.config.object_manipulation_datagen_configs:"
            "RBY1OpenDataGenConfig"
        ),
    },
    "door_opening": {
        "benchmark_rel": "procthor-10k/rby1_benchmarks/door_opening_benchmark",
        "eval_config_cls": (
            "molmo_spaces.data_generation.config.door_opening_configs:"
            "DoorOpeningDataGenConfig"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("describe", "eval"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument(
            "--tasks",
            help="Comma-separated task override, e.g. pick,pnp. Defaults to config tasks.",
        )
        subparser.add_argument(
            "--indices",
            help="Comma-separated index override, e.g. 0,10. Defaults to config indices.",
        )

    eval_parser = subparsers.choices["eval"]
    eval_parser.add_argument(
        "--repeats",
        type=int,
        help="Override number of repeats per task/index.",
    )
    eval_parser.add_argument(
        "--run-id",
        help="Run folder/log suffix. Defaults to timestamp plus Slurm job id or PID.",
    )
    eval_parser.add_argument(
        "--output-base-dir",
        type=Path,
        help="Override config output_base_dir.",
    )
    eval_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )

    # Internal worker command. The public entry point is "eval"; this keeps each
    # repeat in a fresh Python process so global planner/simulator state cannot
    # leak across trials.
    run_one = subparsers.add_parser("run-one")
    run_one.add_argument("--eval-config-cls", required=True)
    run_one.add_argument("--benchmark-dir", type=Path, required=True)
    run_one.add_argument("--output-dir", type=Path, required=True)
    run_one.add_argument("--idx", type=int, required=True)
    run_one.add_argument("--num-workers", type=int, default=1)
    run_one.add_argument("--seed", type=int, default=42)
    run_one.add_argument("--task-horizon-steps", type=int)
    run_one.add_argument("--terminate-upon-success", action="store_true")
    run_one.add_argument("--save-partial-trajectories-on-exception", action="store_true")

    return parser.parse_args()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml_config(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    seen = seen or set()
    if path in seen:
        raise SystemExit(f"Config inheritance cycle detected at {path}")
    seen.add(path)

    data = yaml.safe_load(path.read_text()) or {}
    parent_name = data.pop("extends", None)
    if not parent_name:
        return data

    parent_path = Path(parent_name)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    parent = load_yaml_config(parent_path, seen)
    return deep_merge(parent, data)


def path_from(config: dict[str, Any], key: str) -> Path:
    value = config.get("paths", {}).get(key)
    if value is None:
        raise SystemExit(f"Missing required paths.{key} in config")
    return Path(str(value)).expanduser()


def selected_tasks_and_indices(
    config: dict[str, Any], args: argparse.Namespace
) -> tuple[list[str], list[int]]:
    if getattr(args, "tasks", None):
        tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    else:
        tasks = list(config.get("tasks") or [])

    unknown = [task for task in tasks if task not in TASKS]
    if unknown:
        raise SystemExit(f"Unknown RBY1 planner tasks: {unknown}")

    if getattr(args, "indices", None):
        indices = [int(idx.strip()) for idx in args.indices.split(",") if idx.strip()]
    else:
        indices = [int(idx) for idx in (config.get("indices") or [])]

    if not tasks:
        raise SystemExit("No tasks selected")
    if not indices:
        raise SystemExit("No indices selected")
    if any(idx < 0 for idx in indices):
        raise SystemExit("Episode indices must be non-negative")

    return tasks, indices


def setup_runtime_env(config: dict[str, Any]) -> dict[str, str]:
    paths = config.get("paths", {})
    project_root = path_from(config, "project_root")
    cache_root = path_from(config, "cache_root")

    env = os.environ.copy()
    env.setdefault("MLSPACES_CACHE_DIR", str(cache_root / "molmo-spaces-resources"))
    env.setdefault("MLSPACES_ASSETS_DIR", str(cache_root / "molmospaces/assets"))
    env.setdefault("HF_HOME", str(cache_root / "huggingface"))
    env.setdefault("TRANSFORMERS_CACHE", str(cache_root / "huggingface/transformers"))
    env.setdefault("TORCH_HOME", str(cache_root / "torch"))
    env.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")

    # Determinism helpers. These do not guarantee deterministic MuJoCo/curobo
    # behavior, but they remove avoidable Python/CUDA sources of variation.
    seed = int((config.get("eval") or {}).get("seed", 42))
    env.setdefault("PYTHONHASHSEED", str(seed))
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    pythonpath_parts = [str(project_root / "molmospaces")]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_parts)

    for key, value in (config.get("planner_env") or {}).items():
        env[str(key)] = str(value)

    for key in (
        "MLSPACES_CACHE_DIR",
        "MLSPACES_ASSETS_DIR",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
    ):
        Path(env[key]).mkdir(parents=True, exist_ok=True)

    for key, value in paths.items():
        env[f"RBY1_PLANNER_PATH_{key.upper()}"] = str(value)

    return env


def benchmark_dir(config: dict[str, Any], task: str) -> Path:
    return path_from(config, "benchmark_root") / TASKS[task]["benchmark_rel"]


def slugify(text: str, max_len: int = 90) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return (slug or "episode")[:max_len]


def episode_metadata(benchmark_json: Path, idx: int) -> dict[str, Any]:
    episodes = json.loads(benchmark_json.read_text())
    if idx < 0 or idx >= len(episodes):
        raise SystemExit(f"Episode index {idx} is out of range for {benchmark_json}")

    episode = episodes[idx]
    language = episode.get("language", {})
    task = episode.get("task", {})
    referrals = language.get("referral_expressions", {})
    task_description = (
        language.get("task_description") or task.get("task_type") or "unknown_task"
    )
    object_label = (
        referrals.get("pickup_obj_name")
        or referrals.get("place_receptacle_name")
        or referrals.get("door_body_name")
        or task.get("pickup_obj_name")
        or task.get("door_body_name")
        or "unknown_object"
    )
    return {
        "task_description": task_description,
        "object_label": object_label,
        "task_type": task.get("task_type"),
        "house_index": episode.get("house_index"),
        "scene_dataset": episode.get("scene_dataset"),
        "data_split": episode.get("data_split"),
        "pickup_obj_name": task.get("pickup_obj_name"),
        "referral_expressions": referrals,
        "slug": slugify(f"{task_description} {object_label}"),
    }


def task_horizon(config: dict[str, Any]) -> int | None:
    value = (config.get("eval") or {}).get("task_horizon_steps")
    if value in (None, ""):
        return None
    return int(value)


def num_workers(config: dict[str, Any]) -> int:
    return int((config.get("eval") or {}).get("num_workers", 1))


def seed(config: dict[str, Any]) -> int:
    return int((config.get("eval") or {}).get("seed", 42))


def repeats(config: dict[str, Any], args: argparse.Namespace) -> int:
    value = args.repeats if args.repeats is not None else (config.get("eval") or {}).get("repeats", 1)
    value = int(value)
    if value < 1:
        raise SystemExit("repeats must be >= 1")
    return value


def terminate_upon_success(config: dict[str, Any]) -> bool:
    return bool((config.get("eval") or {}).get("terminate_upon_success", False))


def describe_config(config: dict[str, Any], args: argparse.Namespace) -> None:
    tasks, indices = selected_tasks_and_indices(config, args)
    repeat_count = repeats(config, args) if args.command == "eval" else int((config.get("eval") or {}).get("repeats", 1))
    print(f"name={config.get('name', args.config.stem)}")
    print(f"project_root={path_from(config, 'project_root')}")
    print(f"benchmark_root={path_from(config, 'benchmark_root')}")
    print(f"output_base_dir={path_from(config, 'output_base_dir')}")
    print(f"tasks={' '.join(tasks)}")
    print(f"indices={' '.join(str(idx) for idx in indices)}")
    print(f"task_count={len(tasks)} index_count={len(indices)} repeats={repeat_count}")
    print(f"run_count={len(tasks) * len(indices) * repeat_count}")
    print(f"task_horizon_steps={task_horizon(config) or 'benchmark_default'}")
    print(f"num_workers={num_workers(config)}")
    print(f"seed={seed(config)}")
    print(f"terminate_upon_success={terminate_upon_success(config)}")
    print(f"planner_env={json.dumps(config.get('planner_env') or {}, sort_keys=True)}")
    for task in tasks:
        print(f"planner_config[{task}]={TASKS[task]['eval_config_cls']}")


def make_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return args.run_id
    suffix = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"


def run_subprocess(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    dry_run: bool,
    log_file: Path,
) -> int:
    printable = " ".join(str(part) for part in command)
    if dry_run:
        print(f"[dry-run] cwd={cwd}")
        print(f"[dry-run] {printable}")
        print(f"[dry-run] log={log_file}")
        return 0

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w") as f:
        f.write(f"cwd={cwd}\n")
        f.write(f"command={printable}\n\n")
        f.flush()
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def parse_benchmark_success(log_file: Path) -> tuple[str, str]:
    if not log_file.is_file():
        return "unknown", "missing_log"

    text = log_file.read_text(errors="replace")
    completed_matches = re.findall(r"completed with success=(True|False)", text)
    if completed_matches:
        value = completed_matches[-1] == "True"
        return ("success" if value else "failed", "completed_with_success")

    pass_fail_matches = re.findall(r"house_[^:\n]+/ep\d+:\s*(pass|fail)", text)
    if pass_fail_matches:
        value = pass_fail_matches[-1] == "pass"
        return ("success" if value else "failed", "house_pass_fail")

    rate_matches = re.findall(r"Success rate:\s*([0-9.]+)%", text)
    if rate_matches:
        return ("success" if float(rate_matches[-1]) > 0.0 else "failed", "success_rate")

    count_matches = re.findall(r"Success count:\s*(\d+),\s*Total count:\s*(\d+)", text)
    if count_matches:
        success_count, total_count = (int(part) for part in count_matches[-1])
        if total_count > 0:
            return ("success" if success_count > 0 else "failed", "success_count")

    return "unknown", "not_found"


def hash_h5_outputs(output_root: Path) -> str:
    h5_files = sorted(output_root.glob("**/trajectories*.h5"))
    if not h5_files:
        return ""

    import h5py

    digest = hashlib.sha256()
    for h5_path in h5_files:
        digest.update(str(h5_path.relative_to(output_root)).encode())
        with h5py.File(h5_path, "r") as f:
            def visit(name: str, obj: Any) -> None:
                if not hasattr(obj, "shape"):
                    return
                digest.update(name.encode())
                digest.update(str(obj.shape).encode())
                digest.update(str(obj.dtype).encode())
                value = obj[()]
                try:
                    digest.update(value.tobytes())
                except Exception:
                    digest.update(repr(value).encode(errors="replace"))

            f.visititems(visit)
    return digest.hexdigest()


def write_run_info(
    output_root: Path,
    metadata: dict[str, Any],
    *,
    task: str,
    idx: int,
    repeat: int,
    bench_dir: Path,
    eval_config_cls: str,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata)
    metadata.update(
        {
            "task_arg": task,
            "idx": idx,
            "repeat": repeat,
            "benchmark_dir": str(bench_dir),
            "eval_config_cls": eval_config_cls,
            "output_root": str(output_root),
        }
    )
    run_info_path = output_root / "run_info.json"
    run_info_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return run_info_path


def run_eval(config: dict[str, Any], args: argparse.Namespace) -> int:
    project_root = path_from(config, "project_root")
    molmospaces_root = project_root / "molmospaces"
    env = setup_runtime_env(config)
    tasks, indices = selected_tasks_and_indices(config, args)
    repeat_count = repeats(config, args)

    run_id = make_run_id(args)
    base_output = args.output_base_dir or path_from(config, "output_base_dir")
    output_base_dir = base_output / run_id
    log_dir = project_root / "logs" / f"rby1_planner_{run_id}"
    summary_file = log_dir / "summary.tsv"

    describe_config(config, args)
    print(f"run_id={run_id}")
    print(f"log_dir={log_dir}")
    print(f"output_base_dir={output_base_dir}")
    if args.dry_run:
        print(f"[dry-run] summary={summary_file}")
    else:
        log_dir.mkdir(parents=True, exist_ok=True)
        output_base_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int]] = []
    process_failure_count = 0

    for task in tasks:
        bench_dir = benchmark_dir(config, task)
        benchmark_json = bench_dir / "benchmark.json"
        if not args.dry_run and not benchmark_json.is_file():
            raise SystemExit(f"Missing benchmark.json: {bench_dir}")

        eval_config_cls = TASKS[task]["eval_config_cls"]
        for idx in indices:
            if benchmark_json.is_file():
                metadata = episode_metadata(benchmark_json, idx)
            else:
                metadata = {
                    "task_description": "dry_run",
                    "object_label": "dry_run",
                    "slug": "dry_run",
                }

            for repeat_idx in range(1, repeat_count + 1):
                output_root = (
                    output_base_dir
                    / f"{task}_idx_{idx}_repeat_{repeat_idx}_{metadata['slug']}"
                )
                run_info_path = output_root / "run_info.json"
                if not args.dry_run:
                    run_info_path = write_run_info(
                        output_root,
                        metadata,
                        task=task,
                        idx=idx,
                        repeat=repeat_idx,
                        bench_dir=bench_dir,
                        eval_config_cls=eval_config_cls,
                    )

                log_file = log_dir / f"{task}_idx_{idx}_repeat_{repeat_idx}.log"
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "run-one",
                    "--eval-config-cls",
                    eval_config_cls,
                    "--benchmark-dir",
                    str(bench_dir),
                    "--output-dir",
                    str(output_root),
                    "--idx",
                    str(idx),
                    "--num-workers",
                    str(num_workers(config)),
                    "--seed",
                    str(seed(config)),
                ]
                horizon = task_horizon(config)
                if horizon is not None:
                    cmd.extend(["--task-horizon-steps", str(horizon)])
                if terminate_upon_success(config):
                    cmd.append("--terminate-upon-success")
                if (config.get("eval") or {}).get(
                    "save_partial_trajectories_on_exception", False
                ):
                    cmd.append("--save-partial-trajectories-on-exception")

                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"Starting planner task={task} idx={idx} repeat={repeat_idx}"
                )
                print(f"Log: {log_file}")
                print(f"Run info: {run_info_path}")
                exit_code = run_subprocess(
                    cmd,
                    cwd=molmospaces_root,
                    env=env,
                    dry_run=args.dry_run,
                    log_file=log_file,
                )
                if exit_code != 0:
                    process_failure_count += 1

                benchmark_success, benchmark_success_source = (
                    ("unknown", "dry_run")
                    if args.dry_run
                    else parse_benchmark_success(log_file)
                )
                trajectory_hash = "" if args.dry_run else hash_h5_outputs(output_root)
                rows.append(
                    {
                        "task": task,
                        "idx": idx,
                        "repeat": repeat_idx,
                        "process_status": "success" if exit_code == 0 else "failed",
                        "exit_code": exit_code,
                        "benchmark_success": benchmark_success,
                        "benchmark_success_source": benchmark_success_source,
                        "trajectory_hash": trajectory_hash,
                        "log_file": str(log_file),
                        "output_root": str(output_root),
                    }
                )

    if not args.dry_run:
        with summary_file.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "task",
                    "idx",
                    "repeat",
                    "process_status",
                    "exit_code",
                    "benchmark_success",
                    "benchmark_success_source",
                    "trajectory_hash",
                    "log_file",
                    "output_root",
                ],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(rows)

    unique_hashes = sorted({str(row["trajectory_hash"]) for row in rows if row["trajectory_hash"]})
    print("RBY1 planner eval complete.")
    print(f"Process failures: {process_failure_count}")
    print(f"Runs: {len(rows)}")
    print(f"Unique trajectory hashes: {len(unique_hashes)}")
    print(f"Summary: {summary_file}")
    return 1 if process_failure_count else 0


def run_one(args: argparse.Namespace) -> int:
    # Seed non-solver randomness before run_evaluation creates task samplers.
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)

    import numpy as np

    from molmo_spaces.evaluation.eval_main import run_evaluation

    np.random.seed(args.seed)

    eval_kwargs = {
        "eval_config_cls": args.eval_config_cls,
        "benchmark_dir": args.benchmark_dir,
        "checkpoint_path": None,
        "task_horizon_steps": args.task_horizon_steps,
        "output_dir": args.output_dir,
        "num_workers": args.num_workers,
        "use_wandb": False,
        "episode_idx": args.idx,
        "save_partial_trajectories_on_exception": (
            args.save_partial_trajectories_on_exception
        ),
    }
    # Some planner/data-generation config classes do not currently declare the
    # eval-only terminate_upon_success field. Passing it through run_evaluation
    # makes Pydantic reject the config before rollout starts, so planner repro
    # runs use the task horizon and post-hoc success trace instead.
    if args.terminate_upon_success:
        print(
            "Note: terminate_upon_success is not passed for planner configs; "
            "using task horizon and saved success trace.",
            flush=True,
        )

    results = run_evaluation(**eval_kwargs)
    print(f"Success count: {results.success_count}, Total count: {results.total_count}")
    print(f"Success rate: {results.success_rate:.1%}")
    print(f"Output directory: {results.output_dir}")
    for result in results.episode_results:
        print(f"{result.house_id}/ep{result.episode_idx}: {'pass' if result.success else 'fail'}")
    return 0


def main() -> None:
    args = parse_args()
    if args.command == "run-one":
        raise SystemExit(run_one(args))

    config = load_yaml_config(args.config)
    if args.command == "describe":
        describe_config(config, args)
        return
    if args.command == "eval":
        raise SystemExit(run_eval(config, args))

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
