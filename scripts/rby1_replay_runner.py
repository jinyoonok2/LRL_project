#!/usr/bin/env python3
"""Replay downloaded MolmoBot-data RBY1 trajectories in MolmoSpaces."""

from __future__ import annotations

import argparse
import csv
import datetime
import importlib.util
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EVAL_CONFIG_CLS = "olmo.eval.configure_molmo_spaces:RBY1RecordedTrajectoryEvalConfig"

SCENE_DATASET_BY_CONFIG = {
    "RBY1PickDataGenConfig": "procthor-objaverse",
    "RBY1PickAndPlaceDataGenConfig": "procthor-objaverse",
    "RBY1OpenDataGenConfig": "ithor",
}

TASK_CLS_BY_TASK_TYPE = {
    "pick": "molmo_spaces.tasks.pick_task.PickTask",
    "pick_and_place": "molmo_spaces.tasks.pick_and_place_task.PickAndPlaceTask",
    "open": "molmo_spaces.tasks.opening_tasks.OpeningTask",
    "close": "molmo_spaces.tasks.opening_tasks.OpeningTask",
}

PNP_RECEPTACLE_POS_DISPLACEMENT = 0.15
PNP_RECEPTACLE_ROT_DISPLACEMENT_RAD = 1.0471975511965976  # radians(60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("prepare", "eval"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument(
            "--configs",
            help="Comma-separated dataset config override, e.g. RBY1PickDataGenConfig",
        )
        subparser.add_argument("--dry-run", action="store_true")

    prepare_parser = subparsers.choices["prepare"]
    prepare_parser.add_argument(
        "--prepare-assets",
        action="store_true",
        help="Also install/verify scene, object, and grasp assets for selected replay episodes.",
    )
    prepare_parser.add_argument(
        "--max-episodes-per-config",
        type=int,
        help="Limit asset preparation to the first N replay episodes per config.",
    )
    prepare_parser.add_argument(
        "--indices",
        help="Comma-separated episode indices/ranges for asset preparation, e.g. 0,3,5-7.",
    )
    prepare_parser.add_argument("--variant", default="ceiling", choices=("base", "ceiling"))
    prepare_parser.add_argument("--grasp-source", default="droid_objaverse")

    eval_parser = subparsers.choices["eval"]
    eval_parser.add_argument("--run-id")
    eval_parser.add_argument("--max-episodes-per-config", type=int)
    eval_parser.add_argument("--repeats", type=int, help="Repeats per selected replay episode.")
    eval_parser.add_argument("--output-base-dir", type=Path)

    run_one = subparsers.add_parser("run-one")
    run_one.add_argument("--benchmark-dir", type=Path, required=True)
    run_one.add_argument("--output-dir", type=Path, required=True)
    run_one.add_argument("--idx", type=int, required=True)
    run_one.add_argument("--trajectory-path", type=Path, required=True)
    run_one.add_argument("--trajectory-key", required=True)
    run_one.add_argument("--num-workers", type=int, default=1)
    run_one.add_argument("--task-horizon-steps", type=int)
    run_one.add_argument("--terminate-upon-success", action="store_true")

    return parser.parse_args()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
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
    return deep_merge(load_yaml_config(parent_path, seen), data)


def path_from(config: dict[str, Any], key: str) -> Path:
    value = config.get("paths", {}).get(key)
    if value is None:
        raise SystemExit(f"Missing required paths.{key} in config")
    return Path(str(value)).expanduser()


def selected_configs(config: dict[str, Any], args: argparse.Namespace) -> list[str]:
    if args.configs:
        return [part.strip() for part in args.configs.split(",") if part.strip()]
    return [str(name) for name in (config.get("configs") or [])]


def setup_runtime_env(config: dict[str, Any]) -> dict[str, str]:
    project_root = path_from(config, "project_root")
    cache_root = path_from(config, "cache_root").resolve()
    env = os.environ.copy()
    env["MLSPACES_CACHE_DIR"] = str(cache_root / "molmo-spaces-resources")
    env["MLSPACES_ASSETS_DIR"] = str(cache_root / "molmospaces/assets")
    env["HF_HOME"] = str(cache_root / "huggingface")
    env["TRANSFORMERS_CACHE"] = str(cache_root / "huggingface/transformers")
    env["TORCH_HOME"] = str(cache_root / "torch")
    env["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("HF_HUB_DISABLE_XET", "1")

    pythonpath_parts = [
        str(project_root / "MolmoBot/MolmoBot"),
        str(project_root / "molmospaces"),
    ]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    return env


def apply_runtime_env(config: dict[str, Any]) -> None:
    env = setup_runtime_env(config)
    os.environ.update(env)
    for path in reversed(env["PYTHONPATH"].split(":")):
        if path and path not in sys.path:
            sys.path.insert(0, path)


def benchmark_dir(config: dict[str, Any], dataset_config: str) -> Path:
    return path_from(config, "benchmark_root") / f"{dataset_config}_val"


def prepare_benchmark(
    config: dict[str, Any],
    dataset_config: str,
    *,
    dry_run: bool,
) -> Path:
    project_root = path_from(config, "project_root")
    data_root = path_from(config, "data_root")
    out_dir = benchmark_dir(config, dataset_config)
    benchmark_json = out_dir / "benchmark.json"
    if benchmark_json.is_file():
        return out_dir

    print(f"Preparing replay benchmark: {dataset_config} -> {out_dir}")
    if not dry_run:
        create_replay_benchmark(
            project_root=project_root,
            dataset_root=data_root / dataset_config,
            output_dir=out_dir,
            dataset_config=dataset_config,
        )
    return out_dir


def load_benchmark_converter(project_root: Path):
    converter_path = project_root / "molmospaces/scripts/benchmarks/create_json_benchmark.py"
    spec = importlib.util.spec_from_file_location("molmospaces_create_json_benchmark", converter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load benchmark converter from {converter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_obs_scene(raw: Any) -> dict[str, Any]:
    if isinstance(raw, bytes):
        return json.loads(raw.decode("utf-8"))
    if hasattr(raw, "astype"):
        try:
            value = raw.astype("T")[()]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            return json.loads(str(value))
        except Exception:
            pass
    return json.loads(str(raw))


def create_replay_benchmark(
    *,
    project_root: Path,
    dataset_root: Path,
    output_dir: Path,
    dataset_config: str,
) -> None:
    import h5py

    converter = load_benchmark_converter(project_root)
    scene_dataset = SCENE_DATASET_BY_CONFIG.get(dataset_config, "procthor-objaverse")
    data_split = "val"
    created_date = datetime.datetime.now().strftime("%Y-%m-%d")

    episode_specs: list[dict[str, Any]] = []
    for h5_path in sorted(dataset_root.glob("**/trajectories*.h5")):
        house_name = h5_path.parent.name
        if not house_name.startswith("house_"):
            continue
        house_id = int(house_name.removeprefix("house_"))
        with h5py.File(h5_path, "r") as f:
            for traj_key in sorted(k for k in f.keys() if k.startswith("traj_")):
                traj = f[traj_key]
                if "obs_scene" not in traj or "actions" not in traj:
                    continue
                obs_scene = parse_obs_scene(traj["obs_scene"][()])
                frozen_config = converter.extract_frozen_config(obs_scene)
                action_keys = list(traj["actions"].keys())
                if not action_keys:
                    continue
                source_episode_length = int(len(traj["actions"][action_keys[0]]))
                episode_spec = converter.frozen_config_to_episode_spec(
                    frozen_config=frozen_config,
                    obs_scene=obs_scene,
                    house_id=house_id,
                    scene_dataset=scene_dataset,
                    data_split=data_split,
                    source_h5_file=str(h5_path),
                    source_traj_key=traj_key,
                    source_episode_length=source_episode_length,
                    img_resolution=(1024, 576),
                    camera_system_class="RBY1GoProD455CameraSystem",
                    benchmark_created_date=created_date,
                    task_horizon_sec=30,
                )
                episode_dict = episode_spec.model_dump()
                task_type = obs_scene.get("task_type")
                if task_type:
                    task_dict = episode_dict.setdefault("task", {})
                    task_dict["task_type"] = task_type
                    if task_type in TASK_CLS_BY_TASK_TYPE:
                        task_dict["task_cls"] = TASK_CLS_BY_TASK_TYPE[task_type]
                    if task_type == "pick_and_place":
                        task_dict["max_place_receptacle_pos_displacement"] = (
                            PNP_RECEPTACLE_POS_DISPLACEMENT
                        )
                        task_dict["max_place_receptacle_rot_displacement"] = (
                            PNP_RECEPTACLE_ROT_DISPLACEMENT_RAD
                        )
                task_cls = episode_dict.setdefault("task", {}).get("task_cls")
                if isinstance(task_cls, str) and task_cls.startswith("mujoco_thor.tasks."):
                    episode_dict["task"]["task_cls"] = task_cls.replace(
                        "mujoco_thor.tasks.",
                        "molmo_spaces.tasks.",
                        1,
                    )
                episode_specs.append(episode_dict)

    if not episode_specs:
        raise RuntimeError(f"No replay episodes found under {dataset_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark.json").write_text(json.dumps(episode_specs, indent=2) + "\n")
    metadata = {
        "description": f"Replay benchmark from {dataset_config} validation trajectories",
        "created_at": datetime.datetime.now().isoformat(),
        "source_datagen_path": str(dataset_root),
        "num_episodes": len(episode_specs),
        "num_houses": len({ep["house_index"] for ep in episode_specs}),
        "scene_dataset": scene_dataset,
        "data_split": data_split,
    }
    (output_dir / "benchmark_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def prepare(config: dict[str, Any], args: argparse.Namespace) -> int:
    apply_runtime_env(config)
    for dataset_config in selected_configs(config, args):
        out_dir = prepare_benchmark(config, dataset_config, dry_run=args.dry_run)
        print(f"{dataset_config}: {out_dir}")
        if getattr(args, "prepare_assets", False):
            episodes = [] if args.dry_run else load_episodes(out_dir / "benchmark.json")
            indices = (
                selected_asset_indices(config, args, len(episodes))
                if not args.dry_run
                else parse_indices_arg(getattr(args, "indices", None), 10_000) or []
            )
            print(f"Preparing assets for {dataset_config}: indices={indices}")
            exit_code = prepare_benchmark_assets(
                config,
                benchmark_dir=out_dir,
                indices=indices,
                variant=args.variant,
                grasp_source=args.grasp_source,
                dry_run=args.dry_run,
            )
            if exit_code:
                return exit_code
    return 0


def make_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return args.run_id
    job_id = os.environ.get("SLURM_JOB_ID")
    suffix = job_id or str(os.getpid())
    return f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"


def slugify(text: str, max_len: int = 90) -> str:
    import re

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return (slug or "episode")[:max_len]


def load_episodes(benchmark_json: Path) -> list[dict[str, Any]]:
    return json.loads(benchmark_json.read_text())


def parse_indices_arg(indices_arg: str | None, total_count: int) -> list[int] | None:
    if not indices_arg:
        return None
    indices: list[int] = []
    for part in indices_arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise SystemExit(f"Invalid descending index range: {part}")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))

    unique_indices = sorted(set(indices))
    out_of_range = [idx for idx in unique_indices if idx < 0 or idx >= total_count]
    if out_of_range:
        raise SystemExit(
            f"Replay episode indices out of range {out_of_range}; "
            f"benchmark has {total_count} episodes"
        )
    return unique_indices


def selected_asset_indices(
    config: dict[str, Any],
    args: argparse.Namespace,
    episode_count: int,
) -> list[int]:
    explicit_indices = parse_indices_arg(getattr(args, "indices", None), episode_count)
    if explicit_indices is not None:
        return explicit_indices

    max_eps = getattr(args, "max_episodes_per_config", None)
    if max_eps is None:
        max_eps = (config.get("eval") or {}).get("max_episodes_per_config")
    if max_eps in (None, ""):
        return list(range(episode_count))
    max_eps = int(max_eps)
    if max_eps < 1:
        raise SystemExit("max-episodes-per-config must be >= 1")
    return list(range(min(max_eps, episode_count)))


def prepare_benchmark_assets(
    config: dict[str, Any],
    *,
    benchmark_dir: Path,
    indices: list[int],
    variant: str,
    grasp_source: str,
    dry_run: bool,
) -> int:
    if not indices:
        print(f"No selected replay episodes for asset preparation: {benchmark_dir}")
        return 0

    project_root = path_from(config, "project_root")
    cmd = [
        sys.executable,
        "scripts/benchmarks/prepare_benchmark_assets.py",
        "--benchmark_dir",
        str(benchmark_dir),
        "--idx",
        *[str(idx) for idx in indices],
        "--variant",
        variant,
        "--grasp_source",
        grasp_source,
    ]
    return run_subprocess(
        cmd,
        cwd=project_root / "molmospaces",
        env=setup_runtime_env(config),
        dry_run=dry_run,
        log_file=project_root / "logs" / f"rby1_replay_prepare_assets_{benchmark_dir.name}.log",
    )


def run_subprocess(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    dry_run: bool,
    log_file: Path,
) -> int:
    print("Command:", " ".join(cmd))
    if dry_run:
        return 0
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w") as f:
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode


def run_eval(config: dict[str, Any], args: argparse.Namespace) -> int:
    apply_runtime_env(config)
    project_root = path_from(config, "project_root")
    env = setup_runtime_env(config)
    run_id = make_run_id(args)
    output_base = args.output_base_dir or path_from(config, "output_base_dir")
    output_base = output_base / run_id
    log_dir = project_root / "logs" / f"rby1_replay_{run_id}"
    summary_file = log_dir / "summary.tsv"
    max_eps = args.max_episodes_per_config
    if max_eps is None:
        max_eps = int((config.get("eval") or {}).get("max_episodes_per_config", 3))
    repeats = args.repeats
    if repeats is None:
        repeats = int((config.get("eval") or {}).get("repeats", 1))
    if repeats < 1:
        raise SystemExit("repeats must be >= 1")

    rows: list[dict[str, str | int]] = []
    failures = 0
    for dataset_config in selected_configs(config, args):
        bench_dir = prepare_benchmark(config, dataset_config, dry_run=args.dry_run)
        episodes = load_episodes(bench_dir / "benchmark.json") if not args.dry_run else []
        if max_eps:
            episodes = episodes[:max_eps]
        for idx, episode in enumerate(episodes):
            source = episode["source"]
            task_desc = episode.get("language", {}).get("task_description", dataset_config)
            obj_name = (
                episode.get("task", {}).get("pickup_obj_name")
                or episode.get("task", {}).get("place_receptacle_name")
                or episode.get("task", {}).get("door_body_name")
                or "object"
            )
            episode_slug = f"idx_{idx}_{slugify(task_desc + ' ' + obj_name)}"
            for repeat_idx in range(1, repeats + 1):
                output_root = output_base / dataset_config / episode_slug
                if repeats > 1:
                    output_root = output_root / f"repeat_{repeat_idx}"
                log_file = (
                    log_dir / f"{dataset_config}_idx_{idx}.log"
                    if repeats == 1
                    else log_dir / f"{dataset_config}_idx_{idx}_repeat_{repeat_idx}.log"
                )
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "run-one",
                    "--benchmark-dir",
                    str(bench_dir),
                    "--output-dir",
                    str(output_root),
                    "--idx",
                    str(idx),
                    "--trajectory-path",
                    str(source["h5_file"]),
                    "--trajectory-key",
                    str(source["traj_key"]),
                    "--num-workers",
                    str(int((config.get("eval") or {}).get("num_workers", 1))),
                ]
                horizon = (config.get("eval") or {}).get("task_horizon_steps")
                if horizon not in (None, ""):
                    cmd.extend(["--task-horizon-steps", str(int(horizon))])
                if bool((config.get("eval") or {}).get("terminate_upon_success", False)):
                    cmd.append("--terminate-upon-success")

                print(f"Starting replay config={dataset_config} idx={idx} repeat={repeat_idx}")
                exit_code = run_subprocess(
                    cmd,
                    cwd=project_root / "molmospaces",
                    env=env,
                    dry_run=args.dry_run,
                    log_file=log_file,
                )
                if exit_code:
                    failures += 1
                rows.append(
                    {
                        "dataset_config": dataset_config,
                        "idx": idx,
                        "repeat": repeat_idx,
                        "exit_code": exit_code,
                        "trajectory_path": str(source["h5_file"]),
                        "trajectory_key": str(source["traj_key"]),
                        "log_file": str(log_file),
                        "output_root": str(output_root),
                    }
                )

    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
        with summary_file.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    print("RBY1 replay complete.")
    print(f"Runs: {len(rows)}")
    print(f"Failures: {failures}")
    print(f"Summary: {summary_file}")
    return 1 if failures else 0


def run_one(args: argparse.Namespace) -> int:
    os.environ["RBY1_REPLAY_TRAJECTORY_PATH"] = str(args.trajectory_path)
    os.environ["RBY1_REPLAY_TRAJECTORY_KEY"] = args.trajectory_key

    from molmo_spaces.evaluation.eval_main import run_evaluation

    results = run_evaluation(
        eval_config_cls=EVAL_CONFIG_CLS,
        benchmark_dir=args.benchmark_dir,
        checkpoint_path=None,
        task_horizon_steps=args.task_horizon_steps,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        use_wandb=False,
        episode_idx=args.idx,
        terminate_upon_success=args.terminate_upon_success,
    )
    print(f"Success count: {results.success_count}, Total count: {results.total_count}")
    print(f"Success rate: {results.success_rate:.1%}")
    print(f"Output directory: {results.output_dir}")
    for result in results.episode_results:
        print(f"{result.house_id}/ep{result.episode_idx}: {'pass' if result.success else 'fail'}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "run-one":
        return run_one(args)
    config = load_yaml_config(args.config)
    if args.command == "prepare":
        return prepare(config, args)
    if args.command == "eval":
        return run_eval(config, args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
