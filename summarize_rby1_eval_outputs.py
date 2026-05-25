"""Summarize RBY1 MolmoBot eval outputs into a per-episode CSV.

The stock MolmoSpaces eval_to_csv.py is useful for aggregate success rates.
This helper is more presentation/debug oriented: it records each completed
episode, its success/failure, instruction text, and a suggested experiment name.
By default, it writes a compact one-glance CSV. Use --detailed to include long
paths and reconstructed rerun commands.

Usage:
    python summarize_rby1_eval_outputs.py

Optional:
    python summarize_rby1_eval_outputs.py \
      --eval-root MolmoBot/MolmoBot-SPOC/eval_output/RBY1RigidManipEvalConfig \
      --benchmark-json rby1_pick_benchmark/benchmark.json \
      --output-csv rby1_eval_episode_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_EVAL_ROOT = ROOT / "MolmoBot/MolmoBot-SPOC/eval_output/RBY1RigidManipEvalConfig"
DEFAULT_BENCHMARK_JSON = ROOT / "rby1_pick_benchmark/benchmark.json"
DEFAULT_OUTPUT_CSV = ROOT / "rby1_eval_episode_summary.csv"
DEFAULT_BENCHMARK_DIR = ROOT / "rby1_pick_benchmark"
CONFIG_CLS = "molmobot_spoc.eval.config.rby1_eval_config:RBY1RigidManipEvalConfig"


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.rstrip(b"\x00").decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "U"}:
            return "".join(x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x) for x in value)
        if value.dtype == np.uint8:
            return bytes(value).rstrip(b"\x00").decode("utf-8", errors="replace")
    return str(value)


def _decode_json(value: Any) -> dict[str, Any] | None:
    try:
        text = _as_text(value)
        if not text:
            return None
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _decode_json_sequence(dataset: h5py.Dataset) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        raw = dataset[()]
    except Exception:
        return rows

    if isinstance(raw, np.ndarray) and raw.ndim > 0:
        for item in raw:
            parsed = _decode_json(item)
            if parsed is not None:
                rows.append(parsed)
    else:
        parsed = _decode_json(raw)
        if parsed is not None:
            rows.append(parsed)
    return rows


def _load_benchmark_index(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    episodes = data["episodes"] if isinstance(data, dict) and "episodes" in data else data
    indexed = []
    for idx, episode in enumerate(episodes):
        language = episode.get("language", {})
        task = episode.get("task", {})
        indexed.append(
            {
                "benchmark_idx": idx,
                "house_index": episode.get("house_index"),
                "instruction": language.get("task_description", ""),
                "pickup_obj_name": task.get("pickup_obj_name", ""),
            }
        )
    return indexed


def _match_benchmark_episode(
    benchmark: list[dict[str, Any]],
    house_index: int | None,
    instruction: str,
    pickup_obj_name: str,
) -> dict[str, Any] | None:
    candidates = [ep for ep in benchmark if ep["house_index"] == house_index]
    if instruction:
        for ep in candidates:
            if ep["instruction"] == instruction:
                return ep
    if pickup_obj_name:
        for ep in candidates:
            if ep["pickup_obj_name"] == pickup_obj_name:
                return ep
    return candidates[0] if len(candidates) == 1 else None


def _slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_len].strip("_") or "unknown_instruction"


def _recommended_command(benchmark_idx: int | None) -> str:
    idx_text = "<idx>" if benchmark_idx is None else str(benchmark_idx)
    return (
        "cd /home/jinyoon/workspace/live-robotics-lab-project/MolmoBot/MolmoBot-SPOC && "
        "conda activate mlspaces && "
        'export PYTHONPATH="/home/jinyoon/workspace/live-robotics-lab-project/molmospaces:${PYTHONPATH}" && '
        "export MUJOCO_GL=egl && export PYOPENGL_PLATFORM=egl && "
        "export MUJOCO_EGL_DEVICE_ID=0 && export JAX_PLATFORMS=cpu && "
        f"python -m molmo_spaces.evaluation.eval_main {CONFIG_CLS} "
        f"--benchmark_dir {DEFAULT_BENCHMARK_DIR} --no_wandb --num_workers 1 --idx {idx_text}"
    )


def _find_video_paths(h5_path: Path) -> dict[str, str]:
    folder = h5_path.parent
    videos = {}
    for mp4 in sorted(folder.glob("*.mp4")):
        name = mp4.name
        if "camera_follower" in name:
            videos["camera_follower_video"] = str(mp4)
        elif "head_camera" in name:
            videos["head_camera_video"] = str(mp4)
        elif "wrist_camera_l" in name:
            videos["wrist_camera_l_video"] = str(mp4)
        elif "wrist_camera_r" in name:
            videos["wrist_camera_r_video"] = str(mp4)
    return videos


def _run_id(run_dir: Path) -> str:
    # Default eval runs end with timestamps like 20260524_124624.
    if re.match(r"^\d{8}_\d{6}$", run_dir.name):
        return run_dir.name
    timestamp_dirs = [part for part in run_dir.parts if re.match(r"^\d{8}_\d{6}$", part)]
    return timestamp_dirs[-1] if timestamp_dirs else run_dir.name


def _rounded(value: float | str, digits: int = 6) -> float | str:
    if value == "":
        return ""
    return round(float(value), digits)


def _summarize_h5(h5_path: Path, benchmark: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    run_dir = h5_path.parent.parent
    house_match = re.search(r"house_(\d+)", h5_path.parent.name)
    house_index = int(house_match.group(1)) if house_match else None
    videos = _find_video_paths(h5_path)

    with h5py.File(h5_path, "r") as h5:
        for traj_name in sorted(k for k in h5.keys() if k.startswith("traj_")):
            traj = h5[traj_name]
            success = traj["success"][:] if "success" in traj else np.array([], dtype=bool)
            rewards = traj["rewards"][:] if "rewards" in traj else np.array([], dtype=float)
            task_infos = []
            if "obs/extra/task_info" in traj:
                task_infos = _decode_json_sequence(traj["obs/extra/task_info"])
            obs_scene = _decode_json(traj["obs_scene"][()]) if "obs_scene" in traj else None

            instruction = ""
            pickup_obj_name = ""
            if obs_scene:
                instruction = obs_scene.get("task_description") or obs_scene.get("goal") or ""
                pickup_obj_name = obs_scene.get("pickup_obj_name") or obs_scene.get("object_name") or ""
            if not instruction and task_infos:
                instruction = str(task_infos[0].get("task_description", ""))
            if not pickup_obj_name and task_infos:
                pickup_obj_name = str(task_infos[0].get("pickup_obj_name", ""))

            matched = _match_benchmark_episode(benchmark, house_index, instruction, pickup_obj_name)
            if matched:
                instruction = instruction or matched["instruction"]
                pickup_obj_name = pickup_obj_name or matched["pickup_obj_name"]
            benchmark_idx = matched["benchmark_idx"] if matched else None

            position_errors = [
                float(info["position_error"])
                for info in task_infos
                if isinstance(info, dict) and "position_error" in info
            ]
            success_any = bool(np.any(success)) if len(success) else False
            success_last = bool(success[-1]) if len(success) else False
            status = "success" if success_last else "fail"
            idx_label = f"idx{benchmark_idx}" if benchmark_idx is not None else f"house{house_index}"
            experiment_name = f"RBY1Rigid_{idx_label}_{_slugify(instruction)}_{status}"
            max_reward = float(np.max(rewards)) if len(rewards) else ""
            final_reward = float(rewards[-1]) if len(rewards) else ""
            min_position_error = min(position_errors) if position_errors else ""
            final_position_error = position_errors[-1] if position_errors else ""

            row = {
                "experiment_name": experiment_name,
                "run_id": _run_id(run_dir),
                "status": status,
                "run_dir": str(run_dir),
                "house_index": house_index,
                "benchmark_idx": benchmark_idx,
                "trajectory": traj_name,
                "instruction": instruction,
                "pickup_obj_name": pickup_obj_name,
                "success_any": success_any,
                "success_last": success_last,
                "num_steps": int(len(success)),
                "max_reward": _rounded(max_reward),
                "final_reward": _rounded(final_reward),
                "min_position_error": _rounded(min_position_error),
                "final_position_error": _rounded(final_position_error),
                "h5_path": str(h5_path),
                "recommended_command": _recommended_command(benchmark_idx),
            }
            row.update(videos)
            rows.append(row)
    return rows


def summarize(eval_root: Path, benchmark_json: Path, output_csv: Path, detailed: bool = False) -> list[dict[str, Any]]:
    benchmark = _load_benchmark_index(benchmark_json)
    h5_paths = sorted(eval_root.rglob("trajectories_batch_*.h5"))
    rows: list[dict[str, Any]] = []
    for h5_path in h5_paths:
        rows.extend(_summarize_h5(h5_path, benchmark))

    fieldnames = ["experiment_name", "run_id", "benchmark_idx", "house_index", "instruction", "status", "success_any", "success_last", "num_steps", "max_reward", "final_reward", "min_position_error", "final_position_error"]
    if detailed:
        fieldnames.extend(
            [
                "trajectory",
                "pickup_obj_name",
                "camera_follower_video",
                "head_camera_video",
                "wrist_camera_l_video",
                "wrist_camera_r_video",
                "h5_path",
                "run_dir",
                "recommended_command",
            ]
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RBY1 eval output folders.")
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--benchmark-json", type=Path, default=DEFAULT_BENCHMARK_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--detailed", action="store_true", help="Include long paths, video files, and rerun commands.")
    args = parser.parse_args()

    rows = summarize(args.eval_root, args.benchmark_json, args.output_csv, detailed=args.detailed)
    success_last = sum(1 for row in rows if row["success_last"])
    success_any = sum(1 for row in rows if row["success_any"])
    print(f"Saved {len(rows)} episode rows to {args.output_csv}")
    print(f"success_last: {success_last}/{len(rows)}")
    print(f"success_any: {success_any}/{len(rows)}")


if __name__ == "__main__":
    main()
