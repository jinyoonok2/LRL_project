#!/usr/bin/env python3
"""Create metadata and output paths for RBY1 benchmark evaluation episodes."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare per-episode metadata for RBY1 MolmoBot evaluation."
    )
    parser.add_argument("--benchmark_json", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--benchmark_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_base_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path)
    parser.add_argument("--env_file", type=Path, required=True)
    return parser.parse_args()


def slugify(text: str, max_len: int = 90) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return (slug or "episode")[:max_len]


def episode_metadata(benchmark_json: Path, idx: int) -> dict:
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


def write_shell_env(path: Path, values: dict[str, str]) -> None:
    lines = [f"{key}={shlex.quote(value)}" for key, value in values.items()]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    metadata = episode_metadata(args.benchmark_json, args.idx)

    output_root = args.output_root
    if output_root is None:
        output_root = args.output_base_dir / (
            f"{args.task}_idx_{args.idx}_{metadata['slug']}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    metadata.update(
        {
            "task_arg": args.task,
            "idx": args.idx,
            "benchmark_dir": str(args.benchmark_dir),
            "checkpoint": str(args.checkpoint),
            "output_root": str(output_root),
        }
    )

    run_info_path = output_root / "run_info.json"
    run_info_path.write_text(json.dumps(metadata, indent=2) + "\n")

    write_shell_env(
        args.env_file,
        {
            "TASK_DESCRIPTION": str(metadata["task_description"]),
            "OBJECT_LABEL": str(metadata["object_label"]),
            "EPISODE_SLUG": str(metadata["slug"]),
            "OUTPUT_ROOT": str(output_root),
            "RUN_INFO_PATH": str(run_info_path),
        },
    )


if __name__ == "__main__":
    main()
