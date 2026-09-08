#!/usr/bin/env python3
"""Classify saved nav-to-door trajectories by physical door panel count."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import h5py

from molmo_spaces.molmo_spaces_constants import get_scenes
from lrl_molmospaces_extensions.tasks.util_samplers.door_panels import (
    door_panel_info_by_body_name,
)
from molmo_spaces.utils.scene_metadata_utils import get_scene_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--scene-dataset", default="procthor-10k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--h5-glob", default="**/*.h5")
    return parser.parse_args()


def house_index_from_path(path: Path) -> int:
    for part in reversed(path.parts):
        match = re.fullmatch(r"house_(\d+)", part)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not determine house index from {path}")


def decode_obs_scene(dataset: h5py.Dataset) -> dict:
    value = dataset[()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def main() -> None:
    args = parse_args()
    scene_index = get_scenes(args.scene_dataset, args.split)[args.split]
    metadata_cache: dict[int, dict | None] = {}
    records: list[dict] = []
    file_statuses: dict[Path, list[str]] = {}

    for h5_path in sorted(args.input_root.glob(args.h5_glob)):
        house_index = house_index_from_path(h5_path)
        if house_index not in metadata_cache:
            variants = scene_index[house_index]
            scene_path = variants["base"] if isinstance(variants, dict) else variants
            metadata_cache[house_index] = get_scene_metadata(scene_path)
        panel_info = door_panel_info_by_body_name(metadata_cache[house_index])

        with h5py.File(h5_path, "r") as h5_file:
            for trajectory_name in sorted(h5_file):
                trajectory = h5_file[trajectory_name]
                if "obs_scene" not in trajectory:
                    status = "unclassified"
                    scene = {}
                else:
                    scene = decode_obs_scene(trajectory["obs_scene"])
                    target_name = scene.get("target_door_name")
                    info = panel_info.get(target_name)
                    if info is None:
                        status = "unclassified"
                    elif info.is_single_panel:
                        status = "single_panel"
                    else:
                        status = "double_panel"

                info = panel_info.get(scene.get("target_door_name"))
                records.append(
                    {
                        "h5_path": str(h5_path.relative_to(args.input_root)),
                        "trajectory": trajectory_name,
                        "house_index": house_index,
                        "target_door_name": scene.get("target_door_name"),
                        "status": status,
                        "doorway_name": info.doorway_name if info else None,
                        "door_asset_id": info.asset_id if info else None,
                        "door_panel_count": info.panel_count if info else None,
                    }
                )
                file_statuses.setdefault(h5_path, []).append(status)

    counts = Counter(record["status"] for record in records)
    eligible_files = [
        str(path.relative_to(args.input_root))
        for path, statuses in file_statuses.items()
        if statuses and all(status == "single_panel" for status in statuses)
    ]
    manifest = {
        "input_root": str(args.input_root),
        "scene_dataset": args.scene_dataset,
        "split": args.split,
        "counts": dict(sorted(counts.items())),
        "single_panel_h5_files": eligible_files,
        "records": records,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2))
    eligible_path = args.output_manifest.with_suffix(".single_panel_files.txt")
    eligible_path.write_text("\n".join(eligible_files) + ("\n" if eligible_files else ""))

    print(json.dumps(manifest["counts"], indent=2))
    print(f"Manifest: {args.output_manifest}")
    print(f"Eligible H5 list: {eligible_path}")


if __name__ == "__main__":
    main()
