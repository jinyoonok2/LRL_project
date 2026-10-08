"""Prepare an existing split for the unchanged MolmoBot training loader."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import decord
import h5py
import numpy as np


class TrainingDatasetPreparation:
    def __init__(self, dataset: Path):
        self.dataset = dataset

    def run(self) -> dict:
        counts = {}
        aliases = 0
        frames_total = 0
        for split in ("train", "val"):
            root = self.dataset / split
            files = sorted(root.glob("house_*/*.h5"))
            if not files:
                raise ValueError(f"Empty training split: {root}")
            for path in files:
                with h5py.File(path, "r") as src:
                    mask = src.get("valid_traj_mask")
                    if mask is None or not np.asarray(mask).all():
                        raise ValueError(f"Missing or invalid recorded trajectory mask: {path}")
                    for name in src:
                        if not name.startswith("traj_"):
                            continue
                        trajectory = src[name]
                        frames = len(trajectory["obs/agent/qpos"])
                        frames_total += frames
                        for key in trajectory["actions"]:
                            if len(trajectory["actions"][key]) != frames:
                                raise ValueError(f"Action/observation length mismatch: {path}/{key}")
                        for camera in ("head_camera", "wrist_camera_l", "wrist_camera_r"):
                            filename = trajectory[f"obs/sensor_data/{camera}"][()].tobytes().decode().rstrip("\x00")
                            if Path(filename).is_absolute() or ".." in Path(filename).parts:
                                raise ValueError(f"External video path: {filename}")
                            video = path.parent / filename
                            if not video.exists():
                                actual = path.parent / "other_cameras" / Path(filename).name
                                if not actual.is_file() or video.is_symlink():
                                    raise FileNotFoundError(video)
                                video.symlink_to(Path("other_cameras") / actual.name)
                                aliases += 1
                            reader = decord.VideoReader(str(video), ctx=decord.cpu())
                            if len(reader) != frames:
                                raise ValueError(f"Video length mismatch: {video} ({len(reader)} vs {frames})")
                            reader.get_batch([0, frames // 2, frames - 1])
                            del reader
            validator = Path(__file__).resolve().parents[2] / "molmospaces/scripts/data/validate_trajectories.py"
            subprocess.run([
                sys.executable, str(validator), str(root), "--index-only", "--num-workers", "4",
            ], check=True)
            index = json.loads((root / "valid_trajectory_index.json").read_text())
            counts[split] = sum(len(trajectories) for house in index.values() for trajectories in house.values())
            if counts[split] != len(files):
                raise ValueError(f"Index does not cover every selected trajectory in {root}")
        summary = {"dataset": str(self.dataset), "trajectories": counts, "video_aliases_created": aliases, "total_frames": frames_total}
        (self.dataset / "training_preparation.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(TrainingDatasetPreparation(args.dataset).run(), indent=2), flush=True)


if __name__ == "__main__":
    main()
