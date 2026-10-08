"""Run the modular baseline as one reproducible workflow."""

import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path

import yaml

from lrl_molmospaces_extensions.baseline.config import BaselineConfig, SHARDS


class BaselinePipeline:
    def __init__(self, config: BaselineConfig, run_id: str):
        self.config = config
        self.root = config.run_dir(run_id)
        self.manifest = self.root / "pipeline.json"

    def describe(self):
        return {
            "config": self.config.to_dict(),
            "run_dir": str(self.root),
            "stages": ["generate", "organize", "rerender_with_points", "validate"],
            "raw": str(self.root / "raw"),
            "recordings": str(self.root / "recordings"),
            "dataset": str(self.root / "dataset"),
        }

    def _save(self, state):
        temporary = self.manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(self.manifest)

    def run(self, resume=False):
        if self.root.exists():
            if not resume:
                raise FileExistsError(f"Run exists: {self.root}; use --resume for completed stages")
            state = json.loads(self.manifest.read_text())
            if state["config"] != self.config.to_dict():
                raise ValueError("Resume config differs from saved baseline config")
        else:
            if resume:
                raise FileNotFoundError(self.root)
            self.root.mkdir(parents=True)
            (self.root / "config.yaml").write_text(yaml.safe_dump(self.config.to_dict(), sort_keys=False))
            state = {**self.describe(), "completed_stages": [], "status": "running"}
            self._save(state)
        link = Path(self.config.results_root) / self.config.name / self.root.name
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() != self.root.resolve():
                raise ValueError(f"Result shortcut points elsewhere: {link}")
        elif link.exists():
            raise FileExistsError(link)
        else:
            link.symlink_to(self.root.resolve(), target_is_directory=True)
        stages = {
            "generate": self.generate,
            "organize": self.organize,
            "rerender_with_points": self.rerender,
            "validate": self.validate,
        }
        for name, action in stages.items():
            if name in state["completed_stages"]:
                continue
            state.update(status="running", active_stage=name)
            self._save(state)
            try:
                state[name] = action()
            except Exception as exc:
                state.update(status="failed", error=str(exc))
                self._save(state)
                raise
            state["completed_stages"].append(name)
            state.pop("error", None)
            self._save(state)
        state.update(status="complete", active_stage=None)
        self._save(state)
        return state

    def generate(self):
        # Lazy imports keep describe/dry-run usable without simulator initialization.
        from lrl_molmospaces_extensions.datagen.per_door_rollout_runner import PerDoorRolloutRunner

        raw = self.root / "raw"
        if raw.exists():
            raise RuntimeError("Partial generation exists; preserve it and use a new run ID")
        module, name = self.config.generation_config.split(":", 1)
        config_cls = getattr(importlib.import_module(module), name)
        exp = config_cls()
        exp.output_dir = raw
        exp.seed = self.config.seed
        exp.num_workers = 1
        exp.filter_for_successful_trajectories = True
        exp.grounding_mode = "point_prompt" if self.config.point_prompt else "visible_unique"
        sampler = exp.task_sampler_config
        sampler.target_grounding_mode = exp.grounding_mode
        sampler.house_inds = self.config.house_inds
        sampler.required_door_action = self.config.required_action
        sampler.successes_per_door = self.config.successes_per_door
        sampler.max_rollout_attempts_per_door = self.config.max_attempts_per_door
        raw.mkdir()
        exp.save_config()
        successes, attempts = PerDoorRolloutRunner(exp).run()
        if not successes:
            raise RuntimeError("Generation produced no successful trajectories")
        return {"successes": successes, "attempts": attempts}

    def organize(self):
        recordings = self.root / "recordings"
        count = 0
        for house in sorted((self.root / "raw").glob("house_*")):
            house_id = int(house.name.split("_")[1])
            split = "val" if house_id in self.config.val_house_inds else "train"
            dest = recordings / split / house.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.is_symlink():
                dest.symlink_to(house.resolve(), target_is_directory=True)
            elif dest.resolve() != house.resolve():
                raise ValueError(f"Unexpected recording link: {dest}")
            count += len(list(house.glob("*.h5")))
        if not count:
            raise RuntimeError("No H5 recordings to prepare")
        return {"trajectories": count, "split_mode": "house"}

    def rerender(self):
        from lrl_molmospaces_extensions.baseline.rerender import BaselineRerender

        rows = BaselineRerender(
            self.root / "recordings", self.root / "dataset", self.config
        ).run()
        return {"newly_rendered": len(rows)}

    def validate(self):
        import cv2
        import h5py
        from lrl_molmospaces_extensions.rerender.slow_nav import CAMERAS, SlowNavRerender

        source = self.root / "recordings"
        dataset = self.root / "dataset"
        trajectories = list(source.glob("*/house_*/*.h5"))
        videos = 0
        for recording in trajectories:
            final = dataset / recording.relative_to(source)
            with h5py.File(final, "r") as src:
                frames = len(src["traj_0/obs/agent/qpos"])
                scene = json.loads(src["traj_0/obs_scene"][()].decode())
                if scene.get("baseline") != self.config.name:
                    raise ValueError(f"Missing baseline metadata: {final}")
                filenames = SlowNavRerender._camera_filenames(src)
                for camera in CAMERAS:
                    video = SlowNavRerender._output_video_path(final.parent, filenames[camera])
                    capture = cv2.VideoCapture(str(video))
                    try:
                        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                        valid = capture.isOpened()
                    finally:
                        capture.release()
                    if not valid or count != frames:
                        raise ValueError(f"Video/H5 mismatch: {video} ({count} vs {frames})")
                    for key in ("extrinsic_cv", "intrinsic_cv", "cam2world_gl"):
                        if len(src[f"traj_0/obs/sensor_param/{camera}/{key}"]) != frames:
                            raise ValueError(f"Camera metadata mismatch: {final}")
                    if self.config.point_prompt:
                        for key in ("points", "num_points"):
                            if len(src[f"traj_0/obs/extra/object_image_points/door_handle/{camera}/{key}"]) != frames:
                                raise ValueError(f"Point metadata mismatch: {final}")
                    videos += 1
        if not trajectories:
            raise RuntimeError("Empty dataset")
        return {"trajectories": len(trajectories), "camera_videos": videos}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard", choices=tuple(SHARDS))
    args = parser.parse_args()
    config = BaselineConfig.load(args.config)
    if args.shard:
        config = config.for_shard(args.shard)
    run_id = args.run_id + (f"_shard_{args.shard}" if args.shard else "")
    pipeline = BaselinePipeline(config, run_id)
    result = pipeline.describe() if args.dry_run else pipeline.run(resume=args.resume)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
