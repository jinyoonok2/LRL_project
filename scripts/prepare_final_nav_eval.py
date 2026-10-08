"""Build all held-out trajectory starts for model-controlled evaluation."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import h5py
import numpy as np


def prepare(output: Path):
    project = Path(__file__).resolve().parents[1]
    # Match the assets used to generate these demonstrations and by the
    # completed smoke evaluation, including pending jobs' submitted wrappers.
    os.environ["MLSPACES_ASSETS_DIR"] = str(project / "molmospaces/assets")
    spec = importlib.util.spec_from_file_location("benchmark_converter", project / "molmospaces/scripts/benchmarks/create_json_benchmark.py")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    root = project / "results/final/trajectory/val"
    train = {p.resolve() for p in (project / "results/final/trajectory/train").glob("house_*/*.h5")}
    episodes = []
    for path in sorted(root.glob("house_*/*.h5")):
        assert path.resolve() not in train, f"Train/validation overlap: {path}"
        with h5py.File(path, "r") as source:
            assert np.asarray(source["valid_traj_mask"]).all()
            for name in sorted(k for k in source if k.startswith("traj_")):
                trajectory = source[name]
                scene = json.loads(trajectory["obs_scene"][()].decode())
                frozen = converter.extract_frozen_config(scene)
                length = len(trajectory["obs/agent/qpos"])
                episode = converter.frozen_config_to_episode_spec(
                    frozen, scene, int(path.parent.name.split("_")[1]),
                    "procthor-10k", "train", str(path.resolve()), name, length,
                    (1024, 576), "RBY1GoProD455CameraSystem",
                ).model_dump()
                config = frozen.task_config if hasattr(frozen, "task_config") else frozen["task_config"]
                for field in ("door_body_name", "articulated_joint_range", "articulated_joint_reset_state", "door_openness_threshold"):
                    value = getattr(config, field) if not isinstance(config, dict) else config[field]
                    if hasattr(value, "tolist"):
                        value = value.tolist()
                    episode["task"][field] = value
                episode["task"].update(task_cls="lrl_molmospaces_extensions.tasks.nav_to_door_opening_task.NavToDoorOpeningTask", task_type="nav_to_door_opening")
                episode["seed"] = 42
                episodes.append(episode)
    assert len(episodes) == 90, f"Expected all 90 validation trajectories, got {len(episodes)}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark.json").write_text(json.dumps(episodes, indent=2) + "\n")
    (output / "benchmark_metadata.json").write_text(json.dumps({"episodes": 90, "split": "trajectory validation", "scene_split": "train", "description": "Recorded starts, model-controlled full navigation-to-door task; no expert action replay."}, indent=2) + "\n")
    print(f"Prepared all {len(episodes)} held-out trajectory starts at {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
