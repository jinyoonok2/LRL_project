"""Re-render recorded nav-then-open trajectories with shorter base steps."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import h5py
import mujoco
import numpy as np

CAMERAS = ("head_camera", "wrist_camera_r", "wrist_camera_l")
WRIST_CAMERAS = {"wrist_camera_r", "wrist_camera_l"}


def decode_row(dataset, index: int):
    text = dataset[index].tobytes().decode("utf-8").rstrip("\x00")
    if not text:
        return None
    return json.loads(text)


def encode_json(payload, width: int) -> np.ndarray:
    raw = json.dumps(payload).encode("utf-8")
    if len(raw) > width:
        raise ValueError(f"JSON length {len(raw)} exceeds field width {width}")
    buf = np.zeros(width, dtype=np.uint8)
    buf[: len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return buf


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def lerp_qpos(start: dict, end: dict, t: float) -> dict:
    mixed = {}
    for name, start_value in start.items():
        start_arr = np.asarray(start_value, dtype=np.float64)
        end_arr = np.asarray(end[name], dtype=np.float64)
        if name == "base":
            xy = (1.0 - t) * start_arr[:2] + t * end_arr[:2]
            yaw = start_arr[2] + t * wrap_angle(float(end_arr[2] - start_arr[2]))
            mixed[name] = np.array([xy[0], xy[1], yaw], dtype=np.float64)
        else:
            mixed[name] = (1.0 - t) * start_arr + t * end_arr
    return mixed


def segment_count(start: dict, end: dict, max_step_m: float, max_yaw_rad: float) -> int:
    start_base = np.asarray(start["base"], dtype=np.float64)
    end_base = np.asarray(end["base"], dtype=np.float64)
    distance = float(np.linalg.norm(end_base[:2] - start_base[:2]))
    yaw = abs(wrap_angle(float(end_base[2] - start_base[2])))
    return max(int(np.ceil(distance / max_step_m)), int(np.ceil(yaw / max_yaw_rad)), 1)


def qpos_delta(current: dict, nxt: dict) -> dict:
    delta = {}
    for name in ("base", "left_arm", "right_arm", "torso", "head"):
        if name not in current or name not in nxt:
            continue
        current_arr = np.asarray(current[name], dtype=np.float64)
        next_arr = np.asarray(nxt[name], dtype=np.float64)
        diff = next_arr - current_arr
        if name == "base":
            diff = diff.copy()
            diff[2] = wrap_angle(float(diff[2]))
        delta[name] = diff.tolist()
    return delta


def pose7(base: np.ndarray, z: float) -> np.ndarray:
    yaw = float(base[2])
    return np.array(
        [base[0], base[1], z, np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
        dtype=np.float32,
    )


def write_video(path: Path, frames: list[np.ndarray], fps: float = 10.0) -> None:
    import imageio_ffmpeg

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    process = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(np.ascontiguousarray(frame).tobytes())
    process.stdin.close()
    error = process.stderr.read().decode() if process.stderr is not None else ""
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {error[-2000:]}")


class SlowNavRerender:
    def __init__(
        self,
        dataset: Path,
        output: Path,
        max_step_m: float = 0.05,
        max_yaw_rad: float = 0.15,
        split: str | None = None,
        house: int | None = None,
        limit: int | None = None,
    ):
        self.dataset = dataset
        self.output = output
        self.max_step_m = max_step_m
        self.max_yaw_rad = max_yaw_rad
        self.split = split
        self.house = house
        self.limit = limit

    def run(self) -> list[dict]:
        grouped: dict[int, list[Path]] = {}
        for h5_path in self._iter_h5s():
            grouped.setdefault(int(h5_path.parent.name.split("_")[1]), []).append(h5_path)

        summaries = []
        skipped_total = 0
        for house_index, paths in grouped.items():
            pending = [path for path in paths if not self._output_h5(path).is_file()]
            skipped_total += len(paths) - len(pending)
            if not pending:
                print(json.dumps({"house": house_index, "skipped": len(paths)}), flush=True)
                continue
            env = self._load_house(house_index)
            defaults = self._door_defaults(env)
            print(json.dumps({"house": house_index, "doors": len(defaults), "pending": len(pending)}), flush=True)
            for h5_path in pending:
                summaries.append(self._rewrite_trajectory(env, defaults, h5_path))
            env.close()
        report = self.output / "rerender_report.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(summaries, indent=2) + "\n")
        print(
            f"wrote {len(summaries)} trajectories to {self.output} (skipped {skipped_total})",
            flush=True,
        )
        return summaries

    def _output_h5(self, h5_path: Path) -> Path:
        return self.output / h5_path.parents[1].name / h5_path.parent.name / h5_path.name

    def _iter_h5s(self):
        splits = [self.split] if self.split else ["train", "val"]
        count = 0
        for split_name in splits:
            houses = sorted(
                (self.dataset / split_name).glob("house_*"),
                key=lambda path: int(path.name.split("_")[1]),
            )
            for house_dir in houses:
                if self.house is not None and int(house_dir.name.split("_")[1]) != self.house:
                    continue
                for h5_path in sorted(house_dir.glob("*.h5")):
                    yield h5_path
                    count += 1
                    if self.limit is not None and count >= self.limit:
                        return

    @staticmethod
    def _load_house(house_index: int):
        from lrl_molmospaces_extensions.configs.door_opening import (
            RBY1NavDoorOpeningDebugConfig,
        )

        config = RBY1NavDoorOpeningDebugConfig()
        config.task_sampler_config.house_inds = [house_index]
        config.task_sampler_config.required_door_action = "both"
        sampler = config.task_sampler_config.task_sampler_class(config)
        task = sampler.sample_task(house_index=house_index)
        if task is None or "head_camera" not in sampler.env.camera_manager.registry:
            sampler.setup_cameras(sampler.env)
        print(
            f"loaded house {house_index} cameras={list(sampler.env.camera_manager.registry.keys())}",
            flush=True,
        )
        return sampler.env

    @staticmethod
    def _door_defaults(env) -> dict[str, float]:
        from molmo_spaces.env.data_views import Door

        names = env.object_managers[env.current_batch_index].find_door_names()
        defaults = {}
        for name in names:
            door = Door(name, env.current_data)
            if door.njoints == 0:
                continue
            hinge = door.get_hinge_joint_index()
            adr = door.get_joint_qpos_adr(hinge)
            defaults[name] = float(env.mj_model.qpos0[adr])
        return defaults

    def _rewrite_trajectory(self, env, defaults: dict[str, float], h5_path: Path) -> dict:
        split = h5_path.parents[1].name
        house = h5_path.parent.name
        out_house = self.output / split / house
        with h5py.File(h5_path, "r") as src:
            length = int(src["traj_0/obs/extra/policy_phase"].shape[0])
            qpos_rows = [decode_row(src["traj_0/obs/agent/qpos"], i) for i in range(length)]
            timeline = self._build_timeline(qpos_rows, self._segment_limits(src, length))
            filenames = self._camera_filenames(src)
            scene = json.loads(src["traj_0/obs_scene"][()].decode("utf-8"))
            door_name = scene["target_door_name"]
            door_angles = [
                float(np.asarray(decode_row(src["traj_0/obs/extra/door_state"], i)["joint_angle"]).reshape(-1)[0])
                for i in range(length)
            ]
            z_values = src["traj_0/obs/extra/robot_base_pose"][:, 2]
            action_width = int(src["traj_0/actions/joint_pos_rel"].shape[1])
            qpos_width = int(src["traj_0/obs/agent/qpos"].shape[1])
            joint_pos_rows = [decode_row(src["traj_0/actions/joint_pos"], i) for i in range(length)]
            rel_rows = [decode_row(src["traj_0/actions/joint_pos_rel"], i) for i in range(length)]

            rendered = 0
            video_frames = {camera: [] for camera in CAMERAS}
            qpos_patch = {}
            rel_patch = {}
            joint_patch = {}
            pose_patch = {}
            phase_patch = {}
            false_patch = {}
            reward_patch = {}
            frame_metadata = {}
            for index, record in enumerate(timeline):
                images = self._render_qpos(
                    env, record["qpos"], defaults, door_name, self._door_angle(record, door_angles)
                )
                rendered += 1
                for path, value in self._frame_metadata(env, images).items():
                    frame_metadata.setdefault(path, {})[index] = value
                for camera in CAMERAS:
                    video_frames[camera].append(images[camera])
                if record["qpos"] is qpos_rows[record["source"]]:
                    continue
                qpos_patch[index] = encode_json(
                    {name: np.asarray(value).tolist() for name, value in record["qpos"].items()},
                    qpos_width,
                )
                pose_patch[index] = pose7(
                    np.asarray(record["qpos"]["base"]), float(z_values[record["source"]])
                )
                phase_patch[index] = np.int64(0)
                reward_patch[index] = np.float32(0.0)
                false_patch[index] = np.bool_(False)
                if "delta" in record:
                    rel = dict(rel_rows[record["source"]] or {})
                    rel.update(record["delta"])
                    rel_patch[index] = encode_json(rel, action_width)
                    joint = dict(joint_pos_rows[record["source"]] or {})
                    for name, value in record["qpos"].items():
                        if name in joint:
                            joint[name] = np.asarray(value).tolist()
                    joint_patch[index] = encode_json(joint, action_width)

            patches = {
                "/traj_0/obs/agent/qpos": qpos_patch,
                "/traj_0/actions/joint_pos_rel": rel_patch,
                "/traj_0/actions/joint_pos": joint_patch,
                "/traj_0/obs/extra/robot_base_pose": pose_patch,
                "/traj_0/obs/extra/policy_phase": phase_patch,
                "/traj_0/rewards": reward_patch,
            }
            patches.update(frame_metadata)
            for flag in ("success", "fail", "terminated", "truncated"):
                patches[f"/traj_0/{flag}"] = false_patch

            out_h5 = out_house / h5_path.name
            out_house.mkdir(parents=True, exist_ok=True)
            with h5py.File(out_h5, "w") as dst:
                self._copy_tree(src, dst, length, timeline, patches)
            for camera, filename in filenames.items():
                write_video(self._output_video_path(out_house, filename), video_frames[camera])

        summary = {
            "h5": str(h5_path),
            "output": str(out_h5),
            "old_frames": length,
            "new_frames": len(timeline),
            "rendered_frames": rendered,
            "target_door": door_name,
            "default_doors": len(defaults),
            **getattr(self, "_match_info", {}),
        }
        print(json.dumps(summary), flush=True)
        return summary

    def _frame_metadata(self, env, images: dict) -> dict:
        """Optional observation updates for a newly rendered camera pose."""
        return {}

    def _segment_limits(self, src, length: int) -> list[tuple[float, float] | None]:
        return [(self.max_step_m, self.max_yaw_rad)] * (length - 1)

    def _build_timeline(
        self,
        qpos_rows: list[dict],
        limits: list[tuple[float, float] | None] | None = None,
    ) -> list[dict]:
        timeline = [{"source": 0, "qpos": qpos_rows[0], "rewrite": False, "alpha": 0.0}]
        for index in range(len(qpos_rows) - 1):
            limit = None if limits is None else limits[index]
            if limit is None and limits is not None:
                count = 1
            else:
                step_m, yaw = (self.max_step_m, self.max_yaw_rad) if limit is None else limit
                count = segment_count(qpos_rows[index], qpos_rows[index + 1], step_m, yaw)
            if count > 1:
                timeline[-1]["rewrite"] = True
                for step in range(1, count):
                    timeline.append(
                        {
                            "source": index,
                            "qpos": lerp_qpos(qpos_rows[index], qpos_rows[index + 1], step / count),
                            "rewrite": True,
                            "alpha": step / count,
                        }
                    )
            timeline.append(
                {"source": index + 1, "qpos": qpos_rows[index + 1], "rewrite": False, "alpha": 0.0}
            )
        for index, record in enumerate(timeline[:-1]):
            if record["rewrite"]:
                record["delta"] = qpos_delta(record["qpos"], timeline[index + 1]["qpos"])
        return timeline

    @staticmethod
    def _door_angle(record: dict, angles: list[float]) -> float:
        index = record["source"]
        alpha = float(record["alpha"])
        if alpha == 0.0 or index + 1 >= len(angles):
            return angles[index]
        return (1.0 - alpha) * angles[index] + alpha * angles[index + 1]

    @staticmethod
    def _render_qpos(env, qpos: dict, defaults: dict[str, float], door_name: str, angle: float):
        from molmo_spaces.env.data_views import Door

        robot = env.robots[0].robot_view
        robot.set_qpos_dict({name: np.asarray(value, dtype=np.float64) for name, value in qpos.items()})
        for name, default_angle in defaults.items():
            door = Door(name, env.current_data)
            door.set_joint_position(door.get_hinge_joint_index(), default_angle)
        if door_name in defaults:
            door = Door(door_name, env.current_data)
            door.set_joint_position(door.get_hinge_joint_index(), float(angle))
        mujoco.mj_forward(env.mj_model, env.current_data)
        env.camera_manager.registry.update_all_cameras(env)
        images = {}
        for name in CAMERAS:
            rgb = np.ascontiguousarray(env.render_rgb_frame(name))
            images[name] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return images

    @staticmethod
    def _camera_filenames(h5_file: h5py.File) -> dict[str, str]:
        names = {}
        sensors = h5_file["traj_0/obs/sensor_data"]
        for camera in CAMERAS:
            names[camera] = sensors[camera][()].tobytes().decode("utf-8").rstrip("\x00")
        return names

    @staticmethod
    def _output_video_path(house_dir: Path, filename: str) -> Path:
        name = Path(filename).name
        if any(camera in name for camera in WRIST_CAMERAS):
            return house_dir / "other_cameras" / name
        return house_dir / name

    @staticmethod
    def _copy_tree(src: h5py.Group, dst: h5py.Group, old_length: int, timeline: list[dict], patches: dict) -> None:
        new_length = len(timeline)
        for key, item in src.items():
            if isinstance(item, h5py.Group):
                SlowNavRerender._copy_tree(item, dst.create_group(key), old_length, timeline, patches)
                continue
            path = item.name
            if "depth" in path and "sensor_data" in path:
                continue
            data = item[()]
            if getattr(data, "shape", None) is not None and len(data.shape) >= 1 and data.shape[0] == old_length:
                new = np.empty((new_length,) + data.shape[1:], dtype=data.dtype)
                for index, record in enumerate(timeline):
                    new[index] = data[record["source"]]
                if path in patches:
                    for index, value in patches[path].items():
                        new[index] = value
                dst.create_dataset(key, data=new)
            else:
                dst.create_dataset(key, data=data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/bigtemp/xna8aw/nav_door/data/original"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/bigtemp/xna8aw/nav_door/data/slow_nav"),
    )
    parser.add_argument("--split", choices=("train", "val"))
    parser.add_argument("--house", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-step-m", type=float, default=0.05)
    parser.add_argument("--max-yaw-rad", type=float, default=0.15)
    args = parser.parse_args()
    SlowNavRerender(
        dataset=args.dataset,
        output=args.output,
        max_step_m=args.max_step_m,
        max_yaw_rad=args.max_yaw_rad,
        split=args.split,
        house=args.house,
        limit=args.limit,
    ).run()


if __name__ == "__main__":
    main()
