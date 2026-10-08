from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


def _decode_row(dataset, index: int):
    text = dataset[index].tobytes().decode("utf-8").rstrip("\x00")
    if not text:
        return None
    return json.loads(text)


@dataclass(frozen=True)
class DoorOpeningEpisode:
    path: Path
    frames: int
    success: bool
    first_success_frame: int | None
    grasp_frames: int
    first_grasp_frame: int | None
    last_grasp_frame: int | None
    dt_s: float = 0.1

    @classmethod
    def from_h5(cls, path: Path) -> DoorOpeningEpisode:
        with h5py.File(path, "r") as src:
            success_flags = src["traj_0/success"][:]
            frames = int(success_flags.shape[0])
            success = bool(np.any(success_flags))
            first_success = int(np.argmax(success_flags)) if success else None
            grasp = cls._grasp_mask(src, frames)
            grasp_idx = np.flatnonzero(grasp)
            return cls(
                path=path,
                frames=frames,
                success=success,
                first_success_frame=first_success,
                grasp_frames=int(grasp.sum()),
                first_grasp_frame=int(grasp_idx[0]) if len(grasp_idx) else None,
                last_grasp_frame=int(grasp_idx[-1]) if len(grasp_idx) else None,
            )

    @staticmethod
    def _grasp_mask(src: h5py.File, frames: int) -> np.ndarray:
        mask = np.zeros(frames, dtype=bool)
        left = src.get("traj_0/obs/extra/rby1_left_grasp_state")
        right = src.get("traj_0/obs/extra/rby1_right_grasp_state")
        if left is not None:
            mask |= np.asarray(left).reshape(frames) > 0
        if right is not None:
            mask |= np.asarray(right).reshape(frames) > 0
        info = src.get("traj_0/obs/extra/task_info")
        if info is not None:
            for index in range(frames):
                row = _decode_row(info, index) or {}
                if bool(row.get("handle_grasped")):
                    mask[index] = True
        return mask

    @property
    def first_success_s(self) -> float | None:
        if self.first_success_frame is None:
            return None
        return self.first_success_frame * self.dt_s
