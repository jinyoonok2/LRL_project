"""Opening-motion measurement, independent of rendering and dataset I/O."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NAV_PHASES = frozenset({"navigate", "final_align", "door_opening:navigate_to_door"})
OPEN_PHASE = "door_opening:open_door"
MIN_STEP_M = 0.002
MIN_YAW_RAD = 0.005
FALLBACK_STEP_M = 0.013
FALLBACK_YAW_RAD = 0.015


def measure_opening_speed(
    phase_names: list[str], door_rows: list[dict]
) -> tuple[float, float, int]:
    """Median handle travel and hinge rate during ``door_opening:open_door``.

    Returns meters/frame, radians/frame, and the number of opening steps used.
    """
    handle_steps: list[float] = []
    angle_steps: list[float] = []
    for index, name in enumerate(phase_names[:-1]):
        if name != OPEN_PHASE:
            continue
        start = door_rows[index]
        end = door_rows[index + 1]
        handle_steps.append(
            float(
                np.linalg.norm(
                    np.asarray(end["handle_position"][:2], dtype=float)
                    - np.asarray(start["handle_position"][:2], dtype=float)
                )
            )
        )
        angle_steps.append(
            abs(
                float(np.asarray(end["joint_angle"]).reshape(-1)[0])
                - float(np.asarray(start["joint_angle"]).reshape(-1)[0])
            )
        )
    if not handle_steps:
        return FALLBACK_STEP_M, FALLBACK_YAW_RAD, 0
    return (
        max(float(np.median(handle_steps)), MIN_STEP_M),
        max(float(np.median(angle_steps)), MIN_YAW_RAD),
        len(handle_steps),
    )


def segment_limits(
    phase_names: list[str], step_m: float, yaw_rad: float
) -> list[tuple[float, float] | None]:
    return [
        (step_m, yaw_rad) if phase_names[index] in NAV_PHASES else None
        for index in range(len(phase_names) - 1)
    ]


@dataclass(frozen=True)
class OpeningSpeedProfile:
    """Measured opening motion and the navigation limits derived from it."""

    step_m: float
    yaw_rad: float
    opening_frames: int

    @classmethod
    def measure(cls, phase_names: list[str], door_rows: list[dict]) -> OpeningSpeedProfile:
        return cls(*measure_opening_speed(phase_names, door_rows))

    def navigation_limits(self, phase_names: list[str]) -> list[tuple[float, float] | None]:
        return segment_limits(phase_names, self.step_m, self.yaw_rad)

    def report_fields(self) -> dict:
        return {
            "opening_step_m": self.step_m,
            "opening_yaw_rad": self.yaw_rad,
            "opening_frames": self.opening_frames,
        }
