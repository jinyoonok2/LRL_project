from lrl_molmospaces_extensions.rerender.opening_matched_nav import (
    measure_opening_speed,
    segment_limits,
)
from lrl_molmospaces_extensions.rerender.slow_nav import SlowNavRerender


def _door(handle_x: float, angle: float) -> dict:
    return {"handle_position": [handle_x, 0.0, 1.0], "joint_angle": [angle]}


def test_opening_speed_uses_open_phase_only():
    phases = ["navigate", "door_opening:open_door", "door_opening:open_door", "done"]
    doors = [_door(0.0, 0.0), _door(0.30, 0.2), _door(0.31, 0.21), _door(0.32, 0.22)]
    step_m, yaw, frames = measure_opening_speed(phases, doors)
    assert frames == 2
    assert abs(step_m - 0.01) < 1e-9
    assert abs(yaw - 0.01) < 1e-9


def test_navigation_segments_are_the_only_ones_slowed():
    phases = ["navigate", "door_opening:open_door", "done"]
    limits = segment_limits(phases, 0.01, 0.01)
    assert limits == [(0.01, 0.01), None]


def test_timeline_keeps_opening_steps_intact():
    rerender = SlowNavRerender.__new__(SlowNavRerender)
    qpos = [
        {"base": [0.0, 0.0, 0.0]},
        {"base": [0.2, 0.0, 0.0]},
        {"base": [0.2, 0.0, 0.0]},
    ]
    timeline = rerender._build_timeline(qpos, [(0.05, 0.15), None])
    sources = [record["source"] for record in timeline]
    assert sources[0] == 0
    assert sources[-1] == 2
    assert sources.count(1) == 1
    assert len(timeline) > 3
