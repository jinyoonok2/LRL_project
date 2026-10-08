"""Re-render navigation so each base step matches that clip's door-opening speed.

Door opening is left at the recorded cuRobo frame rate. Only navigate,
final-align, and the door solver's approach phase are subdivided.
"""

from __future__ import annotations

import json
from pathlib import Path

from lrl_molmospaces_extensions.rerender.slow_nav import SlowNavRerender, decode_row

# Re-export helpers to preserve existing imports and scripts.
from lrl_molmospaces_extensions.rerender.opening_speed import (
    FALLBACK_STEP_M,
    FALLBACK_YAW_RAD,
    MIN_STEP_M,
    MIN_YAW_RAD,
    NAV_PHASES,
    OPEN_PHASE,
    OpeningSpeedProfile,
    measure_opening_speed,
    segment_limits,
)


class OpeningMatchedNavRerender(SlowNavRerender):
    """Slow navigation to the handle speed of this trajectory's opening phase."""

    def __init__(
        self,
        dataset: Path,
        output: Path,
        numbers: list[str] | None = None,
        opened_with_grasp: bool = False,
        **kwargs,
    ):
        super().__init__(dataset, output, **kwargs)
        self.numbers = {number.zfill(4) for number in numbers} if numbers else None
        self.opened_with_grasp = opened_with_grasp
        self._match_info: dict = {}

    def _iter_h5s(self):
        if self.opened_with_grasp and not self.numbers:
            self.numbers = self._opened_with_grasp_numbers()
        if not self.numbers:
            yield from super()._iter_h5s()
            return
        index = self.dataset / "numbered" / "index.tsv"
        found = set()
        for line in index.read_text().splitlines()[1:]:
            if not line.strip():
                continue
            number, split, house, checkpoint, *_ = line.split("\t")
            if number not in self.numbers:
                continue
            path = self.dataset / split / house / f"trajectories_batch_1_of_1_checkpoint_{checkpoint}.h5"
            if not path.is_file():
                raise FileNotFoundError(path)
            found.add(number)
            yield path
        missing = self.numbers - found
        if missing:
            raise FileNotFoundError(f"numbered clips not in {index}: {sorted(missing)}")

    def _opened_with_grasp_numbers(self) -> set[str]:
        index = self.dataset / "quality_check" / "index.tsv"
        numbers = set()
        for line in index.read_text().splitlines()[1:]:
            if not line.strip():
                continue
            number, label, *_ = line.split("\t")
            if label == "opened_with_grasp":
                numbers.add(number)
        if not numbers:
            raise ValueError(f"no opened_with_grasp clips in {index}")
        return numbers

    def _segment_limits(self, src, length: int) -> list[tuple[float, float] | None]:
        scene = json.loads(src["traj_0/obs_scene"][()].decode("utf-8"))
        names = {value: key for key, value in scene["policy_phases"].items()}
        phases = src["traj_0/obs/extra/policy_phase"][:]
        phase_names = [names.get(int(value), "unknown") for value in phases]
        door_rows = [decode_row(src["traj_0/obs/extra/door_state"], i) for i in range(length)]
        profile = OpeningSpeedProfile.measure(phase_names, door_rows)
        self._match_info = profile.report_fields()
        return profile.navigation_limits(phase_names)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-render navigation at each trajectory's door-opening speed."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/bigtemp/xna8aw/nav_door/data/original"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/bigtemp/xna8aw/nav_door/data/opening_matched_nav"),
    )
    parser.add_argument("--split", choices=("train", "val"))
    parser.add_argument("--house", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--opened-with-grasp",
        action="store_true",
        help="Render every opened-with-grasp clip and skip grasp failures.",
    )
    parser.add_argument(
        "--numbers",
        nargs="+",
        help="Numbered clip ids to render, for example 0311 0840.",
    )
    args = parser.parse_args()
    OpeningMatchedNavRerender(
        dataset=args.dataset,
        output=args.output,
        split=args.split,
        house=args.house,
        limit=args.limit,
        numbers=args.numbers,
        opened_with_grasp=args.opened_with_grasp,
    ).run()


if __name__ == "__main__":
    main()
