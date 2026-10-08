from __future__ import annotations

import argparse
from pathlib import Path

from lrl_molmospaces_extensions.quality_check.opened_without_grasp import (
    OpenedWithoutGraspCheck,
)

DEFAULT_DATASETS = (
    Path("/bigtemp/xna8aw/nav_door/data/final"),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="File successful door-open videos by whether the handle was grasped."
    )
    parser.add_argument("--dataset", type=Path, action="append")
    parser.add_argument("--min-grasp-frames", type=int, default=30)
    args = parser.parse_args()
    datasets = args.dataset or list(DEFAULT_DATASETS)
    for dataset in datasets:
        OpenedWithoutGraspCheck(dataset, min_grasp_frames=args.min_grasp_frames).run()


if __name__ == "__main__":
    main()
