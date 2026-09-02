#!/usr/bin/env python3
"""Launch nav-to-door-opening data generation with selectable target grounding."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


CONFIG_BY_MODE = {
    "none": "RBY1NavDoorOpeningHandoffSmokeConfig",
    "visible_unique": "RBY1NavDoorOpeningVisibleGroundingSmokeConfig",
    "point_prompt": "RBY1NavDoorOpeningPointPromptGroundingSmokeConfig",
    "room_door_id": "RBY1NavDoorOpeningRoomDoorIdGroundingSmokeConfig",
}
CONFIG_MODULE = "molmo_spaces.data_generation.config.door_opening_configs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grounding-mode",
        choices=sorted(CONFIG_BY_MODE),
        default="visible_unique",
        help="Target-door grounding strategy.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional registered nav-door config name; defaults to a smoke config for the mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without running it.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    molmospaces_root = project_root / "molmospaces"
    config_name = args.config or CONFIG_BY_MODE[args.grounding_mode]
    config_ref = f"{CONFIG_MODULE}:{config_name}"
    command = [
        sys.executable,
        "-m",
        "molmo_spaces.data_generation.main",
        config_ref,
    ]

    env = os.environ.copy()
    env["RBY1_NAV_DOOR_GROUNDING_MODE"] = args.grounding_mode
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{molmospaces_root}:{current_pythonpath}"
        if current_pythonpath
        else str(molmospaces_root)
    )

    if args.dry_run:
        print(f"RBY1_NAV_DOOR_GROUNDING_MODE={args.grounding_mode}")
        print(" ".join(command))
        return

    subprocess.run(command, cwd=molmospaces_root, env=env, check=True)


if __name__ == "__main__":
    main()
