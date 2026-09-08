"""Entry point for long-horizon/per-door LRL data generation."""

import argparse
import datetime
import importlib
import os

from molmo_spaces.data_generation.config_registry import get_config_class

from lrl_molmospaces_extensions.datagen.per_door_rollout_runner import (
    PerDoorRolloutRunner,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "exp_config_cls",
        help="Importable module and registered config name: module:ConfigName",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()
    if ":" not in args.exp_config_cls:
        raise ValueError("Extension configs must be passed as module:ConfigName")

    module_name, config_name = args.exp_config_cls.split(":", 1)
    importlib.import_module(module_name)
    config_cls = get_config_class(config_name)
    exp_config = config_cls()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_config.output_dir = exp_config.output_dir / config_name / timestamp
    os.makedirs(exp_config.output_dir, exist_ok=True)
    exp_config.save_config()

    runner = PerDoorRolloutRunner(exp_config)
    success_count, total_count = runner.run()
    print(f"Success count: {success_count}, Total count: {total_count}")


if __name__ == "__main__":
    main()
