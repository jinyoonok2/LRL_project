"""Evaluation inputs matching the final trajectory fine-tuning run."""

from olmo.eval.configure_molmo_spaces import (
    MolmoBotRBY1DoorPlusOpenEvalConfig,
    MolmoBotRBY1DoorPlusOpenPolicyConfig,
)


class FinalTrajectoryPolicyConfig(MolmoBotRBY1DoorPlusOpenPolicyConfig):
    use_point_prompts: bool = False
    use_conditioning_image: bool = False
    cameras_to_warp: list[str] = []


class FinalTrajectoryDoorEvalConfig(MolmoBotRBY1DoorPlusOpenEvalConfig):
    policy_config: FinalTrajectoryPolicyConfig = FinalTrajectoryPolicyConfig()


# Register the composite task with the existing JSON benchmark sampler.
# Registration stays in extensions; upstream files are unchanged.
from molmo_spaces.tasks import json_eval_task_sampler as json_sampler
from molmo_spaces.evaluation.benchmark_schema import DoorOpeningTaskSpec
from lrl_molmospaces_extensions.tasks.nav_to_door_opening_task import NavToDoorOpeningTaskConfig

json_sampler.TASK_CLASS_TO_CONFIG_CLASS["NavToDoorOpeningTask"] = NavToDoorOpeningTaskConfig
json_sampler.TASK_CLASS_TO_SPEC_CLASS["NavToDoorOpeningTask"] = DoorOpeningTaskSpec


class FinalTrajectoryNavDoorEvalConfig(FinalTrajectoryDoorEvalConfig):
    """Model-controlled navigation and opening from recorded validation starts."""
