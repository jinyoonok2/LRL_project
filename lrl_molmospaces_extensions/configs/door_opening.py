from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.policy_configs import DoorOpeningPolicyConfig
from molmo_spaces.data_generation.config.door_opening_configs import DoorOpeningDataGenConfig
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.tasks.opening_tasks import DoorOpeningTask

from lrl_molmospaces_extensions.configs.policy_configs import NavThenDoorOpeningPolicyConfig
from lrl_molmospaces_extensions.policy.nav_then_door_opening_policy import NavDoorAStarSmoothPlannerPolicy
from lrl_molmospaces_extensions.tasks.nav_to_door_opening_task import NavToDoorOpeningTask, NavToDoorOpeningTaskConfig
from lrl_molmospaces_extensions.tasks.nav_to_door_opening_task_sampler import NavToDoorOpeningTaskSampler, NavToDoorOpeningTaskSamplerConfig, SinglePanelDoorOpeningTaskSampler

@register_config("RBY1NavDoorOpeningDataGenConfig")
class RBY1NavDoorOpeningDataGenConfig(DoorOpeningDataGenConfig):
    """Prototype long-horizon RBY1 navigation followed by door opening."""

    task_type: str = "nav_to_door_opening"
    grounding_mode: Literal["none", "visible_unique", "point_prompt"] = "none"
    task_horizon: int = 1200
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "rby1_nav_door_opening_v1"
    )
    policy_config: NavThenDoorOpeningPolicyConfig | None = None
    task_sampler_config: NavToDoorOpeningTaskSamplerConfig = (
        NavToDoorOpeningTaskSamplerConfig(
            task_sampler_class=NavToDoorOpeningTaskSampler
        )
    )
    task_config: NavToDoorOpeningTaskConfig = NavToDoorOpeningTaskConfig(
        task_cls=NavToDoorOpeningTask
    )

    def _init_policy_config(self) -> NavThenDoorOpeningPolicyConfig:
        opening_policy_config = DoorOpeningDataGenConfig._init_policy_config(self)
        opening_policy_config.max_steps_per_waypoint = 30
        nav_policy_config = NavThenDoorOpeningPolicyConfig().nav_policy_config
        nav_policy_config.path_interpolation_density = 0
        nav_policy_config.path_max_inter_waypoint_dist = 0.5
        nav_policy_config.intermediate_waypoint_xy_threshold_m = 0.25
        nav_policy_config.path_min_dist_to_target_center = 0.75
        nav_policy_config.plan_max_retries = 3
        nav_policy_config.plan_fail_after_waypoint_steps = 20
        nav_policy_config.planner_config.agent_radius = 0.5
        nav_policy_config.policy_cls = NavDoorAStarSmoothPlannerPolicy

        return NavThenDoorOpeningPolicyConfig(
            opening_policy_config=opening_policy_config,
            nav_policy_config=nav_policy_config,
        )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        grounding_mode = os.environ.get(
            "RBY1_NAV_DOOR_GROUNDING_MODE", self.grounding_mode
        )
        if grounding_mode not in {"none", "visible_unique", "point_prompt"}:
            raise ValueError(f"Invalid nav-door grounding mode: {grounding_mode}")
        self.grounding_mode = grounding_mode
        self.task_config.task_cls = NavToDoorOpeningTask
        self.task_sampler_config.task_sampler_class = NavToDoorOpeningTaskSampler
        self.task_sampler_config.target_grounding_mode = grounding_mode
        self.task_sampler_config.robot_safety_radius = 0.35
        self.task_sampler_config.base_pose_sampling_radius_range = (4.0, 12.0)
        self.task_sampler_config.max_robot_placement_attempts = 25

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_datagen"


@register_config("RBY1NavDoorOpeningDebugConfig")
class RBY1NavDoorOpeningDebugConfig(RBY1NavDoorOpeningDataGenConfig):
    """One-episode debug config for nav-to-door-opening validation."""

    num_workers: int = 1
    use_passive_viewer: bool = False
    filter_for_successful_trajectories: bool = False
    seed: int | None = 83067780
    policy_dt_ms: float = 100.0
    task_horizon: int = 1200
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "rby1_nav_door_opening_debug"
    )
    task_sampler_config: NavToDoorOpeningTaskSamplerConfig = (
        NavToDoorOpeningTaskSamplerConfig(
        task_sampler_class=NavToDoorOpeningTaskSampler,
        samples_per_house=1,
        house_inds=[22],
        base_pose_sampling_radius_range=(1.5, 3.0),
        robot_safety_radius=0.35,
        max_robot_placement_attempts=25,
        )
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (1.5, 3.0)
        self.task_sampler_config.robot_safety_radius = 0.35
        self.task_sampler_config.max_robot_placement_attempts = 25

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_debug"


@register_config("RBY1NavDoorOpeningSuccessSearchConfig")
class RBY1NavDoorOpeningSuccessSearchConfig(RBY1NavDoorOpeningDataGenConfig):
    """Small sweep that searches for successful nav-to-door-opening rollouts."""

    num_workers: int = 1
    use_passive_viewer: bool = False
    filter_for_successful_trajectories: bool = True
    seed: int | None = None
    policy_dt_ms: float = 100.0
    task_horizon: int = 1200
    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_success_search"
    )
    task_sampler_config: NavToDoorOpeningTaskSamplerConfig = (
        NavToDoorOpeningTaskSamplerConfig(
        task_sampler_class=NavToDoorOpeningTaskSampler,
        samples_per_house=1,
        house_inds=[0, 1, 2, 3, 4, 5, 6, 7, 22],
        base_pose_sampling_radius_range=(0.8, 2.0),
        robot_safety_radius=0.35,
        max_robot_placement_attempts=35,
        max_total_attempts_multiplier=4,
        check_robot_placement_visibility=False,
        )
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (0.8, 2.0)
        self.task_sampler_config.robot_safety_radius = 0.35
        self.task_sampler_config.max_robot_placement_attempts = 35
        self.task_sampler_config.max_total_attempts_multiplier = 4
        self.task_sampler_config.check_robot_placement_visibility = False

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_success_search"


@register_config("RBY1NavDoorOpeningFarStandoffSearchConfig")
class RBY1NavDoorOpeningFarStandoffSearchConfig(RBY1NavDoorOpeningSuccessSearchConfig):
    """Success search variant that hands off farther from the door handle."""

    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_far_standoff_search"
    )

    def _init_policy_config(self) -> NavThenDoorOpeningPolicyConfig:
        policy_config = super()._init_policy_config()
        policy_config.handoff_max_distance_to_handle_m = 1.45
        policy_config.handoff_after_nav_failure_max_distance_to_handle_m = 1.6
        policy_config.nav_policy_config.path_min_dist_to_target_center = 1.25
        return policy_config

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (1.2, 3.0)

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_far_standoff_search"


@register_config("RBY1NavDoorOpeningLongDistanceSearchConfig")
class RBY1NavDoorOpeningLongDistanceSearchConfig(RBY1NavDoorOpeningSuccessSearchConfig):
    """Long-distance success search for nav-to-door-opening rollouts."""

    task_horizon: int = 1600
    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_long_distance_search"
    )

    def _init_policy_config(self) -> NavThenDoorOpeningPolicyConfig:
        policy_config = super()._init_policy_config()
        policy_config.handoff_max_distance_to_handle_m = 0.95
        policy_config.handoff_after_nav_failure_max_distance_to_handle_m = 1.15
        policy_config.final_align_target_distance_to_handle_m = 0.85
        policy_config.final_align_max_steps = 60
        policy_config.nav_policy_config.path_min_dist_to_target_center = 0.95
        policy_config.nav_policy_config.plan_max_retries = 5
        policy_config.nav_policy_config.plan_fail_after_waypoint_steps = 30
        return policy_config

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (4.0, 12.0)
        self.task_sampler_config.max_total_attempts_multiplier = 3
        self.task_sampler_config.house_inds = [1, 4, 7, 22]

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_long_distance_search"


@register_config("RBY1NavDoorOpeningHandoffSmokeConfig")
class RBY1NavDoorOpeningHandoffSmokeConfig(RBY1NavDoorOpeningDataGenConfig):
    """Small end-to-end smoke test for navigation handoff and final alignment."""

    num_workers: int = 1
    use_passive_viewer: bool = False
    filter_for_successful_trajectories: bool = False
    seed: int | None = 83067780
    policy_dt_ms: float = 100.0
    task_horizon: int = 1200
    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_handoff_smoke"
    )
    task_sampler_config: NavToDoorOpeningTaskSamplerConfig = (
        NavToDoorOpeningTaskSamplerConfig(
        task_sampler_class=NavToDoorOpeningTaskSampler,
        samples_per_house=1,
        house_inds=[22],
        base_pose_sampling_radius_range=(1.5, 3.0),
        robot_safety_radius=0.35,
        max_robot_placement_attempts=25,
        check_robot_placement_visibility=False,
        )
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (1.5, 3.0)
        self.task_sampler_config.house_inds = [22]
        self.task_sampler_config.max_total_attempts_multiplier = 1
        self.task_sampler_config.check_robot_placement_visibility = False

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_handoff_smoke"


@register_config("RBY1NavDoorOpeningVisibleGroundingSmokeConfig")
class RBY1NavDoorOpeningVisibleGroundingSmokeConfig(
    RBY1NavDoorOpeningHandoffSmokeConfig
):
    """Stage-1 smoke test with a unique visible target door."""

    grounding_mode: Literal["none", "visible_unique", "point_prompt"] = "visible_unique"
    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_visible_grounding_smoke"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        sampler_config = self.task_sampler_config
        sampler_config.target_visibility_camera = "head_camera"
        sampler_config.target_door_min_visibility_fraction = 0.0001
        sampler_config.target_handle_min_visibility_fraction = 0.00001
        sampler_config.competing_door_min_visibility_fraction = 0.0001
        sampler_config.max_robot_placement_attempts = 50

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_visible_grounding_smoke"


@register_config("RBY1NavDoorOpeningPointPromptGroundingSmokeConfig")
class RBY1NavDoorOpeningPointPromptGroundingSmokeConfig(
    RBY1NavDoorOpeningVisibleGroundingSmokeConfig
):
    """Stage-2 smoke test with a visible target-handle point prompt."""

    grounding_mode: Literal["none", "visible_unique", "point_prompt"] = "point_prompt"
    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_point_prompt_grounding_smoke"
    )

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_point_prompt_grounding_smoke"


class RBY1NavDoorOpeningDistanceSweepBaseConfig(RBY1NavDoorOpeningDataGenConfig):
    """Base config for small distance-specific nav-to-door-opening sweeps."""

    num_workers: int = 1
    use_passive_viewer: bool = False
    filter_for_successful_trajectories: bool = True
    seed: int | None = None
    policy_dt_ms: float = 100.0
    task_horizon: int = 1400
    task_sampler_config: NavToDoorOpeningTaskSamplerConfig = (
        NavToDoorOpeningTaskSamplerConfig(
        task_sampler_class=NavToDoorOpeningTaskSampler,
        samples_per_house=1,
        house_inds=[1, 4, 7, 22],
        base_pose_sampling_radius_range=(0.8, 2.0),
        robot_safety_radius=0.35,
        max_robot_placement_attempts=35,
        max_total_attempts_multiplier=3,
        check_robot_placement_visibility=False,
        )
    )

    def _init_policy_config(self) -> NavThenDoorOpeningPolicyConfig:
        policy_config = super()._init_policy_config()
        policy_config.handoff_max_distance_to_handle_m = 0.95
        policy_config.handoff_after_nav_failure_max_distance_to_handle_m = 1.15
        policy_config.final_align_target_distance_to_handle_m = 0.85
        policy_config.final_align_max_steps = 60
        policy_config.nav_policy_config.path_min_dist_to_target_center = 0.95
        policy_config.nav_policy_config.plan_max_retries = 5
        policy_config.nav_policy_config.plan_fail_after_waypoint_steps = 30
        return policy_config

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.house_inds = [1, 4, 7, 22]
        self.task_sampler_config.max_robot_placement_attempts = 35
        self.task_sampler_config.max_total_attempts_multiplier = 3
        self.task_sampler_config.check_robot_placement_visibility = False


@register_config("RBY1NavDoorOpeningShortDistanceSweepConfig")
class RBY1NavDoorOpeningShortDistanceSweepConfig(RBY1NavDoorOpeningDistanceSweepBaseConfig):
    """Distance sweep with 0.8-2.0m starts."""

    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_distance_short"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (0.8, 2.0)

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_distance_short"


@register_config("RBY1NavDoorOpeningMediumDistanceSweepConfig")
class RBY1NavDoorOpeningMediumDistanceSweepConfig(RBY1NavDoorOpeningDistanceSweepBaseConfig):
    """Distance sweep with 2.0-4.0m starts."""

    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_distance_medium"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (2.0, 4.0)

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_distance_medium"


@register_config("RBY1NavDoorOpeningLongDistanceSweepConfig")
class RBY1NavDoorOpeningLongDistanceSweepConfig(RBY1NavDoorOpeningDistanceSweepBaseConfig):
    """Distance sweep with 4.0-8.0m starts."""

    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_distance_long"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (4.0, 8.0)

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_distance_long"


class RBY1NavDoorOpeningBalancedSweepBaseConfig(RBY1NavDoorOpeningDistanceSweepBaseConfig):
    """Balanced handoff variant that prioritizes orientation over close standoff."""

    def _init_policy_config(self) -> NavThenDoorOpeningPolicyConfig:
        policy_config = super()._init_policy_config()
        policy_config.handoff_max_distance_to_handle_m = 1.25
        policy_config.handoff_after_nav_failure_max_distance_to_handle_m = 1.4
        policy_config.final_align_target_distance_to_handle_m = 1.1
        policy_config.final_align_yaw_threshold_rad = float(np.deg2rad(15))
        policy_config.nav_policy_config.path_min_dist_to_target_center = 1.1
        return policy_config


@register_config("RBY1NavDoorOpeningMediumDistanceBalancedSweepConfig")
class RBY1NavDoorOpeningMediumDistanceBalancedSweepConfig(
    RBY1NavDoorOpeningBalancedSweepBaseConfig
):
    """Balanced-handoff sweep with 2.0-4.0m starts."""

    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_distance_medium_balanced"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (2.0, 4.0)

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_distance_medium_balanced"


@register_config("RBY1NavDoorOpeningLongDistanceBalancedSweepConfig")
class RBY1NavDoorOpeningLongDistanceBalancedSweepConfig(
    RBY1NavDoorOpeningBalancedSweepBaseConfig
):
    """Balanced-handoff sweep with 4.0-8.0m starts."""

    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_distance_long_balanced"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (4.0, 8.0)

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_distance_long_balanced"


@register_config("RBY1NavDoorOpeningSimpleVisibleDataGenConfig")
class RBY1NavDoorOpeningSimpleVisibleDataGenConfig(
    RBY1NavDoorOpeningBalancedSweepBaseConfig
):
    """Focused dataset with one clearly visible door and 0.8-3.0m starts."""

    grounding_mode: Literal["none", "visible_unique", "point_prompt"] = "visible_unique"
    task_horizon: int = 1200
    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_simple_visible"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        sampler_config = self.task_sampler_config
        sampler_config.base_pose_sampling_radius_range = (0.8, 3.0)
        sampler_config.samples_per_house = 2
        sampler_config.max_total_attempts_multiplier = 4
        sampler_config.max_robot_placement_attempts = 50
        sampler_config.target_visibility_camera = "head_camera"
        sampler_config.target_door_min_visibility_fraction = 0.001
        sampler_config.competing_door_min_visibility_fraction = 0.001

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_simple_visible"


@register_config("RBY1NavDoorOpeningSimpleVisibleSmokeConfig")
class RBY1NavDoorOpeningSimpleVisibleSmokeConfig(
    RBY1NavDoorOpeningSimpleVisibleDataGenConfig
):
    """One-episode smoke test for the simplified visible-door task."""

    filter_for_successful_trajectories: bool = False
    seed: int | None = 83067780
    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_simple_visible_smoke"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        sampler_config = self.task_sampler_config
        sampler_config.house_inds = [22]
        sampler_config.samples_per_house = 1
        sampler_config.max_total_attempts_multiplier = 1

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_simple_visible_smoke"


class RBY1NavDoorOpeningSimpleVisibleProductionBaseConfig(
    RBY1NavDoorOpeningSimpleVisibleDataGenConfig
):
    """Production settings for collecting successful simple visible-door trajectories."""

    filter_for_successful_trajectories: bool = True
    seed: int | None = None
    task_horizon: int = 1200
    trajectory_checkpoint_interval: int = 1

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        sampler_config = self.task_sampler_config
        sampler_config.samples_per_house = 3
        sampler_config.max_total_attempts_multiplier = 6
        sampler_config.max_robot_placement_attempts = 75
        sampler_config.enumerate_all_doors = True
        sampler_config.single_panel_doors_only = True
        sampler_config.successes_per_door = 3
        sampler_config.max_rollout_attempts_per_door = 12
        sampler_config.max_house_episode_attempts = 1000


@register_config("RBY1NavDoorOpeningSimpleVisibleProductionAConfig")
class RBY1NavDoorOpeningSimpleVisibleProductionAConfig(
    RBY1NavDoorOpeningSimpleVisibleProductionBaseConfig
):
    """Production shard A covering ProcTHOR houses 0-10."""

    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_simple_visible_production_a"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.house_inds = list(range(0, 11))

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_simple_visible_production_a"


@register_config("RBY1NavDoorOpeningSimpleVisibleProductionBConfig")
class RBY1NavDoorOpeningSimpleVisibleProductionBConfig(
    RBY1NavDoorOpeningSimpleVisibleProductionBaseConfig
):
    """Production shard B covering ProcTHOR houses 11-22."""

    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_simple_visible_production_b"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.house_inds = list(range(11, 23))

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_simple_visible_production_b"


class RBY1NavDoorOpeningPointPromptSweepBaseConfig(
    RBY1NavDoorOpeningBalancedSweepBaseConfig
):
    """Balanced success-search sweep with visible target-handle point grounding."""

    grounding_mode: Literal["none", "visible_unique", "point_prompt"] = "point_prompt"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        sampler_config = self.task_sampler_config
        sampler_config.target_visibility_camera = "head_camera"
        sampler_config.target_door_min_visibility_fraction = 0.0001
        sampler_config.target_handle_min_visibility_fraction = 0.00001
        sampler_config.max_robot_placement_attempts = 50


@register_config("RBY1NavDoorOpeningMediumDistancePointPromptSweepConfig")
class RBY1NavDoorOpeningMediumDistancePointPromptSweepConfig(
    RBY1NavDoorOpeningPointPromptSweepBaseConfig
):
    """Point-prompt success sweep with 2.0-4.0m starts."""

    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_point_prompt_medium"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (2.0, 4.0)

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_point_prompt_medium"


@register_config("RBY1NavDoorOpeningLongDistancePointPromptSweepConfig")
class RBY1NavDoorOpeningLongDistancePointPromptSweepConfig(
    RBY1NavDoorOpeningPointPromptSweepBaseConfig
):
    """Point-prompt success sweep with 4.0-8.0m starts."""

    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_point_prompt_long"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (4.0, 8.0)

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_point_prompt_long"


@register_config("RBY1NavDoorOpeningVeryLongDistanceSweepConfig")
class RBY1NavDoorOpeningVeryLongDistanceSweepConfig(RBY1NavDoorOpeningDistanceSweepBaseConfig):
    """Distance sweep with 8.0-12.0m starts."""

    task_horizon: int = 1800
    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_distance_very_long"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (8.0, 12.0)

    @property
    def tag(self) -> str:
        return "rby1_nav_door_opening_distance_very_long"


def _production_shard_config(name: str, start: int, stop: int):
    """Create one registered production shard without modifying upstream configs."""

    def model_post_init(self, __context) -> None:
        super(config_cls, self).model_post_init(__context)
        self.task_sampler_config.house_inds = list(range(start, stop))

    config_cls = type(
        name,
        (RBY1NavDoorOpeningSimpleVisibleProductionBaseConfig,),
        {
            "__module__": __name__,
            "__annotations__": {"output_dir": Path},
            "output_dir": (
                ASSETS_DIR
                / "experiment_output"
                / "datagen"
                / f"rby1_nav_door_opening_simple_visible_production_{name[-7].lower()}"
            ),
            "model_post_init": model_post_init,
        },
    )
    return register_config(name)(config_cls)


RBY1NavDoorOpeningSimpleVisibleProductionCConfig = _production_shard_config(
    "RBY1NavDoorOpeningSimpleVisibleProductionCConfig", 23, 42
)
RBY1NavDoorOpeningSimpleVisibleProductionDConfig = _production_shard_config(
    "RBY1NavDoorOpeningSimpleVisibleProductionDConfig", 42, 61
)
RBY1NavDoorOpeningSimpleVisibleProductionEConfig = _production_shard_config(
    "RBY1NavDoorOpeningSimpleVisibleProductionEConfig", 61, 80
)
RBY1NavDoorOpeningSimpleVisibleProductionFConfig = _production_shard_config(
    "RBY1NavDoorOpeningSimpleVisibleProductionFConfig", 80, 100
)


SINGLE_PANEL_TEN_DOOR_HOUSES = [0, 2, 3, 11, 12, 13, 14, 15, 21, 23]
STABILITY_SHARD_HOUSE_INDS = {
    "a": list(range(0, 11)),
    "b": list(range(11, 23)),
    "c": list(range(23, 42)),
    "d": list(range(42, 61)),
    "e": list(range(61, 80)),
    "f": list(range(80, 100)),
}


def _configure_stability_sampler(
    sampler_config: NavToDoorOpeningTaskSamplerConfig,
    house_inds: list[int],
) -> None:
    attempts_per_door = int(
        os.environ.get("RBY1_DOOR_STABILITY_ATTEMPTS_PER_DOOR", "10")
    )
    if not 1 <= attempts_per_door <= 20:
        raise ValueError(
            "RBY1_DOOR_STABILITY_ATTEMPTS_PER_DOOR must be between 1 and 20"
        )
    required_action = os.environ.get(
        "RBY1_DOOR_STABILITY_ACTION", "both"
    ).lower()
    if required_action not in {"both", "push", "pull"}:
        raise ValueError(
            "RBY1_DOOR_STABILITY_ACTION must be one of both, push, or pull"
        )

    sampler_config.house_inds = house_inds
    sampler_config.samples_per_house = attempts_per_door
    sampler_config.max_total_attempts_multiplier = 1
    sampler_config.enumerate_all_doors = True
    sampler_config.single_panel_doors_only = True
    sampler_config.fixed_rollout_attempts_per_door = attempts_per_door
    sampler_config.max_rollout_attempts_per_door = attempts_per_door
    sampler_config.max_house_episode_attempts = 1000
    sampler_config.max_allowed_sequential_task_sampler_failures = 100
    sampler_config.max_allowed_sequential_rollout_failures = attempts_per_door + 5
    sampler_config.required_door_action = required_action
    sampler_config.max_robot_placement_attempts = 50
    sampler_config.max_placement_failure_batches_per_door = 20


def _configure_stability_experiment(
    exp_config: MlSpacesExpConfig,
    house_inds: list[int],
    shard: str | None = None,
) -> None:
    _configure_stability_sampler(exp_config.task_sampler_config, house_inds)
    suffixes = [exp_config.task_sampler_config.required_door_action]
    if shard is not None:
        suffixes.append(f"shard_{shard}")
    exp_config.output_dir = exp_config.output_dir.with_name(
        f"{exp_config.output_dir.name}_{'_'.join(suffixes)}"
    )


def _all_or_shard_stability_houses() -> tuple[list[int], str | None]:
    shard = os.environ.get("RBY1_DOOR_STABILITY_SHARD")
    if shard is None:
        return list(range(100)), None
    shard = shard.lower()
    if shard not in STABILITY_SHARD_HOUSE_INDS:
        raise ValueError(
            "RBY1_DOOR_STABILITY_SHARD must be one of a, b, c, d, e, or f"
        )
    return STABILITY_SHARD_HOUSE_INDS[shard], shard


class RBY1DoorOpeningSinglePanelTestBaseConfig(DoorOpeningDataGenConfig):
    """Door-only fixed-attempt evaluation using the upstream opening policy."""

    filter_for_successful_trajectories: bool = False
    seed: int | None = None
    task_horizon: int = 1000
    trajectory_checkpoint_interval: int = 1
    task_config: NavToDoorOpeningTaskConfig = NavToDoorOpeningTaskConfig(
        task_cls=DoorOpeningTask
    )
    task_sampler_config: NavToDoorOpeningTaskSamplerConfig = (
        NavToDoorOpeningTaskSamplerConfig(
            task_sampler_class=SinglePanelDoorOpeningTaskSampler,
            base_pose_sampling_radius_range=(0.8, 1.2),
            robot_safety_radius=0.35,
            max_robot_placement_attempts=50,
        )
    )

    def _init_policy_config(self) -> DoorOpeningPolicyConfig:
        policy_config = super()._init_policy_config()
        policy_config.max_steps_per_waypoint = 30
        return policy_config


@register_config("RBY1DoorOpeningSinglePanelSameDoorTestConfig")
class RBY1DoorOpeningSinglePanelSameDoorTestConfig(
    RBY1DoorOpeningSinglePanelTestBaseConfig
):
    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_door_opening_single_panel_same_door_test_v2"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _configure_stability_experiment(self, [3])


@register_config("RBY1NavDoorOpeningSinglePanelSameDoorTestConfig")
class RBY1NavDoorOpeningSinglePanelSameDoorTestConfig(
    RBY1NavDoorOpeningSimpleVisibleDataGenConfig
):
    filter_for_successful_trajectories: bool = False
    trajectory_checkpoint_interval: int = 1
    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_single_panel_same_door_test_v2"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _configure_stability_experiment(self, [3])


@register_config("RBY1DoorOpeningSinglePanelTenDoorTestConfig")
class RBY1DoorOpeningSinglePanelTenDoorTestConfig(
    RBY1DoorOpeningSinglePanelTestBaseConfig
):
    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_door_opening_single_panel_ten_door_test_v2"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _configure_stability_experiment(self, SINGLE_PANEL_TEN_DOOR_HOUSES)


@register_config("RBY1NavDoorOpeningSinglePanelTenDoorTestConfig")
class RBY1NavDoorOpeningSinglePanelTenDoorTestConfig(
    RBY1NavDoorOpeningSimpleVisibleDataGenConfig
):
    filter_for_successful_trajectories: bool = False
    trajectory_checkpoint_interval: int = 1
    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_single_panel_ten_door_test_v2"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _configure_stability_experiment(self, SINGLE_PANEL_TEN_DOOR_HOUSES)


@register_config("RBY1DoorOpeningAllSinglePanelDoorsTestConfig")
class RBY1DoorOpeningAllSinglePanelDoorsTestConfig(
    RBY1DoorOpeningSinglePanelTestBaseConfig
):
    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_door_opening_all_single_panel_doors_test_v2"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        house_inds, shard = _all_or_shard_stability_houses()
        _configure_stability_experiment(self, house_inds, shard)


@register_config("RBY1NavDoorOpeningAllSinglePanelDoorsTestConfig")
class RBY1NavDoorOpeningAllSinglePanelDoorsTestConfig(
    RBY1NavDoorOpeningSimpleVisibleDataGenConfig
):
    filter_for_successful_trajectories: bool = False
    trajectory_checkpoint_interval: int = 1
    output_dir: Path = (
        ASSETS_DIR
        / "experiment_output"
        / "datagen"
        / "rby1_nav_door_opening_all_single_panel_doors_test_v2"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        house_inds, shard = _all_or_shard_stability_houses()
        _configure_stability_experiment(self, house_inds, shard)
