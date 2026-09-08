"""Policy configuration for LRL long-horizon MolmoSpaces extensions."""

import numpy as np
from pydantic import Field

from molmo_spaces.configs.policy_configs import (
    AStarNavToObjPolicyConfig,
    BasePolicyConfig,
    CuroboPickAndPlacePlannerPolicyConfig,
    DoorOpeningPolicyConfig,
)


class LRLAStarNavToObjPolicyConfig(AStarNavToObjPolicyConfig):
    """A* configuration with LRL waypoint and completion semantics."""

    policy_cls: type = None
    intermediate_waypoint_xy_threshold_m: float = 0.2

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        from lrl_molmospaces_extensions.policy.navigation.astar_planner_policy import (
            AStarSmoothPlannerPolicy,
        )

        self.policy_cls = AStarSmoothPlannerPolicy


class NavThenPickAndPlacePolicyConfig(BasePolicyConfig):
    """Composite policy config for A* navigation followed by pick-and-place."""

    policy_cls: type = None
    policy_type: str = "planner"
    handoff_max_distance_to_pickup_m: float = 0.75
    final_approach_enabled: bool = True
    final_approach_distance_to_pickup_m: float = 0.55
    final_approach_step_size_m: float = 0.05
    final_approach_after_nav_failure_max_distance_m: float = 1.25
    final_approach_max_steps: int = 60
    final_approach_fallback_handoff_max_distance_m: float = 0.85
    final_approach_stall_window_steps: int = 10
    final_approach_min_progress_m: float = 0.02
    nav_policy_config: LRLAStarNavToObjPolicyConfig = Field(
        default_factory=LRLAStarNavToObjPolicyConfig
    )
    manip_policy_config: CuroboPickAndPlacePlannerPolicyConfig = Field(
        default_factory=CuroboPickAndPlacePlannerPolicyConfig
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            from lrl_molmospaces_extensions.policy.nav_then_pick_and_place_policy import (
                NavThenPickAndPlacePolicy,
            )

            self.policy_cls = NavThenPickAndPlacePolicy


class NavThenDoorOpeningPolicyConfig(BasePolicyConfig):
    """Composite policy config for A* navigation followed by door opening."""

    policy_cls: type = None
    policy_type: str = "planner"
    handoff_max_distance_to_handle_m: float = 1.0
    handoff_after_nav_failure_max_distance_to_handle_m: float = 1.25
    final_align_enabled: bool = True
    final_align_target_distance_to_handle_m: float = 0.85
    final_align_max_steps: int = 40
    final_align_step_size_m: float = 0.15
    final_align_yaw_threshold_rad: float = float(np.deg2rad(15))
    nav_policy_config: LRLAStarNavToObjPolicyConfig = Field(
        default_factory=LRLAStarNavToObjPolicyConfig
    )
    opening_policy_config: DoorOpeningPolicyConfig | None = None

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            from lrl_molmospaces_extensions.policy.nav_then_door_opening_policy import (
                NavThenDoorOpeningPolicy,
            )

            self.policy_cls = NavThenDoorOpeningPolicy
