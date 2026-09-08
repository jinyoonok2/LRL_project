from pathlib import Path

from molmo_spaces.configs.policy_configs import CuroboPickAndPlacePlannerPolicyConfig
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import RBY1PickAndPlaceDataGenConfig
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR

from lrl_molmospaces_extensions.configs.policy_configs import NavThenPickAndPlacePolicyConfig
from lrl_molmospaces_extensions.tasks.nav_to_pick_and_place_task import NavToPickAndPlaceTask
from lrl_molmospaces_extensions.tasks.nav_to_pick_and_place_task_sampler import NavToPickAndPlaceTaskSampler

@register_config("RBY1NavPickAndPlaceDataGenConfig")
class RBY1NavPickAndPlaceDataGenConfig(RBY1PickAndPlaceDataGenConfig):
    """Prototype long-horizon RBY1 navigation followed by pick-and-place."""

    task_type: str = "nav_to_pick_and_place"
    task_horizon: int | None = 1000
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "rby1_nav_pnp_v1"
    policy_config: NavThenPickAndPlacePolicyConfig | None = None

    def _init_policy_config(self) -> NavThenPickAndPlacePolicyConfig:
        try:
            manip_policy_config = RBY1PickAndPlaceDataGenConfig._init_policy_config(self)
        except RuntimeError as e:
            error_msg = str(e)
            if "NVIDIA" not in error_msg and "CUDA" not in error_msg and "GPU" not in error_msg:
                raise

            # Login/manager nodes may not have CUDA. Keep enough config to compile
            # scenes and test navigation; full manipulation still needs a GPU worker.
            print(
                f"Warning: Using non-GPU nav-to-PnP debug policy config because cuRobo "
                f"initialization failed: {error_msg}"
            )
            from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (
                PickAndPlacePlannerPolicy,
            )

            manip_policy_config = CuroboPickAndPlacePlannerPolicyConfig(
                policy_cls=PickAndPlacePlannerPolicy,
                enable_collision_avoidance=True,
            )

        return NavThenPickAndPlacePolicyConfig(
            manip_policy_config=manip_policy_config
        )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_config.task_cls = NavToPickAndPlaceTask
        self.task_sampler_config.task_sampler_class = NavToPickAndPlaceTaskSampler
        self.task_sampler_config.robot_safety_radius = 0.35
        self.task_sampler_config.base_pose_sampling_radius_range = (4.0, 20.0)
        self.task_sampler_config.max_robot_placement_attempts = 25
        self.task_sampler_config.max_robot_to_obj_dist = 20.0
        self.task_sampler_config.max_robot_to_place_receptacle_dist = 20.0
        self.policy_config.nav_policy_config.path_min_dist_to_target_center = 0.55

    @property
    def tag(self) -> str:
        return "rby1_nav_pick_and_place_datagen"


@register_config("RBY1NavPickAndPlaceDebugDataGenConfig")
class RBY1NavPickAndPlaceDebugDataGenConfig(RBY1NavPickAndPlaceDataGenConfig):
    """One-episode debug config for simulation and visualization checks."""

    num_workers: int = 1
    task_horizon: int | None = 1000
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "rby1_nav_pnp_debug"
    use_wandb: bool = False
    log_level: str = "debug"
    filter_for_successful_trajectories: bool = False

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.samples_per_house = 1
        self.task_sampler_config.max_tasks = 1
        self.task_sampler_config.house_inds = [0]
        self.task_sampler_config.randomize_textures = False
        self.task_sampler_config.check_robot_placement_visibility = False

    @property
    def tag(self) -> str:
        return "rby1_nav_pick_and_place_debug_datagen"


@register_config("RBY1NavPickAndPlaceFastDebugDataGenConfig")
class RBY1NavPickAndPlaceFastDebugDataGenConfig(RBY1NavPickAndPlaceDebugDataGenConfig):
    """Short local smoke test for navigation completion and policy handoff."""

    task_horizon: int | None = 300
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "rby1_nav_pnp_fast_debug"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.base_pose_sampling_radius_range = (1.2, 2.0)
        self.task_sampler_config.max_robot_to_obj_dist = 2.0
        self.task_sampler_config.max_robot_to_place_receptacle_dist = 2.0
        self.task_sampler_config.max_robot_placement_attempts = 40
        self.policy_config.nav_policy_config.path_min_dist_to_target_center = 0.55
        self.camera_config.img_resolution = (320, 180)

    @property
    def tag(self) -> str:
        return "rby1_nav_pick_and_place_fast_debug_datagen"


