"""Run a fixed RBY1 benchmark episode with a MolmoSpaces planner policy.

This is an experimental diagnostic script. It uses the same JSON benchmark episode
format as MolmoBot eval, but swaps the learned MolmoBot policy for the matching
RBY1 MolmoSpaces planner from data generation configs.

Use this to test whether a benchmark episode is physically/planner-solvable:

    conda run -n mlspaces python run_rby1_curobo_benchmark_eval.py --benchmark-kind pick --idx 9

If CuRobo succeeds and MolmoBot fails, the failure is likely learned-policy
quality/generalization. If CuRobo also fails, the task may be hard because of
object geometry, grasp assets, placement, or planner/collision constraints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from molmo_spaces.configs.policy_configs import ObjectManipulationPlannerPolicyConfig
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    RBY1OpenDataGenConfig,
    RBY1PickAndPlaceDataGenConfig,
    RBY1PickDataGenConfig,
)
from molmo_spaces.data_generation.config.door_opening_configs import DoorOpeningDataGenConfig
from molmo_spaces.evaluation.eval_main import run_evaluation


ROOT = Path(__file__).resolve().parent
BENCHMARKS = {
    "pick": ROOT / "rby1_pick_benchmark",
    "pnp": Path(
        "/home/jinyoon/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260415/"
        "procthor-objaverse/rby1_benchmarks/pnp_benchmark"
    ),
    "opening": Path(
        "/home/jinyoon/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260415/"
        "ithor/rby1_benchmarks/opening_benchmark"
    ),
    "door_opening": Path(
        "/home/jinyoon/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260415/"
        "procthor-10k/rby1_benchmarks/door_opening_benchmark"
    ),
}
DEFAULT_OUTPUT_DIR = ROOT / "MolmoBot/MolmoBot-SPOC/eval_output/RBY1PlannerJsonEval"


class _PlannerJsonEvalMixin:
    """Common eval settings for planner-on-JSON diagnostic runs."""

    use_wandb: bool = False
    use_passive_viewer: bool = False
    filter_for_successful_trajectories: bool = False

    def _finalize_planner_eval_config(self) -> None:
        if self.policy_config is None:
            self.policy_config = self._init_policy_config()

        self.robot_config.action_noise_config.enabled = False

        if isinstance(self.policy_config, ObjectManipulationPlannerPolicyConfig):
            # Use local CuRobo planning instead of a remote planner server.
            self.policy_config.server_urls = []
            # JsonEvalTaskSampler does not currently inject the temporary
            # grasp_collision_* bodies expected by this prefilter.
            self.policy_config.filter_colliding_grasps = False


class RBY1PlannerPickJsonEvalConfig(_PlannerJsonEvalMixin, RBY1PickDataGenConfig):
    """RBY1 pick config with CuRobo planner, adapted for JSON benchmark eval."""

    @property
    def tag(self) -> str:
        return "rby1_planner_pick_json_eval"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self._finalize_planner_eval_config()


class RBY1PlannerPickAndPlaceJsonEvalConfig(
    _PlannerJsonEvalMixin, RBY1PickAndPlaceDataGenConfig
):
    """RBY1 pick-and-place config with CuRobo planner for JSON benchmark eval."""

    @property
    def tag(self) -> str:
        return "rby1_planner_pnp_json_eval"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self._finalize_planner_eval_config()


class RBY1PlannerOpeningJsonEvalConfig(_PlannerJsonEvalMixin, RBY1OpenDataGenConfig):
    """RBY1 opening config with CuRobo open/close planner for JSON benchmark eval."""

    @property
    def tag(self) -> str:
        return "rby1_planner_opening_json_eval"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self._finalize_planner_eval_config()


class RBY1PlannerDoorOpeningJsonEvalConfig(_PlannerJsonEvalMixin, DoorOpeningDataGenConfig):
    """RBY1 door-opening config with door planner for JSON benchmark eval."""

    @property
    def tag(self) -> str:
        return "rby1_planner_door_opening_json_eval"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self._finalize_planner_eval_config()


CONFIGS = {
    "pick": RBY1PlannerPickJsonEvalConfig,
    "pnp": RBY1PlannerPickAndPlaceJsonEvalConfig,
    "opening": RBY1PlannerOpeningJsonEvalConfig,
    "door_opening": RBY1PlannerDoorOpeningJsonEvalConfig,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RBY1 JSON benchmark with MolmoSpaces planner.")
    parser.add_argument(
        "--benchmark-kind",
        choices=sorted(BENCHMARKS),
        default="pick",
        help="Which local RBY1 benchmark/planner pairing to run.",
    )
    parser.add_argument("--idx", type=int, required=True, help="Benchmark episode index to run.")
    parser.add_argument("--benchmark-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task-horizon-sec", type=float, default=None)
    parser.add_argument("--task-horizon-steps", type=int, default=None)
    parser.add_argument("--viewer", action="store_true", help="Launch MuJoCo passive viewer.")
    args = parser.parse_args()

    if args.task_horizon_sec is not None and args.task_horizon_steps is not None:
        raise ValueError("Use only one of --task-horizon-sec or --task-horizon-steps.")

    config_cls = CONFIGS[args.benchmark_kind]
    benchmark_dir = args.benchmark_dir or BENCHMARKS[args.benchmark_kind]
    config_cls.use_passive_viewer = args.viewer

    results = run_evaluation(
        eval_config_cls=config_cls,
        benchmark_dir=benchmark_dir,
        task_horizon_sec=args.task_horizon_sec,
        task_horizon_steps=args.task_horizon_steps,
        output_dir=args.output_dir,
        num_workers=1,
        use_wandb=False,
        episode_idx=args.idx,
    )

    print(f"Benchmark kind: {args.benchmark_kind}")
    print(f"Benchmark dir: {benchmark_dir}")
    print(f"Output dir: {results.output_dir}")
    print(f"Success rate: {results.success_count}/{results.total_count}")
    for result in results.episode_results:
        print(f"{result.house_id}/ep{result.episode_idx}: {'pass' if result.success else 'fail'}")


if __name__ == "__main__":
    main()
