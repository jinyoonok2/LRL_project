import json
import hashlib
import logging
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.ndimage import label as label_connected_components

from molmo_spaces.configs.task_sampler_configs import DoorOpeningTaskSamplerConfig
from molmo_spaces.env.data_views import Door
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.tasks.opening_task_samplers import DoorOpeningTaskSampler
from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask, RobotPlacementError

from lrl_molmospaces_extensions.tasks.nav_to_door_opening_task import (
    DoorHandleNavTarget,
)
from lrl_molmospaces_extensions.tasks.util_samplers.door_panels import (
    door_panel_info_by_body_name,
    filter_single_panel_door_names,
)
from lrl_molmospaces_extensions.tasks.util_samplers.navgoal_sampler import NavGoalSampler

log = logging.getLogger(__name__)


class NavToDoorOpeningTaskSamplerConfig(DoorOpeningTaskSamplerConfig):
    """Sampler controls used only by the nav-to-door-opening extension."""

    target_grounding_mode: Literal["none", "visible_unique", "point_prompt"] = "none"
    target_visibility_camera: str = "head_camera"
    target_door_min_visibility_fraction: float = 0.0001
    target_handle_min_visibility_fraction: float = 0.00001
    competing_door_min_visibility_fraction: float = 0.0001
    single_panel_doors_only: bool = False
    enumerate_all_doors: bool = False
    successes_per_door: int = 1
    max_rollout_attempts_per_door: int = 12
    fixed_rollout_attempts_per_door: int | None = None
    max_house_episode_attempts: int = 1000
    required_door_action: Literal["both", "push", "pull"] = "both"
    max_action_side_placement_attempts: int = 25
    max_robot_placement_attempts: int = 50
    max_placement_failure_batches_per_door: int = 20


class NavToDoorOpeningTaskSampler(DoorOpeningTaskSampler):
    """Sample door-opening tasks with a far robot start for navigation."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self._connectivity_labels: np.ndarray | None = None
        self._door_schedule: list[str] = []
        self._door_cursor = 0
        self._door_attempts: Counter[str] = Counter()
        self._door_successes: Counter[str] = Counter()
        self._door_push_attempts: Counter[str] = Counter()
        self._door_push_successes: Counter[str] = Counter()
        self._door_pull_attempts: Counter[str] = Counter()
        self._door_pull_successes: Counter[str] = Counter()
        self._door_placement_failures: Counter[str] = Counter()
        self._door_action_side_rejections: Counter[str] = Counter()
        self._door_status: dict[str, str] = {}
        self._active_door_name: str | None = None
        self._door_schedule_house_index: int | None = None

    def _sample_task(self, env: CPUMujocoEnv):
        """Sample until the robot starts on the requested push or pull side."""

        required_action = self.config.task_sampler_config.required_door_action
        if required_action == "both":
            return super()._sample_task(env)

        max_attempts = (
            self.config.task_sampler_config.max_action_side_placement_attempts
        )
        for placement_attempt in range(1, max_attempts + 1):
            task = super()._sample_task(env)
            actual_action = "push" if task._is_pushing_door else "pull"
            if actual_action == required_action:
                log.info(
                    "[DOOR-ACTION FILTER] Accepted %s start on attempt %d/%d",
                    required_action,
                    placement_attempt,
                    max_attempts,
                )
                return task

            log.info(
                "[DOOR-ACTION FILTER] Rejected %s start while requesting %s "
                "on attempt %d/%d",
                actual_action,
                required_action,
                placement_attempt,
                max_attempts,
            )
            if self._active_door_name is not None:
                self._door_action_side_rejections[self._active_door_name] += 1
                self._record_sampling_event(
                    self._active_door_name,
                    "action_side_rejection",
                    f"sampled_{actual_action}_while_requesting_{required_action}",
                )
            self.config.task_config.robot_base_pose = None

        if self.config.task_sampler_config.enumerate_all_doors:
            self.mark_active_door_unplaceable(
                "required_action_side_unavailable"
            )
        self.config.task_config.door_body_name = None
        self.config.task_config.robot_base_pose = None
        raise RobotPlacementError(
            f"Could not place robot on the {required_action} side of the active door"
        )

    def init_scene(self, env) -> None:
        super().init_scene(env)
        self._connectivity_labels = None
        self._reset_door_schedule()

    def reset(self) -> None:
        super().reset()
        self._connectivity_labels = None
        self._reset_door_schedule()

    def _reset_door_schedule(self) -> None:
        self._door_schedule = []
        self._door_cursor = 0
        self._door_attempts.clear()
        self._door_successes.clear()
        self._door_push_attempts.clear()
        self._door_push_successes.clear()
        self._door_pull_attempts.clear()
        self._door_pull_successes.clear()
        self._door_placement_failures.clear()
        self._door_action_side_rejections.clear()
        self._door_status.clear()
        self._active_door_name = None
        self._door_schedule_house_index = None

    def _initialize_door_schedule(self, door_body_names: list[str]) -> None:
        if not self._door_schedule:
            self._door_schedule = sorted(set(door_body_names))
            self._door_schedule_house_index = self.current_house_index
            self._door_status = {
                door_name: "pending" for door_name in self._door_schedule
            }
            log.info(
                "[PER-DOOR] Initialized deterministic schedule with %d doors",
                len(self._door_schedule),
            )
            self._sync_outcome_counters_from_records()
            self._advance_door_cursor()

    def _advance_door_cursor(self) -> None:
        sampler_config = self.config.task_sampler_config
        while self._door_cursor < len(self._door_schedule):
            door_name = self._door_schedule[self._door_cursor]
            if self._door_status.get(door_name) == "sampling_shortfall":
                self._door_cursor += 1
                continue
            successes = self._door_successes[door_name]
            attempts = self._door_attempts[door_name]
            fixed_attempts = sampler_config.fixed_rollout_attempts_per_door
            if fixed_attempts is not None and attempts >= fixed_attempts:
                self._door_status[door_name] = "complete"
                self._door_cursor += 1
            elif fixed_attempts is not None:
                break
            elif successes >= sampler_config.successes_per_door:
                self._door_status[door_name] = "complete"
                self._door_cursor += 1
            elif attempts >= sampler_config.max_rollout_attempts_per_door:
                self._door_status[door_name] = "attempts_exhausted"
                self._door_cursor += 1
            else:
                break

    def is_house_sampling_complete(self, house_index: int | None = None) -> bool:
        if not self.config.task_sampler_config.enumerate_all_doors:
            return False
        if (
            house_index is not None
            and self._door_schedule_house_index != house_index
        ):
            return False
        self._advance_door_cursor()
        return bool(self._door_schedule) and self._door_cursor >= len(self._door_schedule)

    def _write_per_door_manifest(self) -> None:
        if not self.config.task_sampler_config.enumerate_all_doors:
            return
        self._sync_outcome_counters_from_records()
        manifest_dir = Path(self.config.output_dir) / "per_door_manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        house_index = self._door_schedule_house_index
        if house_index is None:
            return
        manifest_path = manifest_dir / f"house_{house_index}.json"

        def rate(successes: int, attempts: int) -> float | None:
            return successes / attempts if attempts else None

        manifest = {
            "house_index": house_index,
            "required_door_action": (
                self.config.task_sampler_config.required_door_action
            ),
            "single_panel_doors_only": (
                self.config.task_sampler_config.single_panel_doors_only
            ),
            "successes_per_door_target": (
                self.config.task_sampler_config.successes_per_door
            ),
            "max_rollout_attempts_per_door": (
                self.config.task_sampler_config.max_rollout_attempts_per_door
            ),
            "fixed_rollout_attempts_per_door": (
                self.config.task_sampler_config.fixed_rollout_attempts_per_door
            ),
            "doors": {
                door_name: {
                    "attempts": self._door_attempts[door_name],
                    "successes": self._door_successes[door_name],
                    "success_rate": rate(
                        self._door_successes[door_name],
                        self._door_attempts[door_name],
                    ),
                    "push": {
                        "attempts": self._door_push_attempts[door_name],
                        "successes": self._door_push_successes[door_name],
                        "success_rate": rate(
                            self._door_push_successes[door_name],
                            self._door_push_attempts[door_name],
                        ),
                    },
                    "pull": {
                        "attempts": self._door_pull_attempts[door_name],
                        "successes": self._door_pull_successes[door_name],
                        "success_rate": rate(
                            self._door_pull_successes[door_name],
                            self._door_pull_attempts[door_name],
                        ),
                    },
                    "placement_failures": self._door_placement_failures[door_name],
                    "action_side_rejections": (
                        self._door_action_side_rejections[door_name]
                    ),
                    "status": self._door_status.get(door_name, "pending"),
                }
                for door_name in self._door_schedule
            },
        }
        temp_path = manifest_path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(manifest, indent=2))
        temp_path.replace(manifest_path)

    def _sync_outcome_counters_from_records(self) -> None:
        """Rebuild rollout counters from durable trial records."""

        if self._door_schedule_house_index is None:
            return
        attempts: Counter[str] = Counter()
        successes: Counter[str] = Counter()
        push_attempts: Counter[str] = Counter()
        push_successes: Counter[str] = Counter()
        pull_attempts: Counter[str] = Counter()
        pull_successes: Counter[str] = Counter()
        trial_root = (
            Path(self.config.output_dir)
            / "trial_outcomes"
            / f"house_{self._door_schedule_house_index}"
        )
        for record_path in trial_root.glob("*/trial_*.json"):
            try:
                record = json.loads(record_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            door_name = record.get("door_body_name")
            if door_name not in self._door_schedule:
                continue
            success = bool(record.get("success"))
            action = record.get("action")
            attempts[door_name] += 1
            successes[door_name] += int(success)
            if action == "push":
                push_attempts[door_name] += 1
                push_successes[door_name] += int(success)
            elif action == "pull":
                pull_attempts[door_name] += 1
                pull_successes[door_name] += int(success)

        self._door_attempts = attempts
        self._door_successes = successes
        self._door_push_attempts = push_attempts
        self._door_push_successes = push_successes
        self._door_pull_attempts = pull_attempts
        self._door_pull_successes = pull_successes
        self._door_placement_failures = Counter()
        self._door_action_side_rejections = Counter()
        sampling_root = (
            Path(self.config.output_dir)
            / "sampling_events"
            / f"house_{self._door_schedule_house_index}"
        )
        for event_path in sampling_root.glob("*/*.json"):
            try:
                event = json.loads(event_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            door_name = event.get("door_body_name")
            if door_name not in self._door_schedule:
                continue
            if event.get("event_type") == "placement_failure":
                self._door_placement_failures[door_name] += 1
            elif event.get("event_type") == "action_side_rejection":
                self._door_action_side_rejections[door_name] += 1

    def _write_house_exclusion(self, reason: str) -> None:
        manifest_dir = Path(self.config.output_dir) / "house_exclusions"
        manifest_path = manifest_dir / f"house_{self.current_house_index}.json"
        self._atomic_write_json(
            manifest_path,
            {
                "schema_version": 1,
                "house_index": self.current_house_index,
                "status": "excluded",
                "reason": reason,
            },
        )

    def _trial_record_path(self, door_name: str, attempt: int) -> Path:
        house_index = self._door_schedule_house_index
        door_hash = hashlib.sha1(door_name.encode("utf-8")).hexdigest()[:12]
        return (
            Path(self.config.output_dir)
            / "trial_outcomes"
            / f"house_{house_index}"
            / door_hash
            / f"trial_{attempt:03d}.json"
        )

    def _record_sampling_event(
        self, door_name: str, event_type: str, reason: str
    ) -> None:
        door_hash = hashlib.sha1(door_name.encode("utf-8")).hexdigest()[:12]
        if event_type == "placement_failure":
            event_index = self._door_placement_failures[door_name]
        else:
            event_index = self._door_action_side_rejections[door_name]
        path = (
            Path(self.config.output_dir)
            / "sampling_events"
            / f"house_{self._door_schedule_house_index}"
            / door_hash
            / f"{event_type}_{event_index:03d}.json"
        )
        self._atomic_write_json(
            path,
            {
                "schema_version": 1,
                "house_index": self._door_schedule_house_index,
                "door_body_name": door_name,
                "event_type": event_type,
                "reason": reason,
            },
        )

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(
                payload,
                indent=2,
                default=lambda value: (
                    value.tolist()
                    if hasattr(value, "tolist")
                    else str(value)
                ),
            )
        )
        temp_path.replace(path)

    def report_rollout_result(
        self,
        task,
        success: bool,
        *,
        seed: int | None = None,
        termination_reason: str = "completed",
        exception: BaseException | None = None,
    ) -> Path | None:
        """Record an outcome and advance only after success or attempt exhaustion."""

        if not self.config.task_sampler_config.enumerate_all_doors:
            return None
        door_name = task.door_object.name
        self._door_attempts[door_name] += 1
        if success:
            self._door_successes[door_name] += 1
        action = "push" if task._is_pushing_door else "pull"
        if action == "push":
            self._door_push_attempts[door_name] += 1
            if success:
                self._door_push_successes[door_name] += 1
        else:
            self._door_pull_attempts[door_name] += 1
            if success:
                self._door_pull_successes[door_name] += 1
        target_attempts = (
            self.config.task_sampler_config.fixed_rollout_attempts_per_door
            or self.config.task_sampler_config.max_rollout_attempts_per_door
        )
        log.info(
            "[PER-DOOR] door=%s attempt=%d/%d successes=%d action=%s",
            door_name,
            self._door_attempts[door_name],
            target_attempts,
            self._door_successes[door_name],
            action,
        )
        attempt = self._door_attempts[door_name]
        task_config = task.config.task_config
        record_path = self._trial_record_path(door_name, attempt)
        self._atomic_write_json(
            record_path,
            {
                "schema_version": 1,
                "house_index": self._door_schedule_house_index,
                "door_body_name": door_name,
                "doorway_name": getattr(task_config, "target_doorway_name", None),
                "door_asset_id": getattr(task_config, "target_door_asset_id", None),
                "door_panel_count": getattr(
                    task_config, "target_door_panel_count", None
                ),
                "action": action,
                "attempt": attempt,
                "seed": seed,
                "robot_base_pose": getattr(task_config, "robot_base_pose", None),
                "success": bool(success),
                "termination_reason": termination_reason,
                "exception_type": (
                    type(exception).__name__ if exception is not None else None
                ),
                "exception_message": (
                    str(exception) if exception is not None else None
                ),
                "h5_saved": False,
                "h5_path": None,
            },
        )
        self._advance_door_cursor()
        self._write_per_door_manifest()
        return record_path

    def update_trial_h5_status(
        self, record_path: Path | None, *, h5_saved: bool, h5_path: Path | None
    ) -> None:
        if record_path is None or not record_path.exists():
            return
        payload = json.loads(record_path.read_text())
        payload["h5_saved"] = bool(h5_saved)
        payload["h5_path"] = str(h5_path) if h5_path is not None else None
        self._atomic_write_json(record_path, payload)

    def mark_active_door_unplaceable(self, reason: str = "placement_failed") -> None:
        if self._active_door_name is None:
            return
        door_name = self._active_door_name
        self._door_placement_failures[door_name] += 1
        self._record_sampling_event(door_name, "placement_failure", reason)
        failure_batches = self._door_placement_failures[door_name]
        max_failure_batches = (
            self.config.task_sampler_config.max_placement_failure_batches_per_door
        )
        if failure_batches >= max_failure_batches:
            self._door_status[door_name] = "sampling_shortfall"
            log.info(
                "[PER-DOOR] door='%s' sampling_shortfall after %d executed "
                "attempts and %d failed placement batches (reason=%s)",
                door_name,
                self._door_attempts[door_name],
                failure_batches,
                reason,
            )
        else:
            self._door_status[door_name] = "pending"
            log.info(
                "[PER-DOOR] Retrying door='%s' after failed placement batch "
                "%d/%d (executed attempts=%d, reason=%s)",
                door_name,
                failure_batches,
                max_failure_batches,
                self._door_attempts[door_name],
                reason,
            )
        self._advance_door_cursor()
        self._write_per_door_manifest()

    @property
    def connectivity_labels(self) -> np.ndarray:
        """Label the same four-connected, downscaled free-space grid used by A*."""

        if self._connectivity_labels is None:
            if self._cached_thormap is None:
                raise RuntimeError("Occupancy map is unavailable for connectivity validation")

            grid = np.asarray(self._cached_thormap.occupancy, dtype=bool)
            downscale = (
                self.config.policy_config.nav_policy_config.planner_config.downscale_factor
            )
            padded = np.zeros(
                (
                    grid.shape[0] + (downscale - grid.shape[0] % downscale),
                    grid.shape[1] + (downscale - grid.shape[1] % downscale),
                ),
                dtype=bool,
            )
            padded[: grid.shape[0], : grid.shape[1]] = grid
            downscaled_grid = (
                padded.reshape(
                    padded.shape[0] // downscale,
                    downscale,
                    padded.shape[1] // downscale,
                    downscale,
                )
                .min(axis=1)
                .min(axis=-1)
            )
            four_connected = np.array(
                [
                    [0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0],
                ],
                dtype=np.uint8,
            )
            self._connectivity_labels, num_components = label_connected_components(
                downscaled_grid, structure=four_connected
            )
            log.info(
                "[NAV-TO-DOOR CONNECTIVITY] Found %d navigable components",
                num_components,
            )

        return self._connectivity_labels

    def _component_at_position(self, position: np.ndarray) -> int | None:
        """Resolve a world position to a nearby A*-navigable component."""

        downscale = self.config.policy_config.nav_policy_config.planner_config.downscale_factor
        max_search = (
            self.config.policy_config.nav_policy_config.planner_config.max_start_goal_distance
        )
        discrete = np.floor(
            self._cached_thormap.pos_m_to_px(np.asarray(position)) / downscale
        ).astype(np.int32)
        labels = self.connectivity_labels

        def component_at(row: int, col: int) -> int | None:
            if row < 0 or col < 0 or row >= labels.shape[0] or col >= labels.shape[1]:
                return None
            component = int(labels[row, col])
            return component if component > 0 else None

        component = component_at(int(discrete[0]), int(discrete[1]))
        if component is not None:
            return component

        for radius in range(1, max_search + 1):
            for row_offset in range(-radius, radius + 1):
                for col_offset in range(-radius, radius + 1):
                    if abs(row_offset) != radius and abs(col_offset) != radius:
                        continue
                    component = component_at(
                        int(discrete[0] + row_offset),
                        int(discrete[1] + col_offset),
                    )
                    if component is not None:
                        return component

        return None

    def _navigation_goal_position(self, door_object: Door) -> np.ndarray | None:
        """Reproduce the runtime navigation goal sampled around the door handle."""

        handle_position = door_object.get_handle_pose()[:3]
        nav_target = DoorHandleNavTarget(
            name=f"{door_object.name}:handle",
            position=handle_position,
        )
        goal_sampler = NavGoalSampler(self._cached_thormap)
        goal_sampler.set_target(nav_target)
        sampled_goal = goal_sampler.sample()
        if sampled_goal is None:
            return None
        return np.asarray(sampled_goal[0])

    def _positions_are_connected(
        self,
        robot_position: np.ndarray,
        goal_position: np.ndarray,
    ) -> bool:
        start_component = self._component_at_position(robot_position)
        goal_component = self._component_at_position(goal_position)
        connected = start_component is not None and start_component == goal_component
        log.info(
            "[NAV-TO-DOOR CONNECTIVITY] start_component=%s goal_component=%s connected=%s",
            start_component,
            goal_component,
            connected,
        )
        return connected

    def _evaluate_target_grounding(
        self,
        env: CPUMujocoEnv,
        door_object: Door,
        door_body_names: list[str],
    ) -> tuple[bool, float, float, list[str]]:
        """Check whether the selected door is visible and unambiguous."""

        sampler_config = self.config.task_sampler_config
        camera_name = sampler_config.target_visibility_camera
        if camera_name not in env.camera_manager.registry:
            self.setup_cameras(env, deterministic_only=True)
        env.camera_manager.registry.update_all_cameras(env)

        try:
            segmentation = env.render_segmentation_frame(camera_name)
        except Exception as exc:
            log.warning(
                "[NAV-TO-DOOR GROUNDING] Failed to render '%s': %s",
                camera_name,
                exc,
            )
            return False, 0.0, 0.0, []

        door_visibility: dict[str, float] = {}
        for door_name in door_body_names:
            try:
                door_visibility[door_name] = env.segmentation_fraction(
                    segmentation, door_name
                )
            except (KeyError, ValueError):
                door_visibility[door_name] = 0.0

        target_visibility = door_visibility.get(door_object.name, 0.0)
        handle_visibility = 0.0
        for handle_index in range(door_object.num_handles):
            try:
                handle_visibility = max(
                    handle_visibility,
                    env.segmentation_fraction(
                        segmentation, door_object.handle_name(handle_index)
                    ),
                )
            except (KeyError, ValueError):
                continue

        competing_doors = sorted(
            door_name
            for door_name, visibility in door_visibility.items()
            if door_name != door_object.name
            and visibility >= sampler_config.competing_door_min_visibility_fraction
        )
        target_visible = (
            target_visibility >= sampler_config.target_door_min_visibility_fraction
            and (
                sampler_config.target_grounding_mode != "point_prompt"
                or handle_visibility
                >= sampler_config.target_handle_min_visibility_fraction
            )
        )
        unambiguous = (
            sampler_config.target_grounding_mode != "visible_unique"
            or len(competing_doors) == 0
        )
        accepted = target_visible and unambiguous
        log.info(
            "[NAV-TO-DOOR GROUNDING] camera=%s target_visibility=%.6f "
            "handle_visibility=%.6f competing_doors=%d accepted=%s",
            camera_name,
            target_visibility,
            handle_visibility,
            len(competing_doors),
            accepted,
        )
        return accepted, target_visibility, handle_visibility, competing_doors

    def _record_target_grounding(
        self,
        target_visibility: float,
        handle_visibility: float,
        competing_doors: list[str],
    ) -> None:
        task_config = self.config.task_config
        task_config.target_grounding_mode = (
            self.config.task_sampler_config.target_grounding_mode
        )
        task_config.target_door_visibility_fraction = target_visibility
        task_config.target_handle_visibility_fraction = handle_visibility
        task_config.visible_competing_door_names = competing_doors

    def _record_target_panel_info(
        self,
        env: CPUMujocoEnv,
        door_body_name: str,
    ) -> None:
        """Store physical-door classification for downstream dataset filtering."""

        task_config = self.config.task_config
        panel_info = door_panel_info_by_body_name(env.current_scene_metadata).get(
            door_body_name
        )
        if panel_info is None:
            task_config.target_doorway_name = None
            task_config.target_door_asset_id = None
            task_config.target_door_panel_count = None
            task_config.target_is_single_panel = None
            return

        task_config.target_doorway_name = panel_info.doorway_name
        task_config.target_door_asset_id = panel_info.asset_id
        task_config.target_door_panel_count = panel_info.panel_count
        task_config.target_is_single_panel = panel_info.is_single_panel

    def _sample_door_and_place_robot(self, env: CPUMujocoEnv, door_body_names: list[str]):
        if len(door_body_names) == 0:
            raise HouseInvalidForTask("No doors found in the scene")

        all_scene_door_body_names = list(door_body_names)
        if self.config.task_sampler_config.single_panel_doors_only:
            door_body_names, rejected_door_names = filter_single_panel_door_names(
                door_body_names,
                env.current_scene_metadata,
            )
            log.info(
                "[DOOR-PANEL FILTER] Kept %d single-panel candidates; "
                "rejected %d double-panel or unclassified candidates",
                len(door_body_names),
                len(rejected_door_names),
            )
            if len(door_body_names) == 0:
                self._write_house_exclusion("no_confirmed_single_panel_doors")
                raise HouseInvalidForTask(
                    "No confirmed single-panel doors found in the scene"
                )

        enumerate_all_doors = self.config.task_sampler_config.enumerate_all_doors
        if enumerate_all_doors:
            self._initialize_door_schedule(door_body_names)
            self._advance_door_cursor()
            if self.is_house_sampling_complete():
                raise HouseInvalidForTask("All doors in the house have been processed")
            door_body_names = [self._door_schedule[self._door_cursor]]
            self._active_door_name = door_body_names[0]
            log.info(
                "[PER-DOOR] Processing door %d/%d: %s",
                self._door_cursor + 1,
                len(self._door_schedule),
                self._active_door_name,
            )
        else:
            door_body_names = list(door_body_names)
            np.random.shuffle(door_body_names)

        task_config = self.config.task_config
        task_config.target_grounding_mode = "none"
        task_config.target_door_visibility_fraction = None
        task_config.target_handle_visibility_fraction = None
        task_config.visible_competing_door_names = []
        task_config.target_doorway_name = None
        task_config.target_door_asset_id = None
        task_config.target_door_panel_count = None
        task_config.target_is_single_panel = None

        sampling_radius_range = self.config.task_sampler_config.base_pose_sampling_radius_range
        max_attempts = self.config.task_sampler_config.max_robot_placement_attempts
        for chosen_door_name in door_body_names:
            door_object = Door(chosen_door_name, env.mj_datas[0])
            door_object.set_joint_position(door_object.get_hinge_joint_index(), 0.0)
            goal_position = self._navigation_goal_position(door_object)
            if goal_position is None:
                log.info(
                    "[NAV-TO-DOOR CONNECTIVITY] No navigation goal for door '%s'; "
                    "trying next door",
                    door_object.name,
                )
                continue

            log.info(
                "[NAV-TO-DOOR] Placing robot %.2f-%.2fm from door '%s'",
                sampling_radius_range[0],
                sampling_radius_range[1],
                door_object.name,
            )
            excluded_positions = list(self.used_robot_positions[door_object.name])
            for placement_attempt in range(1, max_attempts + 1):
                robot_placed = env.place_robot_near(
                    robot_view=env.current_robot.robot_view,
                    target=door_object,
                    max_tries=1,
                    sampling_radius_range=sampling_radius_range,
                    robot_safety_radius=self.config.task_sampler_config.robot_safety_radius,
                    face_target=self.config.task_sampler_config.target_grounding_mode
                    != "none",
                    check_camera_visibility=self.config.task_sampler_config.check_robot_placement_visibility,
                    visibility_resolver=self.get_visibility_resolver(env),
                    excluded_positions=excluded_positions,
                )
                if not robot_placed:
                    continue

                robot_pos = env.current_robot.robot_view.base.pose[:3, 3]
                if not self._positions_are_connected(robot_pos, goal_position):
                    excluded_positions.append(robot_pos.copy())
                    log.info(
                        "[NAV-TO-DOOR CONNECTIVITY] Rejected disconnected robot pose "
                        "on attempt %d/%d",
                        placement_attempt,
                        max_attempts,
                    )
                    continue

                grounding_mode = self.config.task_sampler_config.target_grounding_mode
                if grounding_mode in {"visible_unique", "point_prompt"}:
                    (
                        grounding_accepted,
                        target_visibility,
                        handle_visibility,
                        competing_doors,
                    ) = self._evaluate_target_grounding(
                        env,
                        door_object,
                        all_scene_door_body_names,
                    )
                    if not grounding_accepted:
                        excluded_positions.append(robot_pos.copy())
                        log.info(
                            "[NAV-TO-DOOR GROUNDING] Rejected robot pose on attempt %d/%d",
                            placement_attempt,
                            max_attempts,
                        )
                        continue
                    self._record_target_grounding(
                        target_visibility,
                        handle_visibility,
                        competing_doors,
                    )
                handle_pos = door_object.get_handle_pose()[:3]
                log.info(
                    "[NAV-TO-DOOR] Robot start=(%.3f, %.3f, %.3f), handle=(%.3f, %.3f, %.3f), distance=%.3fm",
                    robot_pos[0],
                    robot_pos[1],
                    robot_pos[2],
                    handle_pos[0],
                    handle_pos[1],
                    handle_pos[2],
                    float(np.linalg.norm(robot_pos[:2] - handle_pos[:2])),
                )
                self.used_robot_positions[door_object.name].append(robot_pos)
                self._active_door_name = door_object.name
                self._record_target_panel_info(env, door_object.name)
                return door_object

            log.info(
                "[NAV-TO-DOOR] Failed to place a connected robot pose near '%s'. "
                "Trying next door...",
                door_object.name,
            )

        if enumerate_all_doors:
            self.mark_active_door_unplaceable("no_valid_navigation_start")
            if not self.is_house_sampling_complete():
                return self._sample_door_and_place_robot(
                    env, all_scene_door_body_names
                )
            raise HouseInvalidForTask("All doors in the house have been processed")

        raise RobotPlacementError("Was not able to place robot far from any door in the scene.")


class SinglePanelDoorOpeningTaskSampler(NavToDoorOpeningTaskSampler):
    """Per-door evaluator using the baseline door-only placement and opening path."""

    def _place_robot_for_open_only(
        self,
        env: CPUMujocoEnv,
        door_body_names: list[str],
    ):
        """Mirror the upstream placement method but honor the configured retry count."""

        max_tries = self.config.task_sampler_config.max_robot_placement_attempts
        candidates = list(door_body_names)
        np.random.shuffle(candidates)
        for chosen_door_name in candidates:
            door_object = Door(chosen_door_name, env.mj_datas[0])
            door_object.set_joint_position(
                door_object.get_hinge_joint_index(), 0.0
            )
            robot_placed = env.place_robot_near(
                robot_view=env.current_robot.robot_view,
                target=door_object,
                max_tries=max_tries,
                sampling_radius_range=(
                    self.config.task_sampler_config.base_pose_sampling_radius_range
                ),
                robot_safety_radius=(
                    self.config.task_sampler_config.robot_safety_radius
                ),
                face_target=True,
                check_camera_visibility=(
                    self.config.task_sampler_config.check_robot_placement_visibility
                ),
                visibility_resolver=self.get_visibility_resolver(env),
                excluded_positions=self.used_robot_positions[door_object.name],
            )
            if robot_placed:
                self.used_robot_positions[door_object.name].append(
                    env.current_robot.robot_view.base.pose[:3, 3].copy()
                )
                return door_object
        raise ValueError(
            "Was not able to place robot near any requested single-panel door"
        )

    def _sample_door_and_place_robot(
        self,
        env: CPUMujocoEnv,
        door_body_names: list[str],
    ):
        all_scene_door_body_names = list(door_body_names)
        single_panel_names, rejected_names = filter_single_panel_door_names(
            door_body_names,
            env.current_scene_metadata,
        )
        log.info(
            "[DOOR-PANEL FILTER] Door-only branch kept %d single-panel "
            "candidates; rejected %d double-panel or unclassified candidates",
            len(single_panel_names),
            len(rejected_names),
        )
        if not single_panel_names:
            self._write_house_exclusion("no_confirmed_single_panel_doors")
            raise HouseInvalidForTask(
                "No confirmed single-panel doors found in the scene"
            )

        enumerate_all_doors = self.config.task_sampler_config.enumerate_all_doors
        if enumerate_all_doors:
            self._initialize_door_schedule(single_panel_names)
            self._advance_door_cursor()
            if self.is_house_sampling_complete():
                raise HouseInvalidForTask(
                    "All single-panel doors in the house have been processed"
                )
            single_panel_names = [self._door_schedule[self._door_cursor]]
            self._active_door_name = single_panel_names[0]

        try:
            door_object = self._place_robot_for_open_only(env, single_panel_names)
        except ValueError:
            if not enumerate_all_doors:
                raise
            self.mark_active_door_unplaceable("no_valid_open_only_start")
            if not self.is_house_sampling_complete():
                return self._sample_door_and_place_robot(
                    env,
                    all_scene_door_body_names,
                )
            raise HouseInvalidForTask(
                "All single-panel doors in the house have been processed"
            )

        self._active_door_name = door_object.name
        self._record_target_panel_info(env, door_object.name)
        return door_object
