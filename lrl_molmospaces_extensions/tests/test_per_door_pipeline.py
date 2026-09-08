import json
from collections import Counter, defaultdict
from types import SimpleNamespace

from molmo_spaces.tasks.opening_task_samplers import DoorOpeningTaskSampler

from lrl_molmospaces_extensions.datagen import per_door_rollout_runner
from lrl_molmospaces_extensions.datagen.per_door_rollout_runner import (
    PerDoorRolloutRunner,
)
from lrl_molmospaces_extensions.datagen.validate_shard import validate_run
from lrl_molmospaces_extensions.tasks.nav_to_door_opening_task_sampler import (
    NavToDoorOpeningTaskSampler,
    SinglePanelDoorOpeningTaskSampler,
)
from lrl_molmospaces_extensions.tasks.util_samplers.door_panels import (
    door_panel_info_by_body_name,
    filter_single_panel_door_names,
)


def _build_sampler(tmp_path, house_index: int = 7):
    sampler = object.__new__(NavToDoorOpeningTaskSampler)
    sampler.config = SimpleNamespace(
        output_dir=tmp_path,
        task_sampler_config=SimpleNamespace(
            single_panel_doors_only=True,
            enumerate_all_doors=True,
            successes_per_door=1,
            max_rollout_attempts_per_door=3,
            fixed_rollout_attempts_per_door=None,
            required_door_action="both",
            max_placement_failure_batches_per_door=2,
        ),
    )
    sampler._house_inds = [house_index]
    sampler._house_iterator_index = 0
    sampler._door_schedule = []
    sampler._door_cursor = 0
    sampler._door_attempts = Counter()
    sampler._door_successes = Counter()
    sampler._door_push_attempts = Counter()
    sampler._door_push_successes = Counter()
    sampler._door_pull_attempts = Counter()
    sampler._door_pull_successes = Counter()
    sampler._door_placement_failures = Counter()
    sampler._door_action_side_rejections = Counter()
    sampler._door_status = {}
    sampler._active_door_name = None
    sampler._door_schedule_house_index = None
    return sampler


def _task(door_name: str, pushing: bool):
    task_config = SimpleNamespace(
        target_doorway_name="doorway",
        target_door_asset_id="asset",
        target_door_panel_count=1,
        robot_base_pose=[0.0, 0.0, 0.0],
    )
    return SimpleNamespace(
        door_object=SimpleNamespace(name=door_name),
        _is_pushing_door=pushing,
        config=SimpleNamespace(task_config=task_config),
    )


def test_fixed_attempts_are_not_inflated_by_placement_failure(tmp_path):
    sampler = _build_sampler(tmp_path)
    sampler.config.task_sampler_config.fixed_rollout_attempts_per_door = 10
    sampler._initialize_door_schedule(["door_a"])
    sampler._active_door_name = "door_a"
    sampler.report_rollout_result(_task("door_a", True), False, seed=1)
    sampler.mark_active_door_unplaceable("no_valid_pose")

    pending = json.loads(
        (tmp_path / "per_door_manifests" / "house_7.json").read_text()
    )["doors"]["door_a"]
    assert pending["attempts"] == 1
    assert pending["placement_failures"] == 1
    assert pending["status"] == "pending"

    sampler.mark_active_door_unplaceable("no_valid_pose")
    manifest = json.loads(
        (tmp_path / "per_door_manifests" / "house_7.json").read_text()
    )
    result = manifest["doors"]["door_a"]
    assert result["attempts"] == 1
    assert result["placement_failures"] == 2
    assert result["status"] == "sampling_shortfall"


def test_fixed_attempts_record_actions_and_durable_trials(tmp_path):
    sampler = _build_sampler(tmp_path)
    sampler.config.task_sampler_config.fixed_rollout_attempts_per_door = 3
    sampler._initialize_door_schedule(["door_a"])
    first = sampler.report_rollout_result(_task("door_a", True), True, seed=4)
    sampler.report_rollout_result(_task("door_a", True), False, seed=5)
    sampler.report_rollout_result(
        _task("door_a", False),
        False,
        seed=6,
        termination_reason="planner_exception",
        exception=RuntimeError("planning failed"),
    )

    assert sampler.is_house_sampling_complete(7)
    result = json.loads(
        (tmp_path / "per_door_manifests" / "house_7.json").read_text()
    )["doors"]["door_a"]
    assert result["attempts"] == 3
    assert result["push"]["attempts"] == 2
    assert result["pull"]["attempts"] == 1
    assert first is not None and first.exists()
    records = sorted((tmp_path / "trial_outcomes").rglob("trial_*.json"))
    assert len(records) == 3
    assert json.loads(records[-1].read_text())["exception_type"] == "RuntimeError"
    sampler._door_attempts.clear()
    sampler._door_successes.clear()
    sampler._write_per_door_manifest()
    rebuilt = json.loads(
        (tmp_path / "per_door_manifests" / "house_7.json").read_text()
    )["doors"]["door_a"]
    assert rebuilt["attempts"] == 3
    assert rebuilt["successes"] == 1


def test_trial_h5_status_is_updated_atomically(tmp_path):
    sampler = _build_sampler(tmp_path)
    sampler._initialize_door_schedule(["door_a"])
    record = sampler.report_rollout_result(_task("door_a", True), True)
    h5_path = tmp_path / "trajectory.h5"
    sampler.update_trial_h5_status(record, h5_saved=True, h5_path=h5_path)
    payload = json.loads(record.read_text())
    assert payload["h5_saved"]
    assert payload["h5_path"] == str(h5_path)
    assert not record.with_suffix(".json.tmp").exists()


def test_checkpoint_failure_remains_visible_in_trial_record(
    tmp_path, monkeypatch
):
    sampler = _build_sampler(tmp_path)
    sampler._initialize_door_schedule(["door_a"])
    record = sampler.report_rollout_result(_task("door_a", True), False)
    monkeypatch.setattr(
        per_door_rollout_runner,
        "save_house_trajectories",
        lambda *args, **kwargs: None,
    )
    logger = SimpleNamespace()
    PerDoorRolloutRunner._checkpoint_episode(
        logger,
        {"history": {}, "sensor_suite": object()},
        tmp_path,
        SimpleNamespace(),
        "",
        1,
        sampler,
        record,
        None,
        None,
        None,
    )
    payload = json.loads(record.read_text())
    assert not payload["h5_saved"]
    assert payload["h5_path"] is None


def test_per_door_completion_is_house_scoped(tmp_path):
    sampler = _build_sampler(tmp_path)
    sampler._initialize_door_schedule(["door_a"])
    sampler.report_rollout_result(_task("door_a", True), True)
    assert sampler.is_house_sampling_complete(7)
    assert not sampler.is_house_sampling_complete(8)


def test_required_action_filter_retries_wrong_side(monkeypatch, tmp_path):
    sampled_tasks = iter([_task("door_a", False), _task("door_a", True)])
    monkeypatch.setattr(
        DoorOpeningTaskSampler,
        "_sample_task",
        lambda self, env: next(sampled_tasks),
    )
    sampler = _build_sampler(tmp_path)
    sampler.config.task_sampler_config.required_door_action = "push"
    sampler.config.task_sampler_config.max_action_side_placement_attempts = 3
    sampler.config.task_config = SimpleNamespace(
        robot_base_pose=[1.0, 2.0, 3.0],
        door_body_name="door_a",
    )
    sampler._active_door_name = "door_a"

    task = sampler._sample_task(SimpleNamespace())
    assert task._is_pushing_door
    assert sampler._door_action_side_rejections["door_a"] == 1


def test_open_only_placement_uses_fifty_retries(monkeypatch, tmp_path):
    sampler = object.__new__(SinglePanelDoorOpeningTaskSampler)
    sampler.config = SimpleNamespace(
        task_sampler_config=SimpleNamespace(
            max_robot_placement_attempts=50,
            base_pose_sampling_radius_range=(0.8, 1.2),
            robot_safety_radius=0.35,
            check_robot_placement_visibility=False,
        )
    )
    sampler.used_robot_positions = defaultdict(list)
    sampler.get_visibility_resolver = lambda env: None
    captured = {}

    class FakeDoor:
        def __init__(self, name, data):
            self.name = name

        def get_hinge_joint_index(self):
            return 0

        def set_joint_position(self, index, value):
            return None

    monkeypatch.setattr(
        "lrl_molmospaces_extensions.tasks.nav_to_door_opening_task_sampler.Door",
        FakeDoor,
    )
    env = SimpleNamespace(
        mj_datas=[object()],
        current_robot=SimpleNamespace(
            robot_view=SimpleNamespace(
                base=SimpleNamespace(pose=[[0, 0, 0, 0]] * 4)
            )
        ),
    )

    def place_robot_near(**kwargs):
        captured["max_tries"] = kwargs["max_tries"]
        return False

    env.place_robot_near = place_robot_near
    try:
        sampler._place_robot_for_open_only(env, ["door_a"])
    except ValueError:
        pass
    assert captured["max_tries"] == 50


def test_single_panel_filter_uses_physical_doorway_metadata():
    scene_metadata = {
        "objects": {
            "doorway_single": {
                "asset_id": "Doorway_10",
                "name_map": {
                    "bodies": {"panel": "Doorway_10_doorway_door_10"}
                },
            },
            "doorway_double": {
                "asset_id": "Doorway_Double_1",
                "name_map": {
                    "bodies": {
                        "left": "Doorway_Double_1_doorway_door_left",
                        "right": "Doorway_Double_1_doorway_door_right",
                    }
                },
            },
        }
    }
    panel_info = door_panel_info_by_body_name(scene_metadata)
    assert panel_info["panel"].is_single_panel
    assert not panel_info["left"].is_single_panel
    accepted, rejected = filter_single_panel_door_names(
        ["panel", "left", "right", "unknown"], scene_metadata
    )
    assert accepted == ["panel"]
    assert rejected == ["left", "right", "unknown"]


def test_validator_rejects_sampling_shortfall(tmp_path):
    manifest_dir = tmp_path / "per_door_manifests"
    manifest_dir.mkdir()
    (manifest_dir / "house_1.json").write_text(
        json.dumps(
            {
                "house_index": 1,
                "doors": {
                    "door": {
                        "attempts": 2,
                        "successes": 0,
                        "status": "sampling_shortfall",
                        "placement_failures": 1,
                    }
                },
            }
        )
    )
    valid, summary = validate_run(tmp_path, expected_attempts=10)
    assert not valid
    assert summary["incomplete_doors"][0]["attempts"] == 2
