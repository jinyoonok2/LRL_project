# VLA Data-Generation File Manifest

> Historical research record. For current setup, paths, and source versions, use [PROJECT_ENVIRONMENT.md](PROJECT_ENVIRONMENT.md) and [SETUP.md](SETUP.md). Proposed steps and implementation locations below may have been superseded.

Snapshot: 2026-09-08  
Branch: `rby1-long-horizon-vla-training`

## Boundary

Long-horizon VLA data generation is now implemented as a sibling extension
package. It imports MolmoSpaces APIs but does not require VLA-specific behavior
inside existing MolmoSpaces author files.

The earlier, accepted MolmoBot/MolmoSpaces compatibility fixes are intentionally
outside this manifest and remain in place. Those include benchmark compatibility,
RBY1 body-ID/contact fixes, deterministic model evaluation, and local cuRobo
support.

## Added: extension package

All VLA task, policy, sampling, configuration, and accounting behavior now lives
under `lrl_molmospaces_extensions/`.

### Package and documentation

- `lrl_molmospaces_extensions/__init__.py`
- `lrl_molmospaces_extensions/README.md`

### Experiment configurations

- `lrl_molmospaces_extensions/configs/policy_configs.py`
- `lrl_molmospaces_extensions/configs/door_opening.py`
- `lrl_molmospaces_extensions/configs/nav_pick_and_place.py`

### Long-horizon tasks and samplers

- `lrl_molmospaces_extensions/tasks/nav_to_door_opening_task.py`
- `lrl_molmospaces_extensions/tasks/nav_to_door_opening_task_sampler.py`
- `lrl_molmospaces_extensions/tasks/nav_to_pick_and_place_task.py`
- `lrl_molmospaces_extensions/tasks/nav_to_pick_and_place_task_sampler.py`
- `lrl_molmospaces_extensions/tasks/util_samplers/door_panels.py`
- `lrl_molmospaces_extensions/tasks/util_samplers/navgoal_sampler.py`

### Policies and navigation

- `lrl_molmospaces_extensions/policy/nav_then_door_opening_policy.py`
- `lrl_molmospaces_extensions/policy/nav_then_pick_and_place_policy.py`
- `lrl_molmospaces_extensions/policy/navigation/astar_planner.py`
- `lrl_molmospaces_extensions/policy/navigation/astar_planner_policy.py`

### Data-generation runner and validation

- `lrl_molmospaces_extensions/datagen/main.py`
- `lrl_molmospaces_extensions/datagen/per_door_rollout_runner.py`
- `lrl_molmospaces_extensions/datagen/validate_shard.py`

The extension runner owns:

- per-door scheduling and house completion;
- exact executed-rollout accounting;
- separate placement-failure and action-side-rejection accounting;
- repeated placement sampling for the same door (up to 20 batches of 50 pose
  candidates) before declaring a shortfall;
- planner-exception accounting;
- immediate trajectory checkpointing;
- atomic per-trial JSON records;
- `sampling_shortfall` status instead of artificial attempt inflation; and
- optional exact-attempt coverage auditing. This checks dataset completeness,
  not planner quality or training success, and does not gate later shards.

### Regression tests

- `lrl_molmospaces_extensions/tests/test_per_door_pipeline.py`

## Added: project launch and audit scripts

- `scripts/run_rby1_door_stability_kuorobot01.sh`
- `scripts/run_rby1_simple_visible_kuorobot01.sh`
- `scripts/audit_nav_door_panel_trajectories.py`

## Modified: project launchers

These launchers now add `LRL_project` to `PYTHONPATH` and load extension modules
through the `module:ConfigName` interface:

- `scripts/run_rby1_nav_door_grounding.py`

`scripts/run_rby1_door_stability_kuorobot01.sh` supports:

- `nav-open` and `open-only`;
- `push` and `pull`;
- same-door, ten-door, all-door, and shard A-F scopes; and
- an optional attempt-count override for smoke tests, with 10 as production
  default.

## Restored: existing MolmoSpaces author files

VLA-era edits were removed from the following existing author files. Their
relevant content was restored to the pre-VLA MolmoSpaces state, while equivalent
VLA behavior was placed in the extension package:

- `molmospaces/molmo_spaces/data_generation/pipeline.py`
- `molmospaces/molmo_spaces/configs/policy_configs.py`
- `molmospaces/molmo_spaces/configs/task_sampler_configs.py`
- `molmospaces/molmo_spaces/data_generation/config/door_opening_configs.py`
- `molmospaces/molmo_spaces/data_generation/config/object_manipulation_datagen_configs.py`
- `molmospaces/molmo_spaces/env/env.py`
- `molmospaces/molmo_spaces/planner/astar_planner.py`
- `molmospaces/molmo_spaces/policy/solvers/navigation/astar_planner_policy.py`
- `molmospaces/molmo_spaces/tasks/util_samplers/navgoal_sampler.py`

## Removed from MolmoSpaces: relocated VLA modules

These VLA-only modules were removed from the MolmoSpaces package after being
relocated to `lrl_molmospaces_extensions/`:

- `molmospaces/molmo_spaces/policy/solvers/nav_then_door_opening_policy.py`
- `molmospaces/molmo_spaces/policy/solvers/nav_then_pick_and_place_policy.py`
- `molmospaces/molmo_spaces/tasks/nav_to_door_opening_task.py`
- `molmospaces/molmo_spaces/tasks/nav_to_door_opening_task_sampler.py`
- `molmospaces/molmo_spaces/tasks/nav_to_pick_and_place_task.py`
- `molmospaces/molmo_spaces/tasks/nav_to_pick_and_place_task_sampler.py`
- `molmospaces/molmo_spaces/tasks/util_samplers/door_panels.py`
- `molmospaces/mlspaces_tests/data_generation/test_per_door_pipeline.py`

## VLA documentation and presentation artifacts

- `docs/RESEARCH_PLAN.md` consolidates the earlier navigation/fine-tuning plans. Original documents remain available in Git history; the environment/setup guides stay separate.
- `docs/slide_history/2026-09-07_per_door_vla_dataset_progress.md`
- `docs/slide_history/2026-09-14_single_panel_door_stability_plan.md`
- `docs/slide_history/media/`: current 90 expert/model MP4 comparisons and grasp metrics; older GIF galleries were removed.
- `docs/single_panel_trajectory_audit_2026-09-07.json`
- `docs/single_panel_trajectory_audit_2026-09-07.single_panel_files.txt`

Generated run outputs and logs under `molmospaces/assets/experiment_output/`,
`logs/`, and `slurm_logs/` are runtime artifacts, not source-code changes.
