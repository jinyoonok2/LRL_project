# Long-Horizon Rollout Generation And MolmoBot Fine-Tuning Plan

This note records the updated direction for training experiments with RB-Y1 MolmoBot.

The goal is not to train a VLA from scratch. The goal is:

1. Generate new long-horizon MolmoSpaces rollouts.
2. Save them in the same trajectory format used by MolmoBot training.
3. Fine-tune the existing released MolmoBot RB-Y1 checkpoint on those new rollouts.
4. Test whether the existing model can learn the new task distribution.

## Updated Training Direction

The earlier smoke-training experiment showed that the MolmoBot training stack can run, but it used a small public training subset and mainly helped estimate runtime.

The actual research direction is different:

- Create our own rollouts in MolmoSpaces.
- Make the rollouts different from the released MolmoBot data.
- Focus on longer-horizon mobile-manipulation tasks.
- Fine-tune from the released MolmoBot checkpoint instead of starting from random weights.

The desired task structure is:

```text
start far from target
    -> navigate from point A to point B
    -> reach the object/task area
    -> perform manipulation
    -> save the whole episode as one training trajectory
```

Example task families:

- Navigate to an object, then pick it.
- Navigate to an object, then pick and place it.
- Navigate to a receptacle, then place an object.
- Navigate to a cabinet/door, then open it.
- Navigate through a room, then perform a manipulation task.

## What MolmoSpaces Already Provides

MolmoSpaces already has the major pieces needed for this plan.

Navigation:

- `molmo_spaces/tasks/nav_task.py`
  - `NavToObjTask`
- `molmo_spaces/tasks/nav_task_sampler.py`
  - `NavToObjTaskSampler`
- `molmo_spaces/policy/solvers/navigation/astar_planner_policy.py`
  - `AStarPlannerPolicy`
- `molmo_spaces/data_generation/config/nav_to_obj_configs.py`
  - `NavToObjDataGenConfig`

Manipulation:

- `molmo_spaces/data_generation/config/object_manipulation_datagen_configs.py`
  - `RBY1PickDataGenConfig`
  - `RBY1PickAndPlaceDataGenConfig`
  - `RBY1OpenDataGenConfig`
- `molmo_spaces/data_generation/config/door_opening_configs.py`
  - `DoorOpeningDataGenConfig`
- cuRobo-based planner policies already used by the existing RBY1 manipulation datagen configs.

Data generation pipeline:

- `molmo_spaces/data_generation/main.py`
  - main CLI entrypoint
- `molmo_spaces/data_generation/pipeline.py`
  - `ParallelRolloutRunner`
- `molmo_spaces/utils/save_utils.py`
  - H5 trajectory saving utilities

MolmoBot training consumption:

- `MolmoBot/olmo/data/synthmanip_dataset.py`
  - reads generated H5 trajectories through `valid_trajectory_index.json`
- `MolmoBot/olmo/data/synthmanip_presets.py`
  - defines RBY1 action/camera presets
- `MolmoBot/launch_scripts/train_molmobot.py`
  - fine-tuning/training entrypoint

## What Is Missing

MolmoSpaces can generate navigation-only data and manipulation-only data today. We have now added a first prototype of a single registered RBY1 task that performs:

```text
navigation -> manipulation
```

The integration layer now exists as a prototype, but it still needs runtime validation and hardening before it should be treated as a dataset-production pipeline.

## Current Prototype Status

The current prototype is focused on:

```text
navigate to pickup object -> pick and place
```

The added pieces are:

- `molmo_spaces/tasks/nav_to_pick_and_place_task.py`
  - `NavToPickAndPlaceTask`
- `molmo_spaces/tasks/nav_to_pick_and_place_task_sampler.py`
  - `NavToPickAndPlaceTaskSampler`
- `molmo_spaces/policy/solvers/nav_then_pick_and_place_policy.py`
  - `NavThenPickAndPlacePolicy`
- `molmo_spaces/configs/policy_configs.py`
  - `NavThenPickAndPlacePolicyConfig`
- `molmo_spaces/data_generation/config/object_manipulation_datagen_configs.py`
  - `RBY1NavPickAndPlaceDataGenConfig`
  - `RBY1NavPickAndPlaceDebugDataGenConfig`
  - `RBY1NavPickAndPlaceFastDebugDataGenConfig`
- `scripts/run_rby1_nav_pnp_cheetah05.sh`
  - Host-specific debug launcher for `cheetah05`

Supporting fixes were also added for A* navigation on the cluster:

- A* map creation now uses `MUJOCO_EGL_DEVICE_ID`.
- A* waypoint angle handling normalizes scalar/array values more robustly.

So the work is not inventing navigation or manipulation from scratch. The current code connects existing MolmoSpaces components into a first long-horizon rollout generator.

## Known Prototype Gaps

The current implementation is not yet fully validated. Known gaps:

- A* navigation failure can still look like a terminal `done=True` action to the sequential wrapper, so the wrapper needs to distinguish navigation success from navigation failure.
- The sampler places the robot farther from the pickup object, but it does not yet prove that the post-navigation pose is manipulation-ready.
- The handoff from navigation to manipulation needs more explicit logging and success criteria.
- The current debug launcher is tied to `cheetah05`; a Slurm launcher should be added before scaling.
- Successful H5/video output still needs to be confirmed end-to-end.

These gaps should be resolved before generating a large fine-tuning dataset.

## Proposed Implementation

### 1. Add A Composite Task

Add a task representing the whole episode:

```text
NavToManipTask
```

It should track two phases:

```text
NAVIGATE_TO_TARGET
MANIPULATE_TARGET
```

The success condition should require the final manipulation success, and optionally also require that the navigation phase reached a valid target area first.

Possible files:

- `molmo_spaces/tasks/nav_to_manip_task.py`
- or a task-specific version such as:
  - `molmo_spaces/tasks/nav_to_pick_task.py`
  - `molmo_spaces/tasks/nav_to_pick_and_place_task.py`

### 2. Add A Composite Sampler

The sampler should combine:

- navigation-style far robot placement
- manipulation-style object/receptacle selection
- path-validity checks
- task-specific constraints

For pick-and-place, the sampler can reuse logic from:

```text
PickAndPlaceTaskSampler
```

but change the robot start pose so the robot begins farther away, similar to:

```text
NavToObjTaskSampler
```

The sampler must ensure:

- the robot starts away from the target,
- A* can find a path to the target area,
- the object/receptacle placement is valid,
- the final manipulation task is reachable after navigation.

### 3. Add A Sequential Policy

Add a policy wrapper that delegates to existing policies:

```text
NavThenManipPolicy
```

Pseudo-flow:

```text
if phase == NAVIGATE_TO_TARGET:
    action = astar_navigation_policy.act(obs)
    if navigation_success:
        switch to MANIPULATE_TARGET

if phase == MANIPULATE_TARGET:
    action = manipulation_policy.act(obs)
```

The navigation part can use:

```text
AStarPlannerPolicy
```

The manipulation part can reuse the existing planner policy used by RBY1 Pick/PnP/Open datagen.

### 4. Add A New Data Generation Config

Add a registered config such as:

```text
RBY1NavPickAndPlaceDataGenConfig
```

Likely location:

```text
molmo_spaces/data_generation/config/object_manipulation_datagen_configs.py
```

or a new file:

```text
molmo_spaces/data_generation/config/nav_to_manip_configs.py
```

The config should set:

- RBY1 robot config
- RBY1 camera system
- long enough `task_horizon`
- navigation policy config
- manipulation policy config
- output path for generated trajectories

The task horizon needs to be longer than current manipulation-only tasks. A reasonable starting point is:

```text
task_horizon: 800-1000
```

because navigation may take several hundred steps before manipulation begins.

## Data Format For MolmoBot Fine-Tuning

The generated data should follow the existing MolmoSpaces trajectory format:

```text
traj_i/
  obs/agent/qpos
  obs/agent/qvel
  obs/extra/...
  obs/sensor_data/{camera}/...
  obs/sensor_param/...
  actions/joint_pos
  actions/joint_pos_rel
  success
  rewards
  terminated
  truncated
  obs_scene
```

After rollout generation, run the existing postprocessing pipeline:

1. Repair video paths if needed.
2. Validate trajectories.
3. Create `valid_trajectory_index.json`.
4. Compute action/state statistics.

Relevant scripts:

- `molmospaces/scripts/data/process_data.sh`
- `molmospaces/scripts/data/validate_trajectories.py`
- `molmospaces/scripts/data/calculate_stats.py`

MolmoBot training then uses:

```text
--action_preset RBY1_multitask
--camera_preset RBY1_full_with_head_gopro
--data_paths <generated_dataset_dir>
```

## Fine-Tuning Strategy

Start from the released RB-Y1 MolmoBot checkpoint:

```text
MolmoBot-RBY1Multitask
```

Then fine-tune on the generated long-horizon rollouts.

This should test whether the existing model can adapt to:

- longer context before manipulation,
- base navigation actions,
- delayed manipulation success,
- new task distributions,
- combined mobile-manipulation trajectories.

Use the chunked training launcher for long runs:

- `scripts/molmobot_chunked_train.py`
- `scripts/sbatch_molmobot_chunked_train.sh`

The first fine-tuning target should be small:

```text
100-500 generated successful trajectories
```

Then scale up only after confirming:

- the generated H5 files load in MolmoBot,
- normalization stats are valid,
- one short fine-tuning run completes,
- evaluation videos show improved behavior on the new task distribution.

## Recommended Prototype Order

1. Generate nav-only rollouts with `NavToObjDataGenConfig`.
2. Generate manipulation-only RBY1 rollouts with `RBY1PickAndPlaceDataGenConfig`.
3. Verify both data formats can be postprocessed into MolmoBot-readable training data.
4. Implement a simple two-phase policy wrapper:
   - A* navigate to object area.
   - Switch to existing RBY1 pick-and-place planner.
5. Generate a tiny long-horizon dataset.
6. Validate H5 trajectories and compute stats.
7. Fine-tune the existing MolmoBot checkpoint on the tiny dataset.
8. Evaluate qualitatively with videos.
9. Scale up rollout count and training chunks.

## Main Risks

Navigation and manipulation samplers may disagree:

- Navigation wants the robot far from the target.
- Manipulation planners assume the robot starts close enough to solve the task.

The composite sampler must guarantee that after navigation, the robot reaches a pose where the manipulation planner can succeed.

Other risks:

- Long trajectories may exceed model sequence assumptions.
- Failed rollouts must be filtered out before fine-tuning.
- Navigation actions and manipulation actions must use the same RBY1 action format.
- Camera views during navigation may differ from manipulation-only training data.
- The policy may learn navigation but forget manipulation if the new dataset is too narrow.

## Summary

This direction is feasible with the existing MolmoSpaces framework, but it requires a new composite rollout generator.

The main implementation is:

```text
existing A* navigation policy
    + existing RBY1 manipulation planner
    + new composite task/sampler/policy/config
    -> long-horizon rollout H5 data
    -> MolmoBot fine-tuning
```

This lets us test whether the released MolmoBot RB-Y1 model can learn new long-horizon mobile-manipulation tasks from generated demonstrations, without training the VLA from scratch.
