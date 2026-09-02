# Nav-To-Door-Opening Rollout Roadmap

## Current Status

We have a working prototype for generating long-horizon RB-Y1 trajectories that combine navigation and door opening:

- Navigation phase: uses the MolmoSpaces A* navigation policy to move the robot from a farther start pose toward the active door handle.
- Manipulation phase: uses the existing cuRobo-based door-opening policy after navigation handoff.
- Composite layer: uses our added `NavToDoorOpeningTask`, `NavToDoorOpeningTaskSampler`, and `NavThenDoorOpeningPolicy`.
- Output format: successful rollouts are saved as MolmoSpaces/MolmoBot-style H5 trajectories with corresponding camera videos.

The prototype has already produced successful saved examples for both shorter success-search runs and a long-distance run. The main remaining bottleneck is not only navigation, but the full handoff from navigation into cuRobo door opening.

## Completed Ablation Results

The custom nav-door A* policy eliminated the `m > k must hold` spline failure without changing the original MolmoSpaces smooth A* policy.

Distance-specific strict-handoff runs used a `0.95m` maximum handoff distance and a `0.85m` alignment target:

- Short (`0.8-2.0m`): 4 saved successes.
- Medium (`2.0-4.0m`): 2/11 completed attempts succeeded.
- Long (`4.0-8.0m`): 1/11 completed attempts succeeded.

Balanced-handoff runs used a `1.25m` maximum handoff distance, a `1.10m` alignment target, and retained the 15-degree yaw requirement:

- Medium (`2.0-4.0m`): 2/8 completed attempts succeeded.
- Long (`4.0-8.0m`): 2/9 completed attempts succeeded.
- Successful balanced handoffs were approximately `1.05-1.09m` from the handle.
- No balanced episode entered `final_align`, because successful navigation endpoints already satisfied the balanced distance and yaw limits.

The sample size is small, but balanced handoff improved observed medium success from 18% to 25% and long success from 9% to 22%. Remaining failures were dominated by disconnected A* start/goal regions and cuRobo execution timeouts.

## Recent Pipeline Improvements

The latest changes were made on the data-generation/framework side, not in the MolmoBot training loop:

- Added stricter handoff behavior before door opening starts.
  - The policy now avoids starting cuRobo when the robot is still too far from the handle.
  - This should prevent wasting manipulation attempts from poor base poses.

- Added a final base alignment phase before cuRobo.
  - The robot attempts to move closer to a target standoff near the handle.
  - The base is rotated to face the handle before door-opening planning starts.
  - This improves the starting condition for cuRobo without modifying the cuRobo planner itself.

- Kept the changes scoped to the nav-to-door-opening composite policy/config.
  - We did not change the MolmoBot training process.
  - We did not change the core cuRobo door-opening solver.

## Remaining Work

### 1. Add Connectivity-Aware Start Sampling

Reject sampled robot starts that cannot reach the navigation goal on the same downscaled A* free-space graph:

- Reuse the occupancy map cached by the nav-door task sampler.
- Reproduce the door-handle navigation goal used by the runtime navigation policy.
- Map the sampled start and goal to connected-component labels.
- Accept the placement only when both belong to the same navigable component.
- Resample rejected poses before constructing an episode.

This avoids counting geometrically impossible starts as navigation failures and increases useful rollout yield.

Implementation status:

- Implemented in `NavToDoorOpeningTaskSampler`.
- Reuses the occupancy map cached during door-opening scene initialization.
- Labels the same four-connected, downscaled free-space representation used by A*.
- Reproduces the runtime door-handle navigation goal before accepting a placement.
- Resamples collision-free poses when the start and goal component labels differ.
- Smoke job `6555707` found 7 navigable components, accepted a start/goal pair in component 1, completed navigation in 32 steps across 25 waypoints, and saved a successful full nav-to-door-opening trajectory.

### 2. Continue Distance Curriculum Sweeps

After connectivity-aware sampling passes a smoke test, rerun:

- Medium: `2.0-4.0m`
- Long: `4.0-8.0m`
- Very long: `8.0-12.0m`

Continue recording:

- `navigation_succeeded`
- `handoff_success`
- `handoff_distance_to_handle`
- `handoff_yaw_error`
- `phase`
- final saved trajectory success rate

### 3. Tune Handoff Parameters

The current values are first-pass guesses and should be tuned empirically:

- `handoff_max_distance_to_handle_m`
- `handoff_after_nav_failure_max_distance_to_handle_m`
- `final_align_target_distance_to_handle_m`
- `final_align_max_steps`
- `final_align_step_size_m`
- `final_align_yaw_threshold_rad`

Expected direction:

- Too strict: fewer attempts reach cuRobo, but failures are cleaner.
- Too loose: more cuRobo attempts, but many fail because the robot starts from poor poses.
- Best setting: start cuRobo only when the handle is reachable and the base orientation is reasonable.

### 4. Add Target-Door Grounding

Stage 1 uses a target door that is visible and unambiguous in the initial head-camera view:

- Require a minimum target-door segmentation fraction.
- Optionally require target-handle visibility for later point-prompt experiments.
- Reject starts where another door is also visibly competing with the target.
- Face the target during Stage-1 placement so the initial observation contains the goal.
- Save target-door, handle-visibility, and competing-door metadata in `obs_scene`.
- Use the instruction `Navigate to the visible door and ...`.

Implementation status:

- Opt-in grounding fields were added to the door sampler and task configs.
- Visibility and ambiguity rejection were added only to `NavToDoorOpeningTaskSampler`.
- `NavToDoorOpeningTask` now records grounding metadata in generated trajectories.
- `RBY1NavDoorOpeningVisibleGroundingSmokeConfig` provides an isolated Stage-1 test.
- Smoke job `6557043` is queued for end-to-end validation.

Stage 2 uses the existing `object_image_points/door_handle` annotations and MolmoBot point-prompt support:

- Requires both the target door and one of its handles to be visible initially.
- Allows other visible doors because the target handle is explicitly pointed out.
- Uses the instruction `Navigate to the pointed door and ...`.
- Is available through `RBY1NavDoorOpeningPointPromptGroundingSmokeConfig`.
- Point-prompt smoke job `6557045` accepted a target immediately despite four visible competing doors, recorded nine target-handle points in frame 0, and completed navigation. Door manipulation failed, so this smoke trajectory is not training-success data.

The grounding implementation is isolated in nav-door-specific task and sampler config subclasses. Users can select `none`, `visible_unique`, `point_prompt`, or `room_door_id` through:

```bash
python scripts/run_rby1_nav_door_grounding.py --grounding-mode point_prompt
```

The `room_door_id` mode does not require the target door to be initially visible. It supplies:

- The robot's initial `room_#` ID.
- A stable `house_#/door_#` target ID.
- The target handle's map XY coordinates.
- Up to two room IDs nearest the target door.
- A generated instruction containing the same structured goal.

The ID is scene-local and is not visually meaningful by itself; target coordinates make the goal actionable with the base pose already present in robot state. This mode assumes MolmoSpaces or another localization system provides equivalent room/door goal metadata during evaluation.

Stage 3 may add a separate target-door reference image for initially unseen targets.

Stage-1 smoke result:

- `visible_unique` eventually found a valid target and completed navigation, but required 149 placement rejections because other doors remained visible.
- Door manipulation failed in the accepted smoke episode.
- This mode is useful as a controlled baseline but is too restrictive for efficient large-scale generation in multi-door scenes.

### 5. Add Reachability Checks Before cuRobo

Before starting the door-opening policy, add a cheap reachability check:

- Is the handle within a plausible arm/base reach envelope?
- Is the base orientation compatible with reaching the handle?
- Is the robot on the correct side of the door for the selected push/pull behavior?

If the check fails, the policy should either keep aligning or terminate early with clear metadata rather than spending time on cuRobo retries.

### 6. Improve cuRobo Retry Diversity

If cuRobo still fails frequently after alignment, improve the retry strategy:

- Try multiple pregrasp offsets.
- Try multiple approach angles.
- Try slightly different handle standoff poses.
- Recompute push/pull side after final alignment.
- Avoid repeating the same failing grasp attempt many times.

This should increase manipulation success without changing MolmoBot training code.

### 7. Generate A Larger Clean Dataset

After tuning the pipeline:

- Run a larger success-search job.
- Save only successful rollouts.
- Keep a clean output directory structure under `assets/experiment_output/datagen`.
- Generate GIFs for manual review.
- Remove failed/debug/diagnostic outputs that could confuse later training.

### 8. Prepare Data For MolmoBot Fine-Tuning

Once enough successful long-horizon trajectories exist:

- Validate the H5 files.
- Build the MolmoBot-readable training index format.
- Confirm the data can be loaded by `MolmoBot/launch_scripts/train_molmobot.py`.
- Start with a small fine-tuning smoke run before scaling.

The training process itself should remain unchanged at first. The primary goal is to improve the quality and quantity of generated trajectories before introducing training-side changes.

## Recommended Next Step

Repeat the balanced medium/long sweeps with connectivity-aware sampling and compare the proportion of attempts lost to A* disconnected-component failures against the previous runs.
