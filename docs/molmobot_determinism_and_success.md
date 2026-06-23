# MolmoBot Determinism and Task Success

This note explains two related questions:

1. Why MolmoBot model rollouts can vary across repeats, while planner and recorded-trajectory replay are more stable.
2. Where MolmoSpaces decides whether a MolmoBot rollout succeeded or failed.

## Why MolmoBot Can Vary Across Repeats

The shortest explanation is that MolmoBot is a closed-loop vision policy.

For every action chunk:

1. MuJoCo/MolmoSpaces renders camera images.
2. MolmoBot reads those images plus robot state and task text.
3. MolmoBot samples an action chunk.
4. The robot moves in simulation.
5. The next rendered images become the next model input.

Small visual or physics differences can therefore feed back into the model. If the model also samples actions with random noise, the trajectory can diverge quickly.

Planner and recorded replay behave differently:

- Planner rollouts compute actions from deterministic task/planner logic. Small rendering differences usually do not affect the next action because the planner is not sampling actions from camera pixels.
- Recorded-trajectory replay sends the same saved `commanded_action` sequence from the H5 file. Rendering is saved for inspection, but it does not decide the next action.
- MolmoBot uses rendered observations as model input and samples action chunks, so image differences and sampling noise can affect future actions.

## What We Changed For MolmoBot

The main model-side randomness we found was flow-matching action noise from `torch.randn(...)` during action generation.

Before the fix, the action sampler used unseeded random noise. That meant the first predicted action chunk could differ across repeated runs of the same episode.

We changed the RBY1 MolmoBot policy to support a dedicated seeded `torch.Generator`:

- `RBY1_MODEL_ACTION_SEED` environment variable or `action_generator_seed` config field controls the seed.
- The policy creates and resets a `torch.Generator`.
- The generator is passed into model action generation.

Relevant code:

- Seed config and generator setup: [`MolmoBot/MolmoBot/olmo/eval/configure_molmo_spaces.py`](../MolmoBot/MolmoBot/olmo/eval/configure_molmo_spaces.py)
- `_resolve_action_generator_seed()` reads `RBY1_MODEL_ACTION_SEED`.
- `_reset_action_generator()` creates `torch.Generator(...)` and calls `manual_seed(...)`.
- `_populate_action_buffer(...)` passes `generator=self._action_generator` into `self.agent.get_action_chunk(...)`.
- RBY1 model reproducibility config: [`configs/rby1_model_repro.yaml`](../configs/rby1_model_repro.yaml)
- Model runner repeat support: [`scripts/rby1_runner.py`](../scripts/rby1_runner.py)

After this change, the model sampling noise is repeatable. In our repeat tests, high-level success/failure outcomes became consistent. Byte-level identical videos or H5 trajectories are still not guaranteed because GPU rendering, MuJoCo contact dynamics, and floating-point execution can still differ slightly.

## Why Rendering Differences Matter More For Models

Rendering randomness by itself does not automatically cause task-level nondeterminism.

It matters most when the policy consumes rendered images and uses them to choose actions. MolmoBot does that. Planner/replay mostly do not.

So the practical difference is:

- In planner/replay, image differences mostly affect what we save to video.
- In MolmoBot, image differences can affect what the model does next.

That is why a tiny image difference can become a different robot path for MolmoBot, but not necessarily for planner or replay.

## Small Q&A: Did We Fix Rendering?

**Q: Did we change rendering code to make MolmoBot deterministic?**

No. The main change we made was not in MuJoCo rendering or camera setup. The determinism fix was in MolmoBot action sampling: we added a seeded `torch.Generator` and passed it into action generation.

**Q: Was the rendering/camera mode already there?**

Yes. The JSON-eval camera path was already part of MolmoSpaces. In the normal benchmark path, `JsonEvalTaskSampler.setup_cameras(...)` uses the cameras recorded in the benchmark JSON. Texture randomization is also disabled in our evaluation logs.

**Q: What about eval camera randomization?**

MolmoSpaces also has an eval-camera randomization path. When it is enabled, it derives a deterministic seed from the episode identity through `derive_episode_camera_seed(...)`, then uses that seed in `setup_eval_cameras(...)`. That code already existed; it is not the model-side fix we added.

**Q: Why did we talk about rendering, then?**

Because MolmoBot consumes rendered images. Even if the rendering code is not intentionally random, GPU/MuJoCo rendering and physics can still have tiny non-bitwise differences. Those differences matter more for MolmoBot than for planner/replay because MolmoBot chooses future actions from images.

**Q: What exactly did we change?**

We changed model action sampling in [`MolmoBot/MolmoBot/olmo/eval/configure_molmo_spaces.py`](../MolmoBot/MolmoBot/olmo/eval/configure_molmo_spaces.py):

- Read `RBY1_MODEL_ACTION_SEED` / `action_generator_seed`.
- Create a seeded `torch.Generator`.
- Reset it at policy reset.
- Pass it into `self.agent.get_action_chunk(...)`.

## How Success and Failure Are Decided

MolmoSpaces decides success inside the task class, not inside MolmoBot.

The general rollout flow is:

1. The policy returns actions.
2. The rollout loop steps the task in MuJoCo.
3. At the end of the rollout, MolmoSpaces calls `task.judge_success()`.
4. That Boolean is counted in the evaluation summary.
5. The trajectory H5 also stores success information, which can later be collected into per-episode results.

Relevant code:

- Rollout loop and final success call: [`molmospaces/molmo_spaces/data_generation/pipeline.py`](../molmospaces/molmo_spaces/data_generation/pipeline.py)
- Look for `run_single_rollout(...)`, the `task.step(...)` loop, and `success = task.judge_success()`.
- House-level counting and saving: [`molmospaces/molmo_spaces/data_generation/pipeline.py`](../molmospaces/molmo_spaces/data_generation/pipeline.py)
- Look for `house_success_count`, `house_total_count`, and `house_raw_histories`.
- Evaluation result object and success rate: [`molmospaces/molmo_spaces/evaluation/eval_main.py`](../molmospaces/molmo_spaces/evaluation/eval_main.py)
- Per-episode H5 result collection: [`molmospaces/molmo_spaces/utils/eval_utils.py`](../molmospaces/molmo_spaces/utils/eval_utils.py)

## Task-Specific Success Rules

Each task has its own `judge_success()` implementation.

### Pick

Code: [`molmospaces/molmo_spaces/tasks/pick_task.py`](../molmospaces/molmo_spaces/tasks/pick_task.py)

`PickTask.judge_success()` returns `self.get_info()[0]["success"]`.

The task computes success from pick-specific object state, such as whether the target object is lifted/held according to the task metrics.

### Pick And Place

Code: [`molmospaces/molmo_spaces/tasks/pick_and_place_task.py`](../molmospaces/molmo_spaces/tasks/pick_and_place_task.py)

`PickAndPlaceTask.judge_success()` returns `self.get_info()[0]["success"]`.

The success metric checks whether the picked object is supported by the target receptacle and whether the receptacle stayed within allowed displacement/tilt thresholds.

The replay PnP compatibility issue we saw came from benchmark task fields:

- Current JSON eval expects `max_place_receptacle_pos_displacement = 0.15`.
- Current JSON eval expects `max_place_receptacle_rot_displacement = radians(60)`.

Relevant compatibility check:

- [`molmospaces/molmo_spaces/tasks/json_eval_task_sampler.py`](../molmospaces/molmo_spaces/tasks/json_eval_task_sampler.py)

We patched replay benchmark generation so future PnP replay benchmarks use those expected values:

- [`scripts/rby1_replay_runner.py`](../scripts/rby1_replay_runner.py)

### Open / Close

Code: [`molmospaces/molmo_spaces/tasks/opening_tasks.py`](../molmospaces/molmo_spaces/tasks/opening_tasks.py)

`OpeningTask.judge_success()` checks whether reward is above `task_success_threshold`.

In practice, this means the articulated object joint, such as drawer or door opening, reached the task's required open/close amount.

## Where To Start Reading The Code

If you want the shortest path through the code, read in this order:

1. MolmoBot action generation and seeded sampling:
   [`MolmoBot/MolmoBot/olmo/eval/configure_molmo_spaces.py`](../MolmoBot/MolmoBot/olmo/eval/configure_molmo_spaces.py)
2. Evaluation entry point and summary object:
   [`molmospaces/molmo_spaces/evaluation/eval_main.py`](../molmospaces/molmo_spaces/evaluation/eval_main.py)
3. Rollout loop and success counting:
   [`molmospaces/molmo_spaces/data_generation/pipeline.py`](../molmospaces/molmo_spaces/data_generation/pipeline.py)
4. JSON benchmark task construction:
   [`molmospaces/molmo_spaces/tasks/json_eval_task_sampler.py`](../molmospaces/molmo_spaces/tasks/json_eval_task_sampler.py)
5. Task-specific success:
   [`molmospaces/molmo_spaces/tasks/pick_task.py`](../molmospaces/molmo_spaces/tasks/pick_task.py),
   [`molmospaces/molmo_spaces/tasks/pick_and_place_task.py`](../molmospaces/molmo_spaces/tasks/pick_and_place_task.py),
   [`molmospaces/molmo_spaces/tasks/opening_tasks.py`](../molmospaces/molmo_spaces/tasks/opening_tasks.py)

## Bottom Line

Planner and replay are useful controls because their actions are less dependent on rendered pixels. MolmoBot is a vision-conditioned stochastic model, so its trajectory can change when rendering or action sampling changes.

The model-side fix we made was to seed MolmoBot's action generator. Success/failure, however, is still decided by MolmoSpaces task logic through `judge_success()`, not by the model itself.
