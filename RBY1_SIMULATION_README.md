# RBY1 Simulation README

This README is the practical workflow for running RBY1 in simulation after the machine has been set up. For installation, drivers, Conda packages, CUDA/cuRobo, assets, and MolmoBot-SPOC setup, start with:

```text
MOLMOSPACES_RBY1_SETUP.md
```

## What We Are Running

There are two related workflows:

- MolmoSpaces data generation/debug configs run scripted or planner-based policies in simulation.
- MolmoBot-SPOC eval runs a learned policy checkpoint while still using MolmoSpaces for the simulator, RBY1 robot, scenes, cameras, benchmark episodes, and video/data output.

For learned RBY1 rigid manipulation, use:

```text
MolmoBot-SPOC policy/model: allenai/MolmoBot-SPOC-RBY1Rigid
MolmoSpaces benchmark: molmospaces-bench-v2/.../rby1_benchmarks/pick_benchmark
Config: molmobot_spoc.eval.config.rby1_eval_config:RBY1RigidManipEvalConfig
```

Do not use the older benchmark-v1 RBY1 pick path for MolmoBot-SPOC RBY1 rigid eval. It uses the wrong camera setup for this model.

## Prepare Benchmark Assets

Before running a few fixed benchmark episodes, prepare the scene/object/grasp assets for those indices. This avoids missing Objaverse mesh errors during eval without downloading the full Objaverse asset set.

```bash
cd /home/jinyoon/workspace/live-robotics-lab-project/molmospaces
conda activate mlspaces

export PYTHONPATH="/home/jinyoon/workspace/live-robotics-lab-project/molmospaces:${PYTHONPATH}"

python scripts/benchmarks/prepare_benchmark_assets.py \
  --benchmark_dir /home/jinyoon/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260415/procthor-objaverse/rby1_benchmarks/pick_benchmark \
  --idx 0 1 2
```

This script:

- reads the selected benchmark episode indices,
- finds each episode's ProcTHOR-Objaverse scene XML,
- installs missing `objects/objaverse` packages for those scenes,
- installs matching `grasps/droid_objaverse` packages,
- skips scene/object/grasp packages already present in the cache.

The cache is additive. Cancelling after an index finishes does not remove assets already extracted.

## Run MolmoBot-SPOC RBY1 Eval

Run one prepared benchmark episode:

```bash
cd /home/jinyoon/workspace/live-robotics-lab-project/MolmoBot/MolmoBot-SPOC
conda activate mlspaces

export PYTHONPATH="/home/jinyoon/workspace/live-robotics-lab-project/molmospaces:${PYTHONPATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export JAX_PLATFORMS=cpu

python -m molmo_spaces.evaluation.eval_main \
  molmobot_spoc.eval.config.rby1_eval_config:RBY1RigidManipEvalConfig \
  --benchmark_dir /home/jinyoon/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260415/procthor-objaverse/rby1_benchmarks/pick_benchmark \
  --no_wandb \
  --num_workers 1 \
  --idx 2
```

Change `--idx 2` to another prepared episode index. Remove `--idx` only when intentionally running the full benchmark.

The default task horizon comes from the benchmark JSON. For the tested pick episodes, it was `20` seconds, which becomes `200` policy steps with `policy_dt_ms=100.0`. To test a longer rollout:

```bash
python -m molmo_spaces.evaluation.eval_main \
  molmobot_spoc.eval.config.rby1_eval_config:RBY1RigidManipEvalConfig \
  --benchmark_dir /home/jinyoon/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260415/procthor-objaverse/rby1_benchmarks/pick_benchmark \
  --no_wandb \
  --num_workers 1 \
  --idx 2 \
  --task_horizon_sec 40
```

## Outputs

MolmoBot-SPOC eval writes outputs under:

```text
MolmoBot/MolmoBot-SPOC/eval_output/RBY1RigidManipEvalConfig/<timestamp>/
```

Useful files include saved trajectories and camera videos under the per-house directory, for example:

```text
house_1016/
```

`success=False` is not a setup error by itself. It means the episode ran but the policy did not satisfy the benchmark success condition before the horizon ended. A missing-asset/setup failure usually shows a skipped house or `Total count: 0`.

## Current RBY1 Status

The current local results are useful for diagnosis, but they are not successful benchmark completions.

- The provided `allenai/MolmoBot-SPOC-RBY1Rigid` policy runs in the correct RBY1 pick benchmark environment and saves MP4/H5 rollouts, but the tested episodes were all marked `success=False`.
- MolmoSpaces planner-based pick runs also generated MP4/H5 rollouts for `house_1001` and `house_1047`, but they were still marked failed by the official success metric. In `house_1001`, the planner logs show a grasp event, but the final pick success condition was not satisfied.
- Planner attempts for `pnp_benchmark`, `opening_benchmark`, and `door_opening_benchmark` did not produce successful saved rollouts in the current local setup.
- Matching the authors' planner batch settings more closely caused CUDA out-of-memory on the local 8 GB RTX 4060 Laptop GPU. A 16 GB VRAM server is a better target for author-style planner settings.

The cleaned local result set keeps only generated rollouts with saved MP4/H5 files. Failed attempts that produced only logs/configs were removed.

## Next Steps

- Try the public RBY1 multitask checkpoint, `allenai/MolmoBot-RBY1Multitask`, once Hugging Face access is stable.
- Try the articulated RBY1 checkpoint, `allenai/MolmoBot-SPOC-RBY1Articulated`, on `opening_benchmark`.
- Re-run planner diagnostics on a larger GPU so author-style CuRobo batch sizes can be used without CUDA OOM.
- If planner grasping visually succeeds but benchmark success remains false, inspect or tune the post-grasp/lift phase rather than loosening the benchmark success metric.

## Quick MolmoSpaces Debug Runs

Scripted RBY1 door-opening debug with viewer:

```bash
cd /home/jinyoon/workspace/live-robotics-lab-project/molmospaces
conda activate mlspaces

export PYTHONPATH="/home/jinyoon/workspace/live-robotics-lab-project/molmospaces:${PYTHONPATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export JAX_PLATFORMS=cpu

python -m molmo_spaces.data_generation.main DoorOpeningDebugConfig
```

Headless door-opening debug:

```bash
python -m molmo_spaces.data_generation.main DoorOpeningNoViewerDebugConfig
```

Fast teleop sandbox:

```bash
python scripts/datagen/run_fast_teleop.py
```

These are MolmoSpaces scripted/debug workflows, not MolmoBot learned-policy eval.
