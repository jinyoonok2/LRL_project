# RBY1 Simulation And Evaluation Project

This repository is the working project for bringing up RBY1 simulation with
MolmoSpaces and running MolmoBot evaluation workflows.

The project combines two upstream codebases as submodules:

- `molmospaces`: simulation, benchmark assets, scenes, robots, cameras, and evaluation runtime
- `MolmoBot`: learned policy code and model evaluation entry points

The current focus is reproducible RBY1 simulation/evaluation, including asset
preparation, benchmark episode execution, saved rollout videos, and debugging of
Pick/PnP performance.

## Repository Layout

```text
LRL_project/
  MolmoBot/                  # MolmoBot submodule
  molmospaces/               # MolmoSpaces submodule
  rby1_diagram_images/       # RBY1 benchmark/evaluation diagrams
  MOLMOSPACES_RBY1_SETUP.md  # environment and dependency setup notes
  RBY1_SIM_BRINGUP_CHANGELOG.md
```

## Branches

The intended branch split is:

- `main`: clean simulation bringup baseline and documentation
- `rby1-custom`: active RBY1 evaluation workflow, configs, and experiment scripts

Use `rby1-custom` for the latest multitask evaluation workflow. Use `main` as a
clean reference branch.

## Setup

Start with the setup guide:

```text
MOLMOSPACES_RBY1_SETUP.md
```

That file documents the Conda environment, CUDA/cuRobo dependencies,
MolmoSpaces assets, MolmoBot-SPOC setup, and common troubleshooting steps.

After cloning, initialize submodules:

```bash
git submodule update --init --recursive
```

## What Runs Here

There are two related RBY1 workflows:

- MolmoSpaces debug/data-generation configs run scripted or planner-based
  policies in simulation.
- MolmoBot evaluation runs a learned policy checkpoint while using MolmoSpaces
  for the simulator, RBY1 robot, scenes, cameras, benchmark episodes, and saved
  video/data output.

For the original RBY1 rigid MolmoBot-SPOC workflow:

```text
Model: allenai/MolmoBot-SPOC-RBY1Rigid
Benchmark: molmospaces-bench-v2/.../rby1_benchmarks/pick_benchmark
Config: molmobot_spoc.eval.config.rby1_eval_config:RBY1RigidManipEvalConfig
```

Do not use the older benchmark-v1 RBY1 pick path for this model. It uses the
wrong camera setup for the RBY1 rigid checkpoint.

## Prepare Benchmark Assets

Before running fixed benchmark episodes, prepare the scene/object/grasp assets
for those indices. This avoids missing Objaverse mesh errors during evaluation
without downloading the full Objaverse asset set.

```bash
cd molmospaces
conda activate mlspaces

export PYTHONPATH="$PWD:${PYTHONPATH}"

python scripts/benchmarks/prepare_benchmark_assets.py \
  --benchmark_dir ~/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260415/procthor-objaverse/rby1_benchmarks/pick_benchmark \
  --idx 0 1 2
```

The cache is additive. Cancelling after an index finishes does not remove assets
that were already extracted.

## Run MolmoBot-SPOC RBY1 Eval

Run one prepared benchmark episode:

```bash
cd MolmoBot/MolmoBot-SPOC
conda activate mlspaces

export PYTHONPATH="../../molmospaces:${PYTHONPATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export JAX_PLATFORMS=cpu

python -m molmo_spaces.evaluation.eval_main \
  molmobot_spoc.eval.config.rby1_eval_config:RBY1RigidManipEvalConfig \
  --benchmark_dir ~/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260415/procthor-objaverse/rby1_benchmarks/pick_benchmark \
  --no_wandb \
  --num_workers 1 \
  --idx 2
```

Change `--idx 2` to another prepared episode index. Remove `--idx` only when
intentionally running the full benchmark.

## Outputs

MolmoBot-SPOC evaluation writes outputs under:

```text
MolmoBot/MolmoBot-SPOC/eval_output/RBY1RigidManipEvalConfig/<timestamp>/
```

Useful artifacts include per-house saved trajectories and camera videos.

`success=False` is not a setup error by itself. It means the episode ran but the
policy did not satisfy the benchmark success condition before the horizon ended.
A missing-asset/setup failure usually shows a skipped house or `Total count: 0`.

## Quick MolmoSpaces Debug Runs

Scripted RBY1 door-opening debug with viewer:

```bash
cd molmospaces
conda activate mlspaces

export PYTHONPATH="$PWD:${PYTHONPATH}"
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

These are MolmoSpaces scripted/debug workflows, not learned-policy MolmoBot
evaluation runs.

## Notes

The detailed bringup history is kept in:

```text
RBY1_SIM_BRINGUP_CHANGELOG.md
```

For the latest YAML-based RBY1 multitask evaluation workflow, switch to the
`rby1-custom` branch.
# Live Robotics Lab RBY1 Evaluation Project

This repository is the working RBY1 MolmoBot/MolmoSpaces evaluation workspace.
It keeps a clean `main` branch and an active `rby1-custom` branch for RBY1
multitask evaluation fixes, scripts, configs, and experiment results.

For machine setup, Conda/CUDA setup, assets, and Slurm usage, see:

```text
SETUP.md
```

## Repository Layout

```text
LRL_project/
  MolmoBot/                 # MolmoBot submodule
  molmospaces/              # MolmoSpaces submodule
  configs/                  # YAML experiment configs
  scripts/                  # Python runner + Slurm submit wrappers
  logs/                     # Per-run logs and summary.tsv files
  slurm_logs/               # Slurm stdout/stderr files
  MolmoBot/MolmoBot/eval_output/
    baseline/               # Preserved baseline result sets
    experiments/            # Parameter sweep outputs
```

## Active Branches

Use these branches for current work:

- `main`: clean/original-style branch
- `rby1-custom`: active development/evaluation branch

Old branch names such as `uva-cs-server`, `rby1-sim-bringup`, and
`rby1-policy-eval` are no longer part of the intended workflow.

## Current Workflow

The current workflow uses YAML configs plus one Python runner. Shell scripts are
only thin Slurm submission wrappers.

```text
YAML config
  -> sbatch wrapper
    -> scripts/rby1_runner.py
      -> MolmoSpaces asset preparation or MolmoBot run_eval.py
```

Main scripts:

- `scripts/rby1_runner.py`: Python orchestration for `describe`, `prepare`, and `eval`
- `scripts/sbatch_rby1_prepare.sh`: submit asset preparation for one YAML config
- `scripts/sbatch_rby1_eval.sh`: submit evaluation for one YAML config
- `scripts/sbatch_rby1_experiments.sh`: submit several evaluation configs

The old Bash orchestration scripts were removed. The main logic now lives in
`scripts/rby1_runner.py`.

## YAML Configs

Baseline configs:

- `configs/rby1_eval.yaml`: 30 selected indices across `pick`, `pnp`, `opening`, and `door_opening`
- `configs/rby1_pick_pnp_expansion.yaml`: larger Pick/PnP expansion set

Experiment configs:

- `configs/rby1_pick_pnp_binary.yaml`
- `configs/rby1_pick_pnp_horizon450.yaml`
- `configs/rby1_pick_pnp_horizon450_binary.yaml`
- `configs/rby1_pick_pnp_inverted.yaml`
- `configs/rby1_pick_pnp_horizon450_inverted.yaml`
- `configs/rby1_pick_pnp_binary_threshold_pos05.yaml`
- `configs/rby1_pick_pnp_binary_threshold_neg05.yaml`
- `configs/rby1_pick_pnp_binary_terminate.yaml`
- `configs/rby1_pick_pnp_horizon450_binary_terminate.yaml`

## Common Commands

Inspect a config without running anything:

```bash
python scripts/rby1_runner.py describe --config configs/rby1_eval.yaml
```

Dry-run asset preparation:

```bash
python scripts/rby1_runner.py prepare \
  --config configs/rby1_pick_pnp_binary.yaml \
  --dry-run
```

Dry-run evaluation:

```bash
python scripts/rby1_runner.py eval \
  --config configs/rby1_pick_pnp_binary.yaml \
  --tasks pick \
  --indices 10 \
  --dry-run
```

Submit asset preparation:

```bash
bash scripts/sbatch_rby1_prepare.sh configs/rby1_pick_pnp_binary.yaml
```

Submit one evaluation:

```bash
bash scripts/sbatch_rby1_eval.sh configs/rby1_pick_pnp_binary.yaml
```

Submit multiple evaluation configs:

```bash
bash scripts/sbatch_rby1_experiments.sh \
  configs/rby1_pick_pnp_binary.yaml \
  configs/rby1_pick_pnp_horizon450.yaml \
  configs/rby1_pick_pnp_horizon450_binary.yaml
```

## Output Organization

Baseline results:

```text
MolmoBot/MolmoBot/eval_output/baseline/
  rby1_eval/
  rby1_pick_pnp_expansion/
```

Experiment results:

```text
MolmoBot/MolmoBot/eval_output/experiments/
  rby1_pick_pnp_binary/
  rby1_pick_pnp_horizon450/
  rby1_pick_pnp_horizon450_binary/
  ...
```

Each run folder is named:

```text
YYYYMMDD_HHMMSS_SLURMJOBID
```

For example:

```text
20260606_223324_6251237
```

## Logs And Success Rates

Quick success/failure summaries come from `logs/`. Each run has:

```text
logs/rby1_multitask_<run_id>/
  summary.tsv
  pick_idx_<idx>.log
  pnp_idx_<idx>.log
  ...
```

`summary.tsv` records process success/failure. Benchmark success/failure is
inside each episode log, usually as either:

```text
completed with success=True
completed with success=False
```

or:

```text
...: pass
...: fail
```

`eval_output/` contains the richer artifacts: `run_info.json`, videos, HDF5
rollout data, and MolmoSpaces output folders.

## Current Result Summary

The released `MolmoBot-RBY1Multitask` checkpoint runs in the current framework,
but Pick/PnP reproduction remains far below the MolmoBot website numbers.

Baseline result summary:

- 30-index all-task baseline: `pick 0/30`, `pnp 1/30`, `opening 9/30`, `door_opening 15/30`
- Pick/PnP expansion: `pick 0/70`, `pnp 4/63` from available completed logs

Parameter sweeps on the shared 10 Pick/PnP indices did not materially improve
Pick/PnP. The only success in the sweep was `pnp idx 58` under inverted gripper
mapping.

## Maintained Code Changes

The active `rby1-custom` setup includes:

- RBY1-compatible MolmoBot multitask policy/eval fixes
- robust launcher argument filtering for MolmoSpaces API differences
- configurable RBY1 gripper mapping via environment variables
- YAML-driven Python orchestration through `scripts/rby1_runner.py`
- Slurm submission wrappers for repeatable evaluation jobs
- optional `terminate_upon_success` propagation through MolmoBot and MolmoSpaces

## Current Hypothesis

Simple parameter changes do not appear to explain the Pick/PnP gap. The next
debugging direction should focus on deeper alignment issues:

- action semantics and action scaling
- camera/prompt/state ordering expected by the checkpoint
- benchmark success predicates
- differences from the authors' exact evaluation stack
- trajectory-level inspection of failed `.h5` rollouts
