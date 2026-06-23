# Live Robotics Lab RBY1 Evaluation Project

This repository is the working RBY1 MolmoBot/MolmoSpaces evaluation workspace.
The `main` branch is the current project branch for RBY1 multitask evaluation
fixes, scripts, configs, and experiment results.

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

## Branch

Use this branch for current work:

- `main`: active RBY1 evaluation branch

Old branch names such as `uva-cs-server`, `rby1-sim-bringup`, and
`rby1-policy-eval` are no longer part of the intended workflow. The previous
`rby1-custom` branch content has been promoted into `main`.

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

- `configs/rby1_eval.yaml`: 100 indices across `pick`, `pnp`, `opening`, and `door_opening`

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
