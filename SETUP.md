# RBY1 Setup And Slurm Guide

This file is the setup guide for running RBY1 MolmoBot/MolmoSpaces evaluation
on the UVA CS Slurm environment.

## Paths

Current project paths:

```text
Project: /u/xna8aw/workspace/live-robotics-lab/LRL_project
Cache:   /u/xna8aw/workspace/live-robotics-lab/cache
Conda:   /u/xna8aw/workspace/live-robotics-lab/envs/mlspaces
Models:  /u/xna8aw/workspace/live-robotics-lab/hf_models
```

Main checkpoint:

```text
/u/xna8aw/workspace/live-robotics-lab/hf_models/MolmoBot-RBY1Multitask
```

Main benchmark root:

```text
/u/xna8aw/workspace/live-robotics-lab/cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260415
```

## Branches

Use:

```bash
cd /u/xna8aw/workspace/live-robotics-lab/LRL_project
git checkout rby1-custom
git submodule update --init --recursive
```

Expected active branches:

```text
LRL_project: rby1-custom
MolmoBot:    rby1-custom
molmospaces: rby1-custom
```

Keep `main` clean. Use `rby1-custom` for development and experiments.

## Module And Conda Environment

Load cluster modules:

```bash
module load miniforge/25.3.1-py3.12 cuda/12.8.1 gcc/12.4.0
```

Create the environment once if it does not already exist:

```bash
conda create -y -p /u/xna8aw/workspace/live-robotics-lab/envs/mlspaces python=3.11
```

Activate:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /u/xna8aw/workspace/live-robotics-lab/envs/mlspaces
```

Install MolmoSpaces:

```bash
cd /u/xna8aw/workspace/live-robotics-lab/LRL_project/molmospaces
pip install --no-cache-dir -e ".[mujoco]"
```

Install MolmoBot eval-side dependencies:

```bash
cd /u/xna8aw/workspace/live-robotics-lab/LRL_project/MolmoBot/MolmoBot
pip install --no-cache-dir cached_path av hydra-core gcsfs==2023.9.2 accelerate sentencepiece google-cloud-storage
```

Install CUDA/PyTorch support if needed:

```bash
conda install -y -c conda-forge cuda-toolkit=12.8 ninja evdev cuda-nvcc cuda-cudart-dev
pip install --no-cache-dir --force-reinstall "torch==2.7.1" "torchvision==0.22.1" \
  --index-url https://download.pytorch.org/whl/cu128
```

Check CUDA visibility:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## Cache Environment

The Python runner sets these automatically for evaluation jobs, but they are
useful for manual debugging:

```bash
export PROJECT_ROOT=/u/xna8aw/workspace/live-robotics-lab/LRL_project
export CACHE_ROOT=/u/xna8aw/workspace/live-robotics-lab/cache

export MLSPACES_CACHE_DIR="$CACHE_ROOT/molmo-spaces-resources"
export MLSPACES_ASSETS_DIR="$CACHE_ROOT/molmospaces/assets"
export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/transformers"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

export PYTHONPATH="$PROJECT_ROOT/MolmoBot/MolmoBot:$PROJECT_ROOT/molmospaces:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export JAX_PLATFORMS=cpu
export HF_HUB_DISABLE_XET=1
```

## Install Resource Metadata

Run once after installation:

```bash
cd /u/xna8aw/workspace/live-robotics-lab/LRL_project/molmospaces
python -m molmo_spaces.molmo_spaces_constants
```

## Prepare Benchmark Assets

Use the Slurm wrapper:

```bash
cd /u/xna8aw/workspace/live-robotics-lab/LRL_project
bash scripts/sbatch_rby1_prepare.sh configs/rby1_eval.yaml
```

Prepare assets for a specific experiment config:

```bash
bash scripts/sbatch_rby1_prepare.sh configs/rby1_pick_pnp_binary.yaml
```

Dry-run locally without launching preparation:

```bash
python scripts/rby1_runner.py prepare \
  --config configs/rby1_pick_pnp_binary.yaml \
  --dry-run
```

## Run Evaluations

Submit one YAML config:

```bash
bash scripts/sbatch_rby1_eval.sh configs/rby1_pick_pnp_binary.yaml
```

Submit several configs:

```bash
bash scripts/sbatch_rby1_experiments.sh \
  configs/rby1_pick_pnp_binary.yaml \
  configs/rby1_pick_pnp_horizon450.yaml \
  configs/rby1_pick_pnp_horizon450_binary.yaml
```

Dry-run one episode:

```bash
python scripts/rby1_runner.py eval \
  --config configs/rby1_pick_pnp_binary.yaml \
  --tasks pick \
  --indices 10 \
  --dry-run
```

## Slurm Defaults

`scripts/sbatch_rby1_eval.sh` defaults:

```text
partition: gpu
account:   xna8aw_base
gpu:       --gres=gpu:1
cpus:      8
memory:    96G
time:      03:00:00
```

Override without editing the script:

```bash
RBY1_SLURM_TIME=04:00:00 \
RBY1_SLURM_MEM=128G \
bash scripts/sbatch_rby1_eval.sh configs/rby1_eval.yaml
```

`scripts/sbatch_rby1_prepare.sh` defaults:

```text
partition: gpu
account:   xna8aw_base
gpu:       none by default
cpus:      4
memory:    32G
time:      02:00:00
```

Override preparation resources:

```bash
RBY1_PREPARE_TIME=04:00:00 \
RBY1_PREPARE_MEM=64G \
bash scripts/sbatch_rby1_prepare.sh configs/rby1_eval.yaml
```

## Check Jobs

Queue:

```bash
squeue -u "$USER"
```

Accounting:

```bash
sacct -j <job_id> --format=JobID,JobName%45,State,Elapsed,ExitCode,NodeList
```

Slurm logs:

```text
slurm_logs/<job-name>-<job-id>.out
slurm_logs/<job-name>-<job-id>.err
```

Per-episode logs:

```text
logs/rby1_multitask_<run_id>/
```

## Output Layout

Baseline outputs:

```text
MolmoBot/MolmoBot/eval_output/baseline/
  rby1_eval/
```

Experiment outputs:

```text
MolmoBot/MolmoBot/eval_output/experiments/
  rby1_pick_pnp_binary/
  rby1_pick_pnp_horizon450/
  rby1_pick_pnp_horizon450_binary/
  ...
```

Each run uses:

```text
YYYYMMDD_HHMMSS_SLURMJOBID
```

Each episode folder contains `run_info.json` plus MolmoSpaces/MolmoBot output
artifacts such as videos and HDF5 rollout files.

## Validate Code Changes

Before committing:

```bash
cd /u/xna8aw/workspace/live-robotics-lab/LRL_project
bash -n scripts/sbatch_rby1_eval.sh scripts/sbatch_rby1_prepare.sh scripts/sbatch_rby1_experiments.sh
python -m py_compile scripts/rby1_runner.py
python scripts/rby1_runner.py describe --config configs/rby1_eval.yaml
python scripts/rby1_runner.py eval --config configs/rby1_pick_pnp_binary.yaml --tasks pick --indices 10 --dry-run
```

## Troubleshooting

If Conda activation fails with an unbound variable such as:

```text
NVCC_PREPEND_FLAGS: unbound variable
```

make sure Slurm scripts disable `nounset` around activation:

```bash
set +u
module load miniforge/25.3.1-py3.12 cuda/12.8.1 gcc/12.4.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /u/xna8aw/workspace/live-robotics-lab/envs/mlspaces
set -u
```

If an evaluation job exits successfully but benchmark success is low, inspect:

```text
logs/rby1_multitask_<run_id>/*.log
MolmoBot/MolmoBot/eval_output/.../run_info.json
MolmoBot/MolmoBot/eval_output/.../*.mp4
MolmoBot/MolmoBot/eval_output/.../*.h5
```

Low Pick/PnP success is currently an open reproduction/alignment issue, not a
simple Slurm setup failure.
