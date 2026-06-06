# UVA CS Slurm Setup Notes

This is the UVA CS server version of `MOLMOSPACES_RBY1_SETUP.md`. The local setup notes are still useful for package details, but Slurm runs should avoid workstation assumptions such as `sudo`, direct long-running shell commands, local GPU names, and large caches under the default home directory.

## 1. Repository Layout

The project is expected at:

```bash
cd /u/xna8aw/workspace/live-robotics-lab/live-robotics-lab-project
```

Both submodules should be present and on the `uva-cs-server` branch:

```bash
git submodule update --init --recursive
git submodule foreach 'git checkout uva-cs-server && git pull --ff-only origin uva-cs-server'
```

Verify:

```bash
git submodule foreach 'printf "%s " "$name"; git branch --show-current'
```

Expected:

```text
MolmoBot uva-cs-server
molmospaces uva-cs-server
```

## 2. Use Slurm For GPU Work

Do not run GPU simulation or evaluation directly on the login node. Use either an interactive allocation for setup/debugging or `sbatch` for repeatable runs.

First check the local cluster options:

```bash
sinfo
sacctmgr show assoc user=$USER format=Account,Partition,QOS
```

The examples below use placeholders:

```text
<ACCOUNT>       Slurm account, if required
<PARTITION>     GPU partition name
<TIME>          Wall time, for example 02:00:00
```

If the cluster does not require `--account`, remove that line from the scripts.

## 3. Environment Setup

Load the cluster Python/Conda stack and CUDA/compiler modules before setup commands:

```bash
module load miniforge/25.3.1-py3.12 cuda/12.8.1 gcc/12.4.0
```

The environment used on this server is project-local:

```bash
/u/xna8aw/workspace/live-robotics-lab/envs/mlspaces
```

Create the environment once, preferably from an interactive CPU or GPU allocation:

```bash
module load miniforge/25.3.1-py3.12 cuda/12.8.1 gcc/12.4.0
conda create -y -p /u/xna8aw/workspace/live-robotics-lab/envs/mlspaces python=3.11

conda activate /u/xna8aw/workspace/live-robotics-lab/envs/mlspaces

cd /u/xna8aw/workspace/live-robotics-lab/live-robotics-lab-project/molmospaces
pip install --no-cache-dir -e ".[mujoco]"
```

Use `--no-cache-dir` for large pip installs so downloaded wheels do not refill `~/.cache/pip`.

Do not use `sudo apt install` on the cluster. If native builds fail because a compiler, CUDA toolkit, or build dependency is missing, prefer cluster modules, Conda packages, or ask the system administrators.

## 4. Cache Locations

MolmoSpaces assets and Hugging Face models can be large. Prefer workspace, scratch, or project storage instead of filling the home directory.

Example:

```bash
export PROJECT_ROOT=/u/xna8aw/workspace/live-robotics-lab/live-robotics-lab-project
export CACHE_ROOT=/u/xna8aw/workspace/live-robotics-lab/cache

mkdir -p "$CACHE_ROOT"/{molmo-spaces-resources,molmospaces,huggingface,torch,xdg}

export MOLMO_SPACES_RESOURCES_CACHE="$CACHE_ROOT/molmo-spaces-resources"
export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/transformers"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
```

Keep these exports in every setup and run script so all jobs see the same assets and model cache.

## 5. Install Assets

Run this once after the Python environment is installed:

```bash
cd /u/xna8aw/workspace/live-robotics-lab/live-robotics-lab-project/molmospaces
module load miniforge/25.3.1-py3.12 cuda/12.8.1 gcc/12.4.0
conda activate /u/xna8aw/workspace/live-robotics-lab/envs/mlspaces

export PROJECT_ROOT=/u/xna8aw/workspace/live-robotics-lab/live-robotics-lab-project
export CACHE_ROOT=/u/xna8aw/workspace/live-robotics-lab/cache
export MOLMO_SPACES_RESOURCES_CACHE="$CACHE_ROOT/molmo-spaces-resources"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

export PYTHONPATH="$PROJECT_ROOT/molmospaces:${PYTHONPATH:-}"
export MLSPACES_FORCE_INSTALL=True

python -m molmo_spaces.molmo_spaces_constants
```

If the install is slow or needs GPU-node networking, run it through an `sbatch` setup job rather than on the login node.

## 6. CUDA, PyTorch, And cuRobo

The server setup uses CUDA 12.8 with PyTorch 2.7.1:

```bash
module load miniforge/25.3.1-py3.12 cuda/12.8.1 gcc/12.4.0
conda activate /u/xna8aw/workspace/live-robotics-lab/envs/mlspaces

conda install -c conda-forge cuda-toolkit=12.8 ninja evdev cuda-nvcc cuda-cudart-dev

pip install --no-cache-dir --force-reinstall "torch==2.7.1" "torchvision==0.22.1" \
  --index-url https://download.pytorch.org/whl/cu128

pip install --no-cache-dir "fsspec[http]<=2026.2.0,>=2023.1.0" "pillow<12.0,>=9.2.0"
```

Verify inside a GPU allocation:

```bash
nvidia-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
nvcc --version
```

Then install cuRobo:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$(dirname "$(find "$CONDA_PREFIX" -name cuda_runtime_api.h | head -1)"):${CPATH}"
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
export MAX_JOBS=2

pip install --no-cache-dir --no-build-isolation \
  "nvidia-curobo @ git+https://github.com/allenai/curobo.git@87e857d46fa5398f268c7f31d26566351be8671d"
```

If the build runs out of memory, retry with:

```bash
export MAX_JOBS=1
```

## 7. Common Slurm Job Header

Use this as the base for GPU jobs:

Create the log directory before submitting any job. Slurm opens the output and error files before the script body runs.

```bash
mkdir -p slurm_logs
```

```bash
#!/bin/bash
#SBATCH --job-name=rby1-test
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=<TIME>

set -euo pipefail

cd /u/xna8aw/workspace/live-robotics-lab/live-robotics-lab-project

# Pick the activation method that works on the server.
module load miniforge/25.3.1-py3.12 cuda/12.8.1 gcc/12.4.0
conda activate /u/xna8aw/workspace/live-robotics-lab/envs/mlspaces

export PROJECT_ROOT=/u/xna8aw/workspace/live-robotics-lab/live-robotics-lab-project
export CACHE_ROOT=/u/xna8aw/workspace/live-robotics-lab/cache
mkdir -p "$CACHE_ROOT"/{molmo-spaces-resources,molmospaces,huggingface,torch,xdg}

export MOLMO_SPACES_RESOURCES_CACHE="$CACHE_ROOT/molmo-spaces-resources"
export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/transformers"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

export PYTHONPATH="$PROJECT_ROOT/molmospaces:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export JAX_PLATFORMS=cpu
export HF_HUB_DISABLE_XET=1
```

Remove or edit `--account` if the partition does not require it.

## 8. GPU Smoke Test Job

Create `slurm_gpu_smoke.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=rby1-smoke
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -euo pipefail

cd /u/xna8aw/workspace/live-robotics-lab/live-robotics-lab-project
module load miniforge/25.3.1-py3.12 cuda/12.8.1 gcc/12.4.0
conda activate /u/xna8aw/workspace/live-robotics-lab/envs/mlspaces

nvidia-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

Submit:

```bash
mkdir -p slurm_logs
sbatch slurm_gpu_smoke.sh
```

## 9. Door-Opening Debug Job

Create `slurm_door_opening_debug.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=rby1-door-debug
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00

set -euo pipefail

cd /u/xna8aw/workspace/live-robotics-lab/live-robotics-lab-project
module load miniforge/25.3.1-py3.12 cuda/12.8.1 gcc/12.4.0
conda activate /u/xna8aw/workspace/live-robotics-lab/envs/mlspaces

export PROJECT_ROOT=/u/xna8aw/workspace/live-robotics-lab/live-robotics-lab-project
export CACHE_ROOT=/u/xna8aw/workspace/live-robotics-lab/cache
export MOLMO_SPACES_RESOURCES_CACHE="$CACHE_ROOT/molmo-spaces-resources"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

export PYTHONPATH="$PROJECT_ROOT/molmospaces:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export JAX_PLATFORMS=cpu

cd "$PROJECT_ROOT/molmospaces"
python -m molmo_spaces.data_generation.main DoorOpeningNoViewerDebugConfig
```

Submit:

```bash
mkdir -p slurm_logs
sbatch slurm_door_opening_debug.sh
```

## 10. MolmoBot-SPOC Eval Job

Install the SPOC package into the same environment once:

```bash
cd /u/xna8aw/workspace/live-robotics-lab/live-robotics-lab-project/MolmoBot/MolmoBot-SPOC
module load miniforge/25.3.1-py3.12 cuda/12.8.1 gcc/12.4.0
conda activate /u/xna8aw/workspace/live-robotics-lab/envs/mlspaces
pip install --no-cache-dir -e .

cd /u/xna8aw/workspace/live-robotics-lab/live-robotics-lab-project/molmospaces
pip install --no-cache-dir -e ".[mujoco]" --no-deps
pip install --no-cache-dir "mujoco-mjx~=3.5.0" "molmospaces-resources==0.0.1b4"
```

Prepare benchmark assets for the indices you plan to run:

```bash
cd /u/xna8aw/workspace/live-robotics-lab/live-robotics-lab-project/molmospaces
module load miniforge/25.3.1-py3.12 cuda/12.8.1 gcc/12.4.0
conda activate /u/xna8aw/workspace/live-robotics-lab/envs/mlspaces

export PROJECT_ROOT=/u/xna8aw/workspace/live-robotics-lab/live-robotics-lab-project
export CACHE_ROOT=/u/xna8aw/workspace/live-robotics-lab/cache
export MOLMO_SPACES_RESOURCES_CACHE="$CACHE_ROOT/molmo-spaces-resources"
export PYTHONPATH="$PROJECT_ROOT/molmospaces:${PYTHONPATH:-}"

python scripts/benchmarks/prepare_benchmark_assets.py \
  --benchmark_dir "$MOLMO_SPACES_RESOURCES_CACHE/benchmarks/molmospaces-bench-v2/20260415/procthor-objaverse/rby1_benchmarks/pick_benchmark" \
  --idx 0 1 2
```

Create `slurm_rby1_eval.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=rby1-eval
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

IDX="${1:-0}"

cd /u/xna8aw/workspace/live-robotics-lab/live-robotics-lab-project
module load miniforge/25.3.1-py3.12 cuda/12.8.1 gcc/12.4.0
conda activate /u/xna8aw/workspace/live-robotics-lab/envs/mlspaces

export PROJECT_ROOT=/u/xna8aw/workspace/live-robotics-lab/live-robotics-lab-project
export CACHE_ROOT=/u/xna8aw/workspace/live-robotics-lab/cache
export MOLMO_SPACES_RESOURCES_CACHE="$CACHE_ROOT/molmo-spaces-resources"
export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/transformers"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

export PYTHONPATH="$PROJECT_ROOT/molmospaces:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export JAX_PLATFORMS=cpu
export HF_HUB_DISABLE_XET=1

cd "$PROJECT_ROOT/MolmoBot/MolmoBot-SPOC"

python -m molmo_spaces.evaluation.eval_main \
  molmobot_spoc.eval.config.rby1_eval_config:RBY1RigidManipEvalConfig \
  --benchmark_dir "$MOLMO_SPACES_RESOURCES_CACHE/benchmarks/molmospaces-bench-v2/20260415/procthor-objaverse/rby1_benchmarks/pick_benchmark" \
  --no_wandb \
  --num_workers 1 \
  --idx "$IDX"
```

Submit one episode:

```bash
mkdir -p slurm_logs
sbatch slurm_rby1_eval.sh 2
```

## 11. Hugging Face Checkpoints

Download model checkpoints once into project storage. The UVA branch notes include three useful RBY1 checkpoints:

```text
allenai/MolmoBot-SPOC-RBY1Rigid
allenai/MolmoBot-SPOC-RBY1Articulated
allenai/MolmoBot-RBY1Multitask
```

If normal Hugging Face downloads hang because of IPv6, force IPv4 from Python:

```bash
module load miniforge/25.3.1-py3.12 cuda/12.8.1 gcc/12.4.0
conda activate /u/xna8aw/workspace/live-robotics-lab/envs/mlspaces

export CACHE_ROOT=/u/xna8aw/workspace/live-robotics-lab/cache
export MODEL_ROOT=/u/xna8aw/workspace/live-robotics-lab/hf_models
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_DISABLE_XET=1
mkdir -p "$MODEL_ROOT"

python - <<'PY'
import os
import socket
from huggingface_hub import snapshot_download

orig_getaddrinfo = socket.getaddrinfo

def ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

socket.getaddrinfo = ipv4_getaddrinfo

models = {
    "rigid": "allenai/MolmoBot-SPOC-RBY1Rigid",
    "articulated": "allenai/MolmoBot-SPOC-RBY1Articulated",
    "multitask": "allenai/MolmoBot-RBY1Multitask",
}

model_root = os.environ["MODEL_ROOT"]

for label, repo_id in models.items():
    print(f"Downloading {label}: {repo_id}")
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=f"{model_root}/{repo_id.split('/')[-1]}",
    )
    print(path)
PY
```

To download only one checkpoint, keep one entry in the `models` dictionary.

## 12. MolmoBot Multitask Eval Job

The `uva-cs-server` branch also documents the stronger RBY1 multitask checkpoint:

```text
allenai/MolmoBot-RBY1Multitask
```

Install the MolmoBot eval-side dependencies in the same `mlspaces` environment:

```bash
module load miniforge/25.3.1-py3.12 cuda/12.8.1 gcc/12.4.0
conda activate /u/xna8aw/workspace/live-robotics-lab/envs/mlspaces

cd /u/xna8aw/workspace/live-robotics-lab/live-robotics-lab-project/MolmoBot/MolmoBot
pip install --no-cache-dir cached_path av hydra-core gcsfs==2023.9.2 accelerate sentencepiece google-cloud-storage
```

Create `slurm_rby1_multitask_eval.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=rby1-multitask
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00

set -euo pipefail

TASK="${1:-pick}"
IDX="${2:-0}"

cd /u/xna8aw/workspace/live-robotics-lab/live-robotics-lab-project
module load miniforge/25.3.1-py3.12 cuda/12.8.1 gcc/12.4.0
conda activate /u/xna8aw/workspace/live-robotics-lab/envs/mlspaces

export PROJECT_ROOT=/u/xna8aw/workspace/live-robotics-lab/live-robotics-lab-project
export CACHE_ROOT=/u/xna8aw/workspace/live-robotics-lab/cache
export MODEL_ROOT=/u/xna8aw/workspace/live-robotics-lab/hf_models
export MOLMO_SPACES_RESOURCES_CACHE="$CACHE_ROOT/molmo-spaces-resources"
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

case "$TASK" in
  pick)
    CONFIG="olmo.eval.configure_molmo_spaces:MolmoBotRBY1PickPnPEvalConfig"
    BENCHMARK="$MOLMO_SPACES_RESOURCES_CACHE/benchmarks/molmospaces-bench-v2/20260415/procthor-objaverse/rby1_benchmarks/pick_benchmark"
    ;;
  pnp)
    CONFIG="olmo.eval.configure_molmo_spaces:MolmoBotRBY1PickPnPEvalConfig"
    BENCHMARK="$MOLMO_SPACES_RESOURCES_CACHE/benchmarks/molmospaces-bench-v2/20260415/procthor-objaverse/rby1_benchmarks/pnp_benchmark"
    ;;
  opening)
    CONFIG="olmo.eval.configure_molmo_spaces:MolmoBotRBY1DoorPlusOpenEvalConfig"
    BENCHMARK="$MOLMO_SPACES_RESOURCES_CACHE/benchmarks/molmospaces-bench-v2/20260415/ithor/rby1_benchmarks/opening_benchmark"
    ;;
  *)
    echo "Unknown task: $TASK. Use pick, pnp, or opening." >&2
    exit 2
    ;;
esac

cd "$PROJECT_ROOT/MolmoBot/MolmoBot"

python -m molmo_spaces.evaluation.eval_main \
  "$CONFIG" \
  --benchmark_dir "$BENCHMARK" \
  --checkpoint_path "$MODEL_ROOT/MolmoBot-RBY1Multitask" \
  --no_wandb \
  --num_workers 1 \
  --idx "$IDX"
```

Submit one multitask episode:

```bash
mkdir -p slurm_logs
sbatch slurm_rby1_multitask_eval.sh pick 0
```

Use `pnp` or `opening` as the first argument for the other benchmark families. Start with one index before using arrays because this checkpoint is much larger than the rigid SPOC checkpoint and can require substantially more VRAM.

## 13. Optional Array Eval

For multiple prepared benchmark indices, create `slurm_rby1_eval_array.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=rby1-eval-array
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2

set -euo pipefail

bash slurm_rby1_eval.sh "$SLURM_ARRAY_TASK_ID"
```

Submit:

```bash
mkdir -p slurm_logs
sbatch slurm_rby1_eval_array.sh
```

Only use an array after the single-index eval works.

## 14. Reading Results

Check Slurm status:

```bash
squeue -u "$USER"
sacct -j <JOB_ID> --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

Logs are written under:

```text
slurm_logs/
```

MolmoBot-SPOC eval outputs are written under:

```text
MolmoBot/MolmoBot-SPOC/eval_output/RBY1RigidManipEvalConfig/<timestamp>/
```

MolmoBot multitask eval outputs are written under the output directory configured by the MolmoBot/MolmoSpaces eval config.

`success=False` is not necessarily a setup failure. It can mean the episode ran but the policy did not satisfy the benchmark success condition before the task horizon. Setup failures usually show import errors, CUDA errors, missing assets, skipped houses, or `Total count: 0`.

## 15. Differences From Local Setup

- Use Slurm for GPU execution instead of running long commands on the login node.
- Replace `sudo apt install` with modules, Conda packages, or admin-installed dependencies.
- Keep large MolmoSpaces and Hugging Face caches outside default home cache paths.
- Prefer headless configs and EGL rendering; do not rely on desktop viewers.
- Start with one benchmark index before using job arrays.
- Keep `MOLMOSPACES_RBY1_SETUP.md` as the local workstation record and this file as the cluster workflow.
