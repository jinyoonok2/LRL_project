# MolmoSpaces RBY1 Setup Notes

This documents the setup path used to run RBY1 door-opening simulation in `molmospaces`.

## 1. Work On A Branch

Use a branch in the `molmospaces` repo for local simulation fixes:

```bash
cd /home/jinyoon/workspace/live-robotics-lab-project/molmospaces
git checkout -b rby1-sim-bringup
```

If the branch already exists:

```bash
git checkout rby1-sim-bringup
```

## 2. Create The Conda Environment

```bash
cd /home/jinyoon/workspace/live-robotics-lab-project/molmospaces
conda create -n mlspaces python=3.11
conda activate mlspaces
```

Install base MolmoSpaces with MuJoCo support:

```bash
pip install -e ".[mujoco]"
```

If native package builds fail with `gcc: No such file or directory`, install build tools:

```bash
sudo apt update
sudo apt install -y build-essential python3-dev
```

## 3. Install MolmoSpaces Assets

The assets should be installed through the MolmoSpaces resource manager, not by manually downloading random files.

```bash
cd /home/jinyoon/workspace/live-robotics-lab-project/molmospaces
conda activate mlspaces

export PYTHONPATH="${PYTHONPATH}:."
export MLSPACES_FORCE_INSTALL=True

python -m molmo_spaces.molmo_spaces_constants
```

This populates cache/resource directories such as:

```bash
~/.cache/molmo-spaces-resources
~/.cache/molmospaces
```

If a previous interrupted install leaves broken cache state, fix ownership and reset caches:

```bash
sudo chown -R jinyoon:jinyoon ~/.cache/molmo-spaces-resources ~/.cache/molmospaces ~/nltk_data
rm -rf ~/.cache/molmo-spaces-resources ~/.cache/molmospaces
```

Then rerun:

```bash
python -m molmo_spaces.molmo_spaces_constants
```

## 4. Install RBY1/cuRobo Dependencies

RBY1 door-opening uses cuRobo, which is not included in `.[mujoco]`.

Install CUDA toolkit/build dependencies in the conda env:

```bash
conda activate mlspaces
conda install -c conda-forge cuda-toolkit=12.8 ninja evdev cuda-nvcc cuda-cudart-dev -n mlspaces
```

Install PyTorch matching CUDA 12.8:

```bash
pip install --force-reinstall "torch==2.7.1" "torchvision==0.22.1" \
  --index-url https://download.pytorch.org/whl/cu128
```

Verify PyTorch and `nvcc` match:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
PY
nvcc --version
```

Expected:

```text
2.7.1+cu128
12.8
nvcc ... release 12.8
```

Install cuRobo:

```bash
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$(dirname $(find $CONDA_PREFIX -name "cuda_runtime_api.h" | head -1)):$CPATH
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
export MAX_JOBS=2

pip install --no-build-isolation \
  "nvidia-curobo @ git+https://github.com/allenai/curobo.git@87e857d46fa5398f268c7f31d26566351be8671d"
```

If the build runs out of memory, retry with:

```bash
export MAX_JOBS=1
```

## 5. Verify NVIDIA Driver/CUDA Runtime

The conda CUDA toolkit is not enough. The host needs a working NVIDIA driver.

Check:

```bash
nvidia-smi
```

Then check inside the conda env:

```bash
conda activate mlspaces
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

Expected:

```text
cuda available: True
device count: 1
NVIDIA GeForce RTX 4060 ...
```

## 6. Run RBY1 Door Opening Debug

```bash
cd /home/jinyoon/workspace/live-robotics-lab-project/molmospaces
conda activate mlspaces

export PYTHONPATH="${PYTHONPATH}:."
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export JAX_PLATFORMS=cpu

python -m molmo_spaces.data_generation.main DoorOpeningDebugConfig
```

Successful run should end with:

```text
Completed 1 houses, skipped 0 houses
Success count: 1, Total count: 1
Success rate: 100.00%
```

## 7. Optional Teleop Testing

The baseline keyboard teleop path:

```bash
python scripts/datagen/run_pipeline.py --viewer --policy teleop --robot rum
```

A separate fast teleop sandbox was added on the local test branch:

```bash
python scripts/datagen/run_fast_teleop.py --robot rum
```

Useful tuning arguments:

```bash
python scripts/datagen/run_fast_teleop.py \
  --robot rum \
  --step_size 0.03 \
  --rot_step 0.08 \
  --img_width 320 \
  --img_height 240
```

## 8. MolmoBot-SPOC RBY1 Eval Setup

MolmoBot-SPOC uses MolmoSpaces for the simulator, robot, cameras, scenes, and benchmark episodes. The MolmoBot-SPOC side provides the learned policy/model and checkpoint loading.

Install the local SPOC package into the existing `mlspaces` environment:

```bash
cd /home/jinyoon/workspace/live-robotics-lab-project/MolmoBot/MolmoBot-SPOC
conda activate mlspaces
pip install -e .
```

If this changes MolmoSpaces-related dependency versions, restore the local MolmoSpaces install:

```bash
cd /home/jinyoon/workspace/live-robotics-lab-project/molmospaces
conda activate mlspaces
pip install -e .[mujoco] --no-deps
pip install "mujoco-mjx~=3.5.0" "molmospaces-resources==0.0.1b4"
```

Verify that Python imports the local editable repos:

```bash
python - <<'PY'
import molmo_spaces, molmobot_spoc, transformers
import mujoco.mjx
import molmospaces_resources

print("imports ok")
print("molmo_spaces:", molmo_spaces.__file__)
print("molmobot_spoc:", molmobot_spoc.__file__)
print("transformers:", transformers.__version__)
print("molmospaces_resources:", molmospaces_resources.__file__)
PY
```

Expected local paths should point into:

```text
/home/jinyoon/workspace/live-robotics-lab-project/molmospaces
/home/jinyoon/workspace/live-robotics-lab-project/MolmoBot/MolmoBot-SPOC
```

## 9. Hugging Face Model Download On This Ubuntu Setup

This machine had broken/slow IPv6 access to Hugging Face. `curl -4` worked, while `curl -6` timed out, and the normal `hf download ...` command could hang before creating the model cache folder. Use the IPv4-forced Python `snapshot_download(...)` method below for MolmoBot checkpoints instead of `hf download`.

First verify IPv4 access:

```bash
curl -4 -I https://huggingface.co
```

Then download checkpoints with IPv4-only DNS resolution:

```bash
cd /home/jinyoon/workspace/live-robotics-lab-project/MolmoBot/MolmoBot-SPOC
conda activate mlspaces

HF_HUB_DISABLE_XET=1 python - <<'PY'
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

for label, repo_id in models.items():
    print(f"Downloading {label}: {repo_id}")
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=f"/home/jinyoon/workspace/hf_models/{repo_id.split('/')[-1]}",
    )
    print(path)
PY
```

To download only one checkpoint, keep one entry in the `models` dictionary. The default Hugging Face cache also lives under:

```text
~/.cache/huggingface/hub
```

Local patch:

- `MolmoBot/MolmoBot-SPOC/eval/config/rby1_eval_config.py` now resolves the model with `snapshot_download(..., local_files_only=True)` first.
- If the checkpoint is already cached, eval uses the local snapshot path without contacting Hugging Face.
- If the checkpoint is missing, the config falls back to `snapshot_download(...)` with IPv4-only DNS resolution and `HF_HUB_DISABLE_XET=1`.

The eval also downloads the SigLIP image/text encoder weights:

```text
timm/ViT-B-16-SigLIP-256
```

Those are cached under `~/.cache/huggingface/hub/` as well.

## 10. Correct RBY1 Benchmark For MolmoBot-SPOC

Do not use the older RBY1 pick benchmark path for SPOC eval:

```text
~/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v1/20260408/procthor-objaverse/RBY1PickDataGenConfig/RBY1PickDataGenConfig_20260209_json_benchmark
```

That benchmark uses:

```text
camera_system_class: RBY1MjcfCameraSystem
img_resolution: [640, 480]
```

MolmoBot-SPOC's RBY1 image preprocessor expects the GoPro-style camera setup. Use the benchmark-v2 RBY1 pick benchmark instead:

```text
~/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260415/procthor-objaverse/rby1_benchmarks/pick_benchmark
```

Its metadata uses:

```text
camera_system_class: RBY1GoProD455CameraSystem
num_episodes: 2000
num_houses: 1715
```

Prepare the scene/object/grasp assets for the benchmark indices you want to test:

```bash
cd /home/jinyoon/workspace/live-robotics-lab-project/molmospaces
conda activate mlspaces
export PYTHONPATH="/home/jinyoon/workspace/live-robotics-lab-project/molmospaces:${PYTHONPATH}"

python scripts/benchmarks/prepare_benchmark_assets.py \
  --benchmark_dir /home/jinyoon/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260415/procthor-objaverse/rby1_benchmarks/pick_benchmark \
  --idx 0 1 2
```

What this does:

- Reads the selected benchmark episode indices.
- Resolves each episode's house scene XML.
- Installs only the missing `objects/objaverse` and `grasps/droid_objaverse` packages for those scenes.
- Skips packages that are already cached. It does not remove unrelated cached assets.

Run one prepared episode with the MolmoBot-SPOC RBY1 rigid policy:

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

Notes:

- `--idx 0` runs only the first benchmark episode.
- Change to `--idx 1`, `--idx 2`, etc. to run other fixed episodes.
- Remove `--idx` only when intentionally running the full benchmark.
- Default task length comes from the benchmark JSON. For the tested pick episodes, that was `task_horizon_sec=20`, which becomes `200` policy steps with `policy_dt_ms=100.0`.
- To test whether a failure is horizon-limited, add an explicit override such as `--task_horizon_sec 40`.

The benchmark may trigger extraction/checking for:

```text
objects/objaverse
grasps/droid_objaverse
```

`droid_objaverse` is the grasp-data asset family for Objaverse objects. It does not mean the robot changed to DROID/Franka; the robot remains RBY1.

## 11. MolmoBot RBY1 Multitask Eval

The RBY1 rigid SPOC checkpoint is known to be weak for these tasks. For stronger RBY1 model-policy tests, use the multitask checkpoint:

```text
/home/jinyoon/workspace/hf_models/MolmoBot-RBY1Multitask
```

The multitask model can run directly through MolmoSpaces benchmark eval using configs in:

```text
MolmoBot/MolmoBot/olmo/eval/configure_molmo_spaces.py
```

Use:

```text
MolmoBotRBY1PickPnPEvalConfig
```

for `pick_benchmark` and `pnp_benchmark`, and:

```text
MolmoBotRBY1DoorPlusOpenEvalConfig
```

for opening/door tasks.

Because `pip install -e ".[eval]"` previously failed on the old MolmoSpaces dependency URL, install the needed MolmoBot eval dependencies explicitly in the `mlspaces` environment:

```bash
cd /home/jinyoon/workspace/live-robotics-lab-project/MolmoBot/MolmoBot
conda activate mlspaces

pip install cached_path av hydra-core gcsfs==2023.9.2 accelerate sentencepiece google-cloud-storage
```

Then run a single pick episode:

```bash
cd /home/jinyoon/workspace/live-robotics-lab-project/MolmoBot/MolmoBot
conda activate mlspaces

export PYTHONPATH="/home/jinyoon/workspace/live-robotics-lab-project/MolmoBot/MolmoBot:/home/jinyoon/workspace/live-robotics-lab-project/molmospaces:${PYTHONPATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export JAX_PLATFORMS=cpu

python -m molmo_spaces.evaluation.eval_main \
  olmo.eval.configure_molmo_spaces:MolmoBotRBY1PickPnPEvalConfig \
  --benchmark_dir /home/jinyoon/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260415/procthor-objaverse/rby1_benchmarks/pick_benchmark \
  --checkpoint_path /home/jinyoon/workspace/hf_models/MolmoBot-RBY1Multitask \
  --no_wandb \
  --num_workers 1 \
  --idx 0
```

For PnP, use the same eval config with the PnP benchmark:

```bash
python -m molmo_spaces.evaluation.eval_main \
  olmo.eval.configure_molmo_spaces:MolmoBotRBY1PickPnPEvalConfig \
  --benchmark_dir /home/jinyoon/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260415/procthor-objaverse/rby1_benchmarks/pnp_benchmark \
  --checkpoint_path /home/jinyoon/workspace/hf_models/MolmoBot-RBY1Multitask \
  --no_wandb \
  --num_workers 1 \
  --idx 0
```

For opening tasks, use the door/open config:

```bash
python -m molmo_spaces.evaluation.eval_main \
  olmo.eval.configure_molmo_spaces:MolmoBotRBY1DoorPlusOpenEvalConfig \
  --benchmark_dir /home/jinyoon/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260415/ithor/rby1_benchmarks/opening_benchmark \
  --checkpoint_path /home/jinyoon/workspace/hf_models/MolmoBot-RBY1Multitask \
  --no_wandb \
  --num_workers 1 \
  --idx 0
```

Notes:

- `MolmoBot/MolmoBot/pyproject.toml` was patched to use modern direct URL syntax for the optional MolmoSpaces eval dependency.
- `MolmoBot/MolmoBot/olmo/eval/configure_molmo_spaces.py` was patched so the RBY1 multitask eval config matches the current MolmoSpaces policy API and treats older optional policy fields as optional.
- A local smoke test with `--idx 0` reached checkpoint config loading and model construction, then failed with CUDA OOM on the 8 GB RTX 4060 Laptop GPU. This means the code path is wired far enough to load the multitask model, but the local GPU is too small for this checkpoint. A larger GPU is preferred for serious benchmark runs.

## 12. Current Evaluation Findings

The current setup is able to launch RBY1 simulation, load benchmark-v2 pick episodes, run the provided MolmoBot rigid policy, and save MP4/H5 rollouts. However, the tested MolmoBot RBY1 rigid pick episodes were all marked unsuccessful by the benchmark success metric.

The current generated results kept locally are:

```text
MolmoBot/MolmoBot-SPOC/eval_output/RBY1RigidManipEvalConfig/
MolmoBot/MolmoBot-SPOC/eval_output/RBY1PlannerJsonEval/
```

Only output folders with generated MP4/H5 rollouts are kept. Failed attempts that produced only logs/configs were removed.

Planner-based MolmoSpaces diagnostics are separate from MolmoBot model evaluation. They use MolmoSpaces planner policies such as CuRobo, not the learned MolmoBot checkpoint. The planner outputs are currently stored beside the MolmoBot outputs only for convenience. The pick planner generated saved rollouts for `house_1001` and `house_1047`, but those also failed the official benchmark success metric.

Author-style CuRobo planner settings use more GPU memory than the local 8 GB RTX 4060 Laptop GPU can reliably provide. When using those settings locally, several planner runs hit CUDA out-of-memory. Use a larger GPU, ideally 16 GB VRAM or more, before drawing conclusions from author-style planner runs.

## 13. Planned Follow-Up

- Test `allenai/MolmoBot-RBY1Multitask` on `pick_benchmark`, `pnp_benchmark`, and opening tasks using the direct MolmoSpaces eval configs above.
- Download and test `allenai/MolmoBot-SPOC-RBY1Articulated` on `opening_benchmark`.
- Re-run planner baselines on a larger GPU with author-style batch settings.
- Keep the benchmark success metric unchanged when reporting results. If a planner visually grasps an object but reports `success=False`, treat that as a post-grasp/lift/success-condition issue rather than changing the metric.

