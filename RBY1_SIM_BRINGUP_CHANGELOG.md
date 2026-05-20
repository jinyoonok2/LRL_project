# RBY1 Sim Bringup Changelog

Branch: `rby1-sim-bringup`

This changelog summarizes local changes made while bringing up RBY1 simulation, teleop experiments, and MolmoBot-SPOC eval.

## Confirmed Working

- Created and used the `mlspaces` conda environment with Python 3.11.
- Installed `molmospaces` with MuJoCo support.
- Installed MolmoSpaces assets through `python -m molmo_spaces.molmo_spaces_constants`.
- Installed CUDA 12.8 PyTorch and cuRobo.
- Installed and verified NVIDIA driver/CUDA visibility.
- Installed local `MolmoBot-SPOC` into the `mlspaces` environment.
- Downloaded/cached the RBY1 rigid MolmoBot-SPOC checkpoint:

```text
allenai/MolmoBot-SPOC-RBY1Rigid
```

- Ran one benchmark-v2 RBY1 pick eval episode with MolmoBot-SPOC. The rollout completed and saved data, although the policy did not succeed on the selected episode:

```text
Completed 1 houses, skipped 0 houses
Success count: 0, Total count: 1
Success rate: 0.00%
```

- Prepared benchmark-v2 RBY1 pick assets for selected indices with `scripts/benchmarks/prepare_benchmark_assets.py`.
- Confirmed `--idx 2` runs through the simulator after asset preparation:

```text
Sampled task 'pick up the eye.'
Completed 1 houses, skipped 0 houses
Success count: 0, Total count: 1
```

- Ran `DoorOpeningDebugConfig` successfully:

```text
Completed 1 houses, skipped 0 houses
Success count: 1, Total count: 1
Success rate: 100.00%
```

## Code Changes

### `molmospaces/molmo_spaces/env/env.py`

Changed the classic MuJoCo renderer path to pass an explicit EGL device id:

```python
egl_device_id = int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0"))
self._renderer = MjOpenGLRenderer(
    model=self.mj_model,
    width=width,
    height=height,
    device_id=egl_device_id,
)
```

Reason:

- On Linux, the renderer was falling into MuJoCo's macOS CGL path and trying to load:

```text
/System/Library/Frameworks/OpenGL.framework/OpenGL
```

- Passing an EGL device id forces the Linux EGL rendering path.

### `molmospaces/molmo_spaces/utils/scene_maps.py`

Changed occupancy-map renderer helper to normalize `device_id=None` to `MUJOCO_EGL_DEVICE_ID`:

```python
if device_id is None:
    device_id = int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0"))
```

Reason:

- `ProcTHORMap.from_mj_model_path(...)` generated occupancy maps through a separate renderer path.
- That path still passed `device_id=None`, causing the same macOS CGL/OpenGL error during task sampling.

### `molmospaces/molmo_spaces/data_generation/config/door_opening_configs.py`

Added `DoorOpeningNoViewerDebugConfig`, which inherits from `DoorOpeningDebugConfig` and only disables the passive MuJoCo viewer:

```python
@register_config("DoorOpeningNoViewerDebugConfig")
class DoorOpeningNoViewerDebugConfig(DoorOpeningDebugConfig):
    use_passive_viewer: bool = False
```

Reason:

- Allows rerunning the same RBY1 door-opening debug scenario without opening the live MuJoCo viewer.
- Saved trajectories/videos still go under `experiment_output/door_opening_debug/...` when the rollout succeeds.

### `molmospaces/molmo_spaces/utils/synset_utils.py`

Changed NLTK setup to check for local WordNet corpora before calling `nltk.download(...)`:

```python
corpus_paths = {
    "wordnet": ["corpora/wordnet", "corpora/wordnet.zip"],
    "wordnet2022": ["corpora/wordnet2022", "corpora/wordnet2022.zip"],
}
```

Reason:

- The original code called `nltk.download("wordnet")` and `nltk.download("wordnet2022")` on every import.
- Even when data was already installed, this could contact the NLTK server and stall startup on slow/reset network connections.
- The new behavior avoids repeated online checks when the corpora already exist locally.

### `molmospaces/scripts/datagen/run_fast_teleop.py`

Added a separate fast teleop sandbox script instead of changing baseline teleop defaults.

Default settings:

```text
step_size=0.02
rot_step=0.06
img_resolution=480x360
```

Usage:

```bash
cd /home/jinyoon/workspace/live-robotics-lab-project/molmospaces
conda activate mlspaces
python scripts/datagen/run_fast_teleop.py --robot rum
```

Tuning example:

```bash
python scripts/datagen/run_fast_teleop.py \
  --robot rum \
  --step_size 0.03 \
  --rot_step 0.08 \
  --img_width 320 \
  --img_height 240
```

### `MolmoBot/MolmoBot-SPOC/eval/spoc_policy.py`

Adjusted `SPOCModelPolicy` to match the current local MolmoSpaces `InferencePolicy` constructor:

```python
# Current MolmoSpaces InferencePolicy only accepts the experiment config.
super().__init__(config)
```

Reason:

- MolmoBot-SPOC originally called `super().__init__(config, task_type)`.
- Current local MolmoSpaces defines `InferencePolicy.__init__(config)`.
- Without this patch, MolmoBot eval failed during policy creation:

```text
TypeError: InferencePolicy.__init__() takes 2 positional arguments but 3 were given
```

### `molmospaces/scripts/benchmarks/prepare_benchmark_assets.py`

Added a helper script for preparing only the scene/object/grasp assets needed by selected JSON benchmark episode indices:

```bash
python scripts/benchmarks/prepare_benchmark_assets.py \
  --benchmark_dir /home/jinyoon/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260415/procthor-objaverse/rby1_benchmarks/pick_benchmark \
  --idx 0 1 2
```

Reason:

- Full `objects/objaverse` is very large.
- Individual benchmark episodes can reference different ProcTHOR-Objaverse houses and therefore different Objaverse mesh packages.
- Preparing selected indices downloads/extracts only the required `objects/objaverse` and `grasps/droid_objaverse` packages, while skipping packages already cached.
- This fixed the earlier `idx 1` scene setup failure caused by a missing object mesh:

```text
468e585df1b04c859eb6972614a7d8a4_visual.obj
```

## MolmoBot-SPOC Eval Notes

### Hugging Face / IPv4

The RBY1 rigid checkpoint download initially hung in `snapshot_download(...)`.

Diagnosis:

- `curl -4 https://huggingface.co/...` worked.
- `curl -6 https://huggingface.co/...` timed out.
- This indicates broken/slow IPv6 connectivity on the current Ubuntu setup.

Working approach:

```bash
HF_HUB_DISABLE_XET=1 python - <<'PY'
import socket

orig_getaddrinfo = socket.getaddrinfo

def ipv4_getaddrinfo(*args, **kwargs):
    return [info for info in orig_getaddrinfo(*args, **kwargs) if info[0] == socket.AF_INET]

socket.getaddrinfo = ipv4_getaddrinfo

from huggingface_hub import snapshot_download
print(snapshot_download("allenai/MolmoBot-SPOC-RBY1Rigid"))
PY
```

The model cached to:

```text
~/.cache/huggingface/hub/models--allenai--MolmoBot-SPOC-RBY1Rigid/snapshots/01d1c5334e241c739099c2a043c5b93f87ee7eff
```

Follow-up patch in `MolmoBot/MolmoBot-SPOC/eval/config/rby1_eval_config.py`:

- RBY1 eval now tries `snapshot_download(..., local_files_only=True)` before any network request.
- If the snapshot is already cached, subsequent eval runs use the local checkpoint directly.
- If the snapshot is not cached, the download path forces IPv4 DNS resolution and sets `HF_HUB_DISABLE_XET=1`.

### Benchmark Selection

The first MolmoBot-SPOC eval attempt used the older benchmark-v1 RBY1 pick benchmark:

```text
molmospaces-bench-v1/20260408/procthor-objaverse/RBY1PickDataGenConfig/RBY1PickDataGenConfig_20260209_json_benchmark
```

That failed after model loading with:

```text
AssertionError: Image should be raw GoPro format, actually 480x640
```

Reason:

- The benchmark-v1 path uses `RBY1MjcfCameraSystem` with `img_resolution: [640, 480]`.
- MolmoBot-SPOC's RBY1 image preprocessor expects GoPro-style raw images (`768x576`) from the `RBY1GoProD455CameraSystem` setup.

Correct benchmark for the RBY1 rigid/pick SPOC eval:

```text
~/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260415/procthor-objaverse/rby1_benchmarks/pick_benchmark
```

Metadata:

```text
camera_system_class: RBY1GoProD455CameraSystem
num_episodes: 2000
num_houses: 1715
```

Working one-episode command:

```bash
python -m molmo_spaces.evaluation.eval_main \
  molmobot_spoc.eval.config.rby1_eval_config:RBY1RigidManipEvalConfig \
  --benchmark_dir /home/jinyoon/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260415/procthor-objaverse/rby1_benchmarks/pick_benchmark \
  --no_wandb \
  --num_workers 1 \
  --idx 0
```

Output kept:

```text
MolmoBot/MolmoBot-SPOC/eval_output/RBY1RigidManipEvalConfig/20260519_000548
```

Earlier intermediate MolmoBot eval output folders were removed after the final run was produced.

### Asset Extraction

MolmoBot-SPOC eval with RBY1 pick may require:

```text
objects/objaverse
grasps/droid_objaverse
```

Notes:

- `objects/objaverse` contains object meshes/assets used by the benchmark.
- `grasps/droid_objaverse` contains precomputed grasp metadata for Objaverse objects.
- The `droid_objaverse` name refers to the grasp asset family, not the active robot. The active robot remains RBY1.
- Assets are cached and should not be re-extracted for the same completed packages unless caches are deleted or versions change.

## Reverted Changes

### `molmospaces/molmo_spaces/configs/policy_configs_baselines.py`

Temporarily changed teleop defaults to faster values, then reverted them.

Current status:

- Baseline teleop config is back to the original values:

```python
step_size: float = 0.005
rot_step: float = 0.02
pos_sensitivity: float = 0.005
rot_sensitivity: float = 0.02
```

Reason:

- Faster values should be opt-in through `scripts/datagen/run_fast_teleop.py`, not global defaults.

### `molmospaces/molmo_spaces/data_generation/pipeline.py`

Temporarily added lazy policy initialization for `policy_config=None`, then reverted it.

Reason:

- The actual issue was missing NVIDIA driver visibility.
- After installing the NVIDIA driver, the normal policy initialization path worked.

## Important Notes

- The RBY1 door-opening path depends on cuRobo and therefore requires a working NVIDIA GPU driver.
- The conda CUDA toolkit alone is not enough; `nvidia-smi` and `torch.cuda.is_available()` must work.
- `experiment_output/` is ignored by git; deleting it removes only local generated run artifacts, not source-controlled repo files.
- Keyboard teleop through `run_pipeline.py` is slow because it runs inside the full data-generation loop, including sensors, rendering, task checks, and recording logic.
- `MolmoBot` is likely useful later for learned policy inference/training, but it does not directly solve MolmoSpaces keyboard teleop loop latency.

