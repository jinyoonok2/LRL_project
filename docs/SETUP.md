# Live Robotics Lab — Setup and Compatibility Fixes

Verified locally: 2026-10-03. Start with [PROJECT_ENVIRONMENT.md](PROJECT_ENVIRONMENT.md) for storage, current branches, and environment ownership.

## 1. Choose the patched source versions

For this project's working source stack, use the forked submodule commits:

```text
MolmoBot:   jinyoonok2/MolmoBot
            a190df0bc631cd0cfce4544e3d30f7910337e113
MolmoSpaces: jinyoonok2/molmospaces
            a7bcb57ad7be3472df16a68dd401b930e69b19a8
```

Both pins include documentation-only updates after the working implementation commits (MolmoBot `9c2ebfa`, MolmoSpaces `fabf2af`). Both forks use `rby1-compatible`. Their self-contained records provide baseline commits, retained fixes, exact source reconstruction, paired installation, and benchmark verification:

- [MolmoBot changes and reproduction](../MolmoBot/docs/FORK_CHANGES.md)
- [MolmoSpaces changes and reproduction](../molmospaces/docs/FORK_CHANGES.md)

In a fresh clone of `https://github.com/jinyoonok2/LRL_project.git`, initialize submodules and inspect the recorded revisions:

```bash
git submodule update --init --recursive
git submodule status
git -C MolmoBot rev-parse HEAD
git -C molmospaces rev-parse HEAD
```

If reproducing this dated snapshot in a clean clone, explicitly select the two commits above. Do not switch or reset the existing dirty workspace to follow this recipe.

```bash
git -C MolmoBot checkout a190df0bc631cd0cfce4544e3d30f7910337e113
git -C molmospaces checkout a7bcb57ad7be3472df16a68dd401b930e69b19a8
```

For a reconstruction from original upstream source, MolmoBot's `MolmoBot/pyproject.toml` pins MolmoSpaces to `cd23becebcf72dd93a4aa5872a60802d5eff03ef`. Start from that MolmoSpaces base and port the compatibility fixes below. Installing that upstream pin alone omits our patches. Using arbitrary latest versions has not been validated here. The exact patched fork commits are the preferred reference.

## 2. Reuse the installed environment

Run from `/u/xna8aw/workspace/live-robotics-lab/LRL_project`:

```bash
module load miniforge/25.3.1-py3.12 cuda/12.8.1 gcc/12.4.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /u/xna8aw/workspace/live-robotics-lab/envs/mlspaces
```

Batch scripts using `set -u` should temporarily disable it around module/Conda activation, as the existing wrappers do. Installed Python/PyTorch versions and cuRobo provenance are recorded in the environment guide. Python 3.11 and the CUDA 12.8 PyTorch build are the current stack.

On a fresh machine, create a separate Python 3.11 environment. Install MolmoSpaces with its MuJoCo extra and MolmoBot using their package requirements, then install the local patched MolmoSpaces last:

```bash
python -m pip install -e './molmospaces[mujoco]'
python -m pip install -e './MolmoBot/MolmoBot[eval,train]'
python -m pip install -e './molmospaces[mujoco]'
python -m pip check
```

MolmoBot declares a Git dependency on upstream MolmoSpaces, so installation can replace the patched package. Reinstalling the local checkout and checking import paths is essential. These commands describe package wiring; they are not a tested full environment lock. Review CUDA/PyTorch and cuRobo requirements before installing into a new environment. Do not rebuild the working environment merely to follow this document.

## 3. Configure local imports and rendering

The runner sets these for jobs. For manual diagnostics:

```bash
export PROJECT_ROOT=/u/xna8aw/workspace/live-robotics-lab/LRL_project
export CACHE_ROOT=/u/xna8aw/workspace/live-robotics-lab/cache
export MLSPACES_CACHE_DIR="$CACHE_ROOT/molmo-spaces-resources"
export MLSPACES_ASSETS_DIR="$CACHE_ROOT/molmospaces/assets"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/MolmoBot/MolmoBot:$PROJECT_ROOT/molmospaces:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export JAX_PLATFORMS=cpu
export HF_HUB_DISABLE_XET=1
```

Check imports before running evaluation:

```bash
python -c 'import sys, molmo_spaces, olmo; print(sys.executable); print(molmo_spaces.__file__); print(olmo.__file__)'
```

Both packages should resolve to the intended local checkouts. Check GPU availability inside a Slurm GPU allocation.

## 4. Select benchmark resources deliberately

The historical public-reproduction fix uses `molmospaces-bench-v2/20260327`. The pinned upstream `20260325_1` release was documented as lacking the required RB-Y1 benchmark directories; `20260415` was documented as unpublished when checked. Those are historical observations, not current availability checks.

| Task | Relative benchmark directory |
|---|---|
| Pick | `procthor-objaverse/rby1_benchmarks/pick_benchmark` |
| Pick-and-Place | `procthor-objaverse/rby1_benchmarks/pnp_benchmark` |
| Opening | `ithor/rby1_benchmarks/opening_benchmark` |
| Door Opening | `procthor-10k/rby1_benchmarks/door_opening_benchmark` |

There is a current configuration discrepancy: `configs/rby1_eval.yaml` names `20260325_1`, while `scripts/rby1_runner.py` overrides Opening to `20260327`. Other tasks inherit the YAML version. For a fresh public reproduction, create a separate YAML config with both `benchmark_version` and `benchmark_root` set to `20260327`, and verify every resulting path. Do not assume the current local YAML implements the historical recommendation for all tasks. Do not silently retarget existing resources.

Use the runner's `describe` and `--dry-run` output to check resolved paths before preparing assets or submitting jobs.

## 5. Compatibility patches that made the framework work

Paths in this section are relative to the named submodule. The current patched commits already contain these changes; do not apply them twice.

### MolmoSpaces: benchmark loading and correct contact scoring

Commit `9b5f4e1` (`Fix RB-Y1 benchmark paths and contact checks`):

| File | Change and reason |
|---|---|
| `molmo_spaces/robots/robot_views/rby1_view.py` | Add `.id` to gripper EE and base body lookups so `root_body_id` is an integer, not a MuJoCo body view. |
| `molmo_spaces/tasks/pick_task.py` | Map the nested RB-Y1 base ID through `data.model.body_rootid[int(base.root_body_id)]` before comparing robot/object contacts. |
| `molmo_spaces/tasks/pick_and_place_task.py` | Use the same kinematic-tree-root mapping for release/contact checks. |
| `molmo_spaces/tasks/json_eval_task_sampler.py` | Normalize legacy `mujoco_thor.tasks.*` class paths to `molmo_spaces.tasks.*` for importing and inferring task types. |
| `molmo_spaces/evaluation/benchmark_schema.py` | Accept legacy single-joint start/goal positions encoded as `[value]` by unwrapping them to scalars. |

The body-ID and contact-root changes correspond to the fixes recorded from upstream PR #135. Both are needed: `.id` alone does not fix comparisons against the wrong root. The upstream references recorded during debugging are MolmoBot issue `allenai/MolmoBot#9`, MolmoSpaces PR `allenai/molmospaces#135`, body-ID commit `8797f8818086174509d3a9c288eccbaf713ad532`, and contact-root commit `6ebeeb8a38a6899b583e98c5c59cb25ccab76efe`. The benchmark-version bump was associated with upstream commit `e1a2fe16`. These are historical provenance references; this guide was verified against local source rather than current upstream pages.

### MolmoBot: launcher and RB-Y1 policy integration

| File | Maintained changes |
|---|---|
| `MolmoBot/launch_scripts/run_eval.py` | Filter evaluation keyword arguments against the installed `run_evaluation` signature; add single-episode selection; preserve the config's horizon when no override is supplied; expose terminate-on-success. |
| `MolmoBot/olmo/eval/configure_molmo_spaces.py` | Tolerate optional config fields; handle RB-Y1 arm groups in action limiting; provide multitask integration and action diagnostics; support seeded action generation; pass `task_type` to the policy base expected by the pinned MolmoSpaces API. |
| `MolmoBot-SPOC/eval/config/rby1_eval_config.py` | Supporting offline RB-Y1 SPOC evaluation robustness. This is a separate model path from the multitask checkpoint. |
| `MolmoBot-SPOC/eval/spoc_policy.py` | Supporting SPOC policy compatibility. |

Relevant history: `40df752` (offline SPOC), `25e41bf` (multitask/launcher compatibility), `6265178` (restored terminate-on-success), `20d2296` (deterministic action sampling), `b84a58f` (restored original raw gripper outputs), and `9c2ebfa` (pass task type to the policy base).

Earlier binary/inverted gripper mappings were experiments. `b84a58f` removed those mappings from the multitask policy; do not treat them as required setup fixes. The final constructor follows `9c2ebfa`, even though an earlier commit temporarily used a different base-call signature.

Set `RBY1_MODEL_ACTION_SEED` or `action_generator_seed` to enable the dedicated `torch.Generator`, reset at policy reset and passed into action generation. This controls action noise, not all rendering or physics variation.

### Supporting MolmoSpaces bring-up and planner changes

These are useful for assets, rendering, and planner evaluation; they are distinct from the contact-scoring fix.

- `f5168f3`: `molmo_spaces/env/env.py` passes `MUJOCO_EGL_DEVICE_ID` to the classic renderer; `molmo_spaces/utils/synset_utils.py` recognizes cached NLTK corpora before downloading. The commit also changes door-generation config and adds asset-preparation/teleop scripts.
- `abcc194`: planner evaluation robustness across `configs/abstract_exp_config.py`, `data_generation/config/object_manipulation_datagen_configs.py`, `data_generation/pipeline.py`, `evaluation/eval_main.py`, `policy/solvers/curobo_pick_and_place_planner_policy.py`, and `tasks/json_eval_task_sampler.py` (all under `molmo_spaces/`).
- `fabf2af`: restores MolmoSpaces core after moving custom long-horizon behavior into the sibling extension package. Consult [the file manifest](vla_datagen_extension_file_manifest.md) for that boundary.

For the exact patches, inspect `git -C MolmoBot show COMMIT` or `git -C molmospaces show COMMIT`. The tables summarize intent; the final pinned source is authoritative.

### Repeatability and success scoring

MolmoBot chooses actions from rendered images and robot state in a closed loop. Small observation differences can affect later actions; seeded action noise improves repeatability without changing rendering. Planner and recorded replay provide controls because their actions do not depend on camera pixels in the same way.

MolmoSpaces scores the rollout through `task.judge_success()` in `data_generation/pipeline.py`. Pick and Pick-and-Place use their task metrics; Opening checks reward against `task_success_threshold`. The policy does not decide benchmark success. Replay PnP benchmark generation in `scripts/rby1_replay_runner.py` sets the expected receptacle displacement limits: `0.15m` translation and `radians(60)` rotation.

## 6. Validate before a GPU run

From `LRL_project`, with the environment active:

```bash
python scripts/rby1_runner.py describe --config configs/rby1_eval.yaml
python scripts/rby1_runner.py eval --config configs/rby1_eval.yaml --tasks pick --indices 10 --dry-run
python -m pip check
```

These check configuration and package wiring, not model success. After choosing and verifying the benchmark config, submit a small evaluation through `scripts/sbatch_rby1_eval.sh` before a full sweep. Use `scripts/sbatch_rby1_prepare.sh` if assets need preparation.

Inspect episode logs, `run_info.json`, H5 files, and videos. A clean exit only means the process completed. Corrected scoring and runnable inference do not guarantee the published benchmark success rate. Keep planner/replay controls separate from learned-model evaluation.
