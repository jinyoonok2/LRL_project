# Live Robotics Lab — Project Environment and Agent Guide

Verified locally: 2026-10-03. Owner: Jinyoon (`xna8aw`), UVA.

Read this and [SETUP.md](SETUP.md) before changing the RB-Y1 MolmoBot/MolmoSpaces stack. This guide describes Live Robotics Lab. The workspace-root documents with these same names describe the separate Part2Action/PartInstruct project.

## Project and current direction

The project first brought up the released `MolmoBot-RBY1Multitask` checkpoint and evaluated Pick, Pick-and-Place, Opening, and Door Opening. It now generates long-horizon navigation-to-door-opening demonstrations for fine-tuning that checkpoint.

The demonstration policy combines A* navigation, base alignment and handoff, and cuRobo door opening. Successful H5/video trajectories are checked for grasp quality, split into training/validation sets, and optionally rerendered to adjust navigation speed. The learned MolmoBot policy is a separate consumer of these demonstrations.

Custom long-horizon behavior lives in `lrl_molmospaces_extensions/`. Keep the accepted compatibility patches in the submodules; do not confuse them with the later extension code. See [the extension manifest](vla_datagen_extension_file_manifest.md).

## Source and repository state

| Repository | Observed branch | Verified HEAD |
|---|---|---|
| `LRL_project` | `rby1-long-horizon-vla-training` | `4aa392a23efac9f515655f219e5982e5df57c1fb` |
| `MolmoBot` | `rby1-compatible` | `9c2ebfa5cafcf13f9664e70703e25de38e1d5fcb` |
| `molmospaces` | `rby1-compatible` | `0403cb7e3bbf96fc6c622b7d31b4a24ce13cb55d` |

These are dated source snapshots, not promises about future branch tips. The project code, documentation, and report videos are committed. Environments, shared datasets, checkpoints, scheduler logs, and server-specific symlinks are outside Git and need their documented storage paths. Do not commit, push, move data, or submit new experiments unless requested.

The submodule remotes in `.gitmodules` are `jinyoonok2/MolmoBot` and `jinyoonok2/molmospaces`. Use the verified commits and inspect Git state before switching branches.

## Paths and storage

| Purpose | Location |
|---|---|
| Workspace | `/u/xna8aw/workspace/live-robotics-lab` |
| Source | Workspace + `/LRL_project` |
| Python environment | Workspace + `/envs/mlspaces` |
| Resource cache | Workspace + `/cache` → `/bigtemp/xna8aw/LRL/cache` |
| Checkpoints | Workspace + `/hf_models` → `/bigtemp/xna8aw/LRL/hf_models` |
| Released checkpoint | Workspace + `/hf_models/MolmoBot-RBY1Multitask` |
| Simulation experiment output | `LRL_project/molmospaces/assets/experiment_output` → `/bigtemp/xna8aw/nav_door` |
| Final dataset | `LRL_project/results/final` → `/bigtemp/xna8aw/nav_door/data/final` |
| Shared dataset and media | `LRL_project/molmobot_molmospace` → `/p/liverobotics/molmobot_molmospace` |
| Host runtime cache | `/p/liverobotics/xna8aw/molmospaces_kuorobot_cache` |

Shared project: `LRL_project/molmobot_molmospace` → `/p/liverobotics/molmobot_molmospace`. Its `dataset/` contains both split policies, and `media/` contains all 90 grasp-classified expert/model comparison MP4s. The retired RTX 4090 training-smoke launcher has been removed.

The Part2Action/PartInstruct environments under `/p/liverobotics/xna8aw` belong to another project. Live Robotics Lab launchers use the local `envs/mlspaces` environment.

## Installed environment snapshot

Versions read from the existing environment's package metadata:

| Package | Installed version |
|---|---|
| Python | `3.11.15` |
| PyTorch | `2.7.1+cu128` |
| torchvision | `0.22.1+cu128` |
| MuJoCo | `3.4.0` |
| transformers | `4.57.1` |
| pydantic | `2.13.4` |
| nvidia-curobo | `0.0.0` (identify by source commit instead) |

Installed cuRobo metadata identifies `allenai/curobo@87e857d46fa5398f268c7f31d26566351be8671d`. MolmoSpaces' optional dependency instead requests `417c995647fcb173a2bc094d1284b2a4f4b000ad`. Preserve the existing environment and record this discrepancy when rebuilding. This table is a partial snapshot, not a complete lockfile or a newly validated clean installation.

## Running and interpreting experiments

Use Slurm allocations for GPU work, not login/portal nodes. The YAML-driven evaluation entrypoint is `scripts/rby1_runner.py`, with `describe`, `prepare`, and `eval` commands. Submission wrappers are under `scripts/sbatch_rby1_*.sh`.

The retained final dataset contains 601 opening-matched, opened-with-grasp trajectories. `data/final/house` preserves the 432/169 house split; `data/final/trajectory` provides a linked 511/90 trajectory split (seed 42). The original, fixed-speed, and sample versions are retired. The new baseline runner is retained for future generation.

The unified baseline is configured in `configs/nav_door_baseline*.yaml` and submitted with `scripts/sbatch_nav_door_baseline.sh`; each run contains raw recordings, final H5/videos, and a stage manifest. Local shortcuts are organized under `results/`. See the extensions README for commands.

Long-horizon generation is launched through `run_rby1_simple_visible_kuorobot01.sh` and `run_rby1_door_stability_kuorobot01.sh`. Host-specific launchers need their host and paths checked before use. Dataset tools and their commands are described in [the extensions README](../lrl_molmospaces_extensions/README.md).

Query Slurm and inspect logs before reporting live job status. `slurm_logs/` contains scheduler output. Evaluation H5/video artifacts and episode logs establish behavior; a successful process exit does not establish benchmark success. MolmoSpaces task logic calls `judge_success()` to score episodes.

Compatibility fixes made the evaluation stack runnable and corrected scoring. They did not establish reproduction of published Pick/PnP performance. Historical evaluations recorded a substantial performance gap. Seeded model action sampling improves repeatability but does not guarantee byte-identical physics, H5 files, or videos.

## Slurm workflow

Use the existing UVA login session. GPU training, inference, and rendering run inside Slurm allocations. Submit shell wrappers with `bash`; they call `sbatch` themselves. Examples below are operational references, not requests to duplicate running experiments.

### Queue and job history

```bash
squeue -u xna8aw -o "%.18i %.40j %.10T %.12M %R"
sacct -u xna8aw --starttime today --format=JobID,JobName,State,Elapsed,Timelimit,ExitCode
```

For a particular job, replace the example ID with the actual submission ID:

```bash
JOB_ID=1234567
scontrol show job "$JOB_ID"
sacct -j "$JOB_ID" --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS,NodeList
```

`Resources` means requested resources are unavailable; `Priority` means other jobs have scheduling precedence. Once a job leaves the queue, inspect `sacct`, including its batch/step rows. `TIMEOUT` may leave a usable checkpoint; `OUT_OF_MEMORY` requires inspecting memory usage and logs. If scheduler access is blocked by the agent sandbox, request command escalation.

### Submit project jobs

From `/u/xna8aw/workspace/live-robotics-lab/LRL_project`, inspect the config and its resolved benchmark paths first:

```bash
python scripts/rby1_runner.py describe --config configs/rby1_eval.yaml
python scripts/rby1_runner.py eval --config configs/rby1_eval.yaml --tasks pick --indices 10 --dry-run
bash scripts/sbatch_rby1_prepare.sh configs/rby1_eval.yaml
bash scripts/sbatch_rby1_eval.sh configs/rby1_eval.yaml --tasks pick --indices 10
```

The evaluation wrapper defaults to one GPU, 8 CPUs, 96GB RAM, and three hours on partition `gpu`, account `xna8aw_base`. Override resources explicitly when needed:

```bash
RBY1_SLURM_TIME=04:00:00 RBY1_SLURM_MEM=128G bash scripts/sbatch_rby1_eval.sh configs/rby1_eval.yaml
```

The combined baseline submits generation and final video preparation together:

```bash
BASELINE_SLURM_TIME=02:00:00 bash scripts/sbatch_nav_door_baseline.sh configs/nav_door_baseline_smoke.yaml --run-id smoke_001
bash scripts/sbatch_nav_door_baseline.sh configs/nav_door_baseline.yaml --shard a --run-id production_001
```

Its defaults are one GPU, 8 CPUs, 96GB RAM, and 48 hours. `BASELINE_SLURM_TIME` and `BASELINE_SLURM_MEM` override its resource requests. `--dry-run` on the Python baseline runner previews configuration and paths without creating a run.

Current training and full-validation launchers:

```bash
DATASET_SPLIT=trajectory bash scripts/sbatch_molmobot_opened_with_grasp_train.sh llm h100_94gb
sbatch scripts/sbatch_final_llm_nav_eval.sh
```

These examples are references, not instructions to duplicate completed jobs. Training requests one GPU, 200GB host RAM, and 48 hours; LLM + action-expert training peaked near 85GB GPU memory, so use H100 94GB for the current configuration. The launcher defaults to the house split; `DATASET_SPLIT=trajectory` selects the trajectory split. Checkpoints are separated under `/bigtemp/xna8aw/nav_door/train/final/SPLIT/MODE`.

Full validation accepts A40, A100 40/80GB, or H100 94GB via node features, using generic `gpu:1` to avoid GPU-label mismatches. Nine shards cover all 90 held-out trajectory starts, with at most two simultaneous jobs. The older fixed-speed rerender and retired-dataset training-smoke wrappers have been removed; use the unified baseline for future generation.

Shared resources are linked at `LRL_project/molmobot_molmospace`, pointing to `/p/liverobotics/molmobot_molmospace`. Both split policies live under `dataset/`; grasp-classified comparison videos and metrics live under `media/`.

### Logs and failures

Wrappers create `slurm_logs/` before submission. `%x` is the job name and `%j` is its ID:

```bash
JOB_NAME=actual-job-name
JOB_ID=1234567
tail -n 60 "slurm_logs/${JOB_NAME}-${JOB_ID}.out"
tail -n 60 "slurm_logs/${JOB_NAME}-${JOB_ID}.err"
```

Combine scheduler state with the last training step, saved checkpoint, and episode results. Check output directories declared in the launcher; Slurm completion alone does not prove training progress or task success. Avoid printing credentials when inspecting logs.

### Resume training

First confirm the old job stopped, identify the latest complete checkpoint, and inspect its saved configuration. The MolmoBot launcher accepts training configuration overrides through its trailing dotlist arguments, including `--load_path=/absolute/checkpoint/path`. It also exposes `--reset_optimizer`, which intentionally discards optimizer state and should only be used when needed.

The current training wrappers do **not** expose a `RESUME` environment variable or forward arbitrary extra arguments. For a requested resume, adapt a separate submission script to pass the verified `--load_path`, preserve the dataset and training configuration, and set the intended step limit. Keep the original run intact and verify the resumed step and optimizer state in logs. Do not copy Part2Action's `RESUME=...` convention into these wrappers.

### Interactive GPU diagnostics and cancellation

```bash
srun -p gpu --account=xna8aw_base --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=01:00:00 --pty bash
```

Inside the allocation, activate `envs/mlspaces` as described in [SETUP.md](SETUP.md), then check:

```bash
hostname
nvidia-smi
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
exit
```

Cancel only a job the user has asked to stop: `scancel JOB_ID`. Batch jobs survive logout; interactive allocations should be exited when diagnostics finish.

## Related documents

- [Setup and compatibility patch inventory](SETUP.md)
- [Extension commands](../lrl_molmospaces_extensions/README.md)
- [Dated research updates](slide_history/README.md)

## Changelog

- 2026-10-03 — Added this guide and `docs/SETUP.md` from local source history, installed package metadata, and existing research notes. No environments, jobs, datasets, or submodule source were changed.

- 2026-10-03 — Removed superseded project-root `SETUP.md` and `docs/rby1_molmobot_eval_compatibility.md` after consolidating their setup and compatibility guidance. Updated the landing README; retained research history and separate Part2Action documents.

- 2026-10-03 — Consolidated determinism and success-scoring details into the setup guide; removed the three superseded documents. Research plans and next steps were excluded from the maintained guides. Dated slides and Git history retain the research record.

- 2026-10-03 — Added Slurm queue/history commands, verified project submission examples, log inspection, resume limitations, and interactive diagnostics. No jobs were submitted or changed.

- 2026-10-03 — Added unified baseline output layout and local `results/legacy` shortcuts. Existing datasets were not moved or relabeled. New runs use `/bigtemp/xna8aw/nav_door/baselines`.

- 2026-10-03 — Retained the existing 601-trajectory opening-matched, grasp-filtered dataset as `data/final`; retired original, fixed-speed, and sample dataset versions. Updated result shortcuts and the training launcher. The baseline runner remains for future use; no new generation was requested.

- 2026-10-03 — Organized the retained dataset under separate house and trajectory split folders. Physical files live under house; trajectory uses links. Preserved the house split and added a deterministic 511/90 trajectory split without copying videos or H5 files.

- 2026-10-03 — Submitted job `6668431` (`molmobot-final-trajectory-llm`): LLM + action expert on the existing final trajectory split (511 train / 90 val), with vision backbone and connector frozen. Requests one H100 94GB, 200GB host RAM, 48 hours, and 5,000 steps. Output: `/bigtemp/xna8aw/nav_door/train/final/trajectory/llm`. The extension `dataset.prepare_training` first checks video/action frame counts, adds wrist-video aliases, and builds trajectory indexes without changing H5 files or upstream code. Status at submission: pending. Logs: `slurm_logs/molmobot-final-trajectory-llm-6668431.out` and `.err`.

- 2026-10-04 — Submitted action-expert-only job `6673698` on one A100 80GB, using the final trajectory split (511 train / 90 val), 5,000 steps and a 48-hour limit. LLM, vision backbone and connector are frozen. Output: `/bigtemp/xna8aw/nav_door/train/final/trajectory/action_expert`. Job `6668431` remains queued independently for LLM + action-expert training. Initial A100 status: pending (`ReqNodeNotAvail`).

- 2026-10-05 — Canceled pending action-expert training job `6673698` at user request. Submitted evaluation job `6676831`: convert final LLM checkpoint to unsharded inference format, then run 10 standard door-opening benchmark episodes (indices 0,10,...,90), seed 42, horizon 1000, one A40, 96GB host RAM, 12 hours. Extension evaluation config disables point prompts, conditioning image, and camera warping to match fine-tuning inputs. This is a manipulation benchmark, not composite navigation evaluation. Output: `/bigtemp/xna8aw/nav_door/eval/final/trajectory/llm/door_smoke_6676831`.

- 2026-10-06 — Shared dataset copy requested at `/p/liverobotics/molmobot_molmospace/dataset`. Contains the final 601 speed-matched, opened-with-grasp demonstrations and both split policies (house 432/169; trajectory 511/90). The original `/bigtemp/xna8aw/nav_door/data/final` remains in place. Shared-copy completion and verification are recorded in the destination `shared_copy_manifest.json`. Use `trajectory` as the MolmoBot data root for the completed training configuration.

Shared-copy verification completed: 2,417 regular files matched original sizes (19,618,893,303 bytes), and all 4,808 symlinks resolve inside the shared folder. The destination includes a usage README and verification manifest; group-access permissions are inherited from `/p/liverobotics`.

- 2026-10-06 — Broadened pending evaluation job `6676831` in place to `gpu:1` with features `a40|a100_40gb|a100_80gb|h100_94gb`, preserving queue age. Updated the submission script to match. Unallocated A40 GPUs on jaguar01 are reserved for another user until October 19; unallocated counts alone do not establish scheduler access.

- 2026-10-06 — Submitted full held-out-trajectory navigation-to-door evaluation array `6687763` (9 shards × 10 episodes, concurrency 2). Builds all 90 benchmark starts from `results/final/trajectory/val`, retaining ProcTHOR train scene identities despite trajectory validation membership. Uses recorded robot placement/joints, target door, reset state, openness threshold and instruction; learned model controls every phase. Seed 42; horizon `max(2000, 2 * recorded_length)`. Compatible GPU features A40/A100/H100; 96GB RAM, 12h per shard. Output `/bigtemp/xna8aw/nav_door/eval/final/trajectory/llm/full_validation_6687763`. Preparation verified exactly 90 episodes across 42 houses, no H5 overlap with training, and initial joint agreement for every episode. Earlier standard door-only benchmark `6676831` completed with 2/10 successes and no process errors; it is distinct from this full validation run.

- 2026-10-07 — Full evaluation array 6687763 completed all 90 episodes: 28 successes (31.1%), 62 task failures, zero process errors. Replaced slide-history media at user request with 90 matched expert/model comparison GIFs, split into `media/success` and `media/failure`; all names and image headers include outcomes. Manifest retains source H5/video references.

- 2026-10-07 — Applied the existing 30 cumulative grasp-frame filter to full-validation model H5s: 1 opened with qualifying grasp, 27 opened without qualifying grasp (26 with zero frames, one with 1 frame), 62 did not open. Metrics under slide-history `media/grasp_index.tsv` and `grasp_summary.json`; comparison MP4s organized accordingly. Removed all media GIFs at user request. Original rollout H5s and raw results were unchanged.

- 2026-10-07 — Reorganized shared resources under `/p/liverobotics/molmobot_molmospace`: `dataset/` contains both house and trajectory train/val splits, and `media/` contains 90 comparison MP4s with grasp metrics and galleries. The former shared dataset folder was moved beneath this project root at user request; working bigtemp data and local media remain in place.

- 2026-10-07 — Added the shared-project symlink; removed obsolete ten-episode/GIF and retired-dataset launchers, simplified comparison export to MP4-only, and cleared extension/script bytecode caches. Kept reusable modules, working results, current launchers, and historical logs.

- 2026-10-08 — Cleaned the parent repository before preparing a project summary. Retained only local branches `main`, `rby1`, and `rby1-long-horizon-vla-training`; removed stale worktree registration and local branches `backup/rby1-dev-before-mainbase-20260603_032414` (tip `4d9f04c`), `rby1-custom` (`998fb74`), `rby1-dev` (`16105c0`), `rby1-manual-teleop` (`5a08d97`), and `rby1-solo-panel-evaluation` (`4aa392a`). Remote branches and submodule branches were unchanged. Removed the retired RTX 4090 training-smoke and cheetah05 Pick/PnP debug launchers, the old 450-step Pick/PnP experiment config, 40 unreferenced logs from obsolete runs with Python exceptions, and generated test caches. Removed five empty abandoned-training directories and the broken `molmobot_datasets/nav_door_shards_abcdef` link to the retired dataset, then removed its empty parent directory. Retained historical completed checkpoints, audit records, current training/evaluation logs, the final dataset, and all 90 comparison videos.

- 2026-10-08 — Consolidated the identical MolmoBot branches `rby1-custom` and `rby1-long-horizon-vla-training` into `rby1-compatible`, preserving commit `9c2ebfa`. Published the new branch and removed both old branch names locally and from the MolmoBot fork. The parent repository and MolmoSpaces branch names and submodule commit pins were unchanged.

- 2026-10-08 — Consolidated MolmoSpaces onto `rby1-long-horizon-vla`, removing the redundant `rby1-custom` name locally and from the fork. Added [the fork change record](../molmospaces/docs/FORK_CHANGES.md), covering `cd23bec` through compatibility commit `9b5f4e1` and final refactor `fabf2af`. The final source trees at `9b5f4e1` and `fabf2af` are identical; long-horizon implementation resides in the parent extensions. Updated the submodule pin to `d4bacaeae830c085f02b433a9da9296d394df515`, which adds only that record and its README link, and updated setup guidance.

- 2026-10-08 — Renamed the retained MolmoSpaces branch to `rby1-compatible` to match MolmoBot, preserving the final VLA implementation and history. Published the new name and removed `rby1-long-horizon-vla` from the fork. Updated the fork change record and parent submodule pin to `0403cb7e3bbf96fc6c622b7d31b4a24ce13cb55d`; only documentation changed.
