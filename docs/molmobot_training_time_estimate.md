# MolmoBot RB-Y1 Training Time Estimate

This note records the small-subset MolmoBot RB-Y1 training pilot and explains how to estimate larger training runs from it.

## Purpose

We wanted to check whether the original MolmoBot-style VLA training path can run end-to-end, then use the measured throughput to estimate the cost of a larger run.

The pilot used the released RB-Y1 MolmoBot-style model setup:

```text
model_name: molmoact
language backbone: Qwen/Qwen3-4B-Instruct-2507
action_dim: 20
action_horizon: 16
```

This means the pilot is estimating the cost of training/fine-tuning the same kind of VLA used by the released MolmoBot RB-Y1 checkpoint, not a separate policy architecture.

## Data Used

The pilot used a small public MolmoBot training-data subset from Hugging Face:

```text
dataset: allenai/MolmoBot-Data
task config: RBY1PickAndPlaceDataGenConfig
part: part0
split: train
h5 files: 127
valid trajectories: 216
invalid trajectories: 6
data size: about 2.2 GB
```

The dataset was validated first so the trainer could read `valid_trajectory_index.json`.

## Completed Pilot Run

Slurm job:

```text
job id: 6533646
job name: molmobot-a100-smoke
node: cheetah04
state: COMPLETED
exit code: 0
elapsed wall time: 00:31:26
```

Hardware:

```text
4 x A100 80GB
```

Training settings:

```text
action_preset: RBY1_multitask
camera_preset: RBY1_full_with_head_gopro
seq_len: 1024
max_duration: 100 steps
device_batch_size: 8
global_batch_size: 256
precision: amp_bf16
validation: disabled for this pilot
wandb: offline
```

Important outputs:

```text
checkpoint: rby1_pnp_smoke_a100/checkpoints/step100
normalization stats: rby1_pnp_smoke_a100/rby1_pnp_smoke_norm_stats.yaml
run output size: about 19 GB
```

## Observed Metrics

From the completed A100 pilot:

```text
peak GPU memory: 13,459 MB per GPU
final device throughput: about 9,836 tokens/sec/device
final device batch throughput: about 0.150 batches/sec/device
total tokens at step 100: 26,214,400
```

The token count matches the training settings:

```text
global_batch_size * seq_len * steps
= 256 * 1024 * 100
= 26,214,400 tokens
```

The run had three useful timing views:

```text
total Slurm wall time: about 31.4 minutes
training section only: about 18.0 minutes
steady-state throughput estimate: about 11.1 minutes for 100 steps
```

The gap between these numbers comes from setup overhead, normalization-stat computation, model initialization, warmup, dataloader setup, and checkpoint saving.

## Estimation Methods

### 1. Conservative Wall-Time Estimate

Use the full Slurm elapsed time:

```text
31.4 minutes / 100 steps = 0.314 minutes/step
```

For `100,000` steps:

```text
100,000 * 0.314 minutes = 31,400 minutes
31,400 minutes / 60 = 523 hours
523 hours / 24 = 21.8 days
```

This is conservative because the one-time startup overhead is counted as if it repeats every 100 steps.

### 2. Training-Section Estimate

Use the time from the start of training to final checkpoint saving:

```text
18.0 minutes / 100 steps = 0.18 minutes/step
```

For `100,000` steps:

```text
100,000 * 0.18 minutes = 18,000 minutes
18,000 minutes / 60 = 300 hours
300 hours / 24 = 12.5 days
```

This is the better practical estimate for a long run on the same hardware and settings.

### 3. Steady-State Throughput Estimate

Use the final observed token throughput:

```text
device throughput ~= 9,836 tokens/sec/device
num devices = 4
total throughput ~= 39,344 tokens/sec
tokens per step = 256 * 1024 = 262,144
seconds per step ~= 262,144 / 39,344 = 6.66 sec/step
```

For `100,000` steps:

```text
100,000 * 6.66 sec = 666,000 sec
666,000 sec / 3600 = 185 hours
185 hours / 24 = 7.7 days
```

This is the optimistic estimate. It assumes the run stays close to the final observed throughput and ignores checkpointing, validation, and startup overhead.

## Practical Estimate

For the same setup:

```text
4 x A100 80GB
seq_len=1024
device_batch_size=8
global_batch_size=256
max_duration=100,000
```

Use this range:

```text
optimistic steady-state estimate: about 8 days
practical training-section estimate: about 12-13 days
very conservative wall-time extrapolation: about 22 days
```

The planning estimate from this pilot is:

```text
about 8-13 days for 100k steps on 4 x A100 80GB
```

## Scaling Formula

For another run with the same model and similar data-loading behavior:

```text
tokens_per_step = global_batch_size * seq_len
total_tokens = tokens_per_step * max_duration
total_throughput = tokens_per_second_per_device * num_gpus
estimated_seconds = total_tokens / total_throughput
```

Using the measured A100 value:

```text
tokens_per_second_per_device ~= 9,800
```

Example:

```text
global_batch_size = 256
seq_len = 1024
steps = 100000
num_gpus = 4

tokens_per_step = 262,144
total_tokens = 26,214,400,000
total_throughput = 9,800 * 4 = 39,200 tokens/sec
estimated_seconds = 26,214,400,000 / 39,200 = 668,735 sec
estimated_days = 7.7 days
```

Then add overhead for checkpointing, validation, startup, slower early steps, and queue interruptions. That is why the practical estimate is closer to `12-13 days`.

## Caveats

This was a training smoke test on a small subset with `216` valid trajectories. It proves the training path works and gives a useful throughput estimate, but it does not prove final model quality.

The estimate depends strongly on:

- GPU type and count,
- `global_batch_size`,
- `device_batch_size`,
- `seq_len`,
- whether validation is enabled,
- checkpoint interval,
- data-loading speed,
- whether normalization stats are already computed,
- whether the run uses the same model and optimizer settings.

The released checkpoint config records much larger original-scale training metadata, including `world_size: 128` and `max_duration: 100000`. If the goal is to match the original global batch or distributed scale, recalculate from the target settings rather than directly reusing the 4-GPU estimate.

## Recommended Next Step

Run a longer pilot on the same A100 setup:

```text
max_duration=1000
```

Use the already-computed normalization stats so the pilot mostly measures training throughput instead of setup overhead.

After that run, recompute:

```text
seconds_per_step = training_section_seconds / completed_steps
estimated_days = seconds_per_step * target_steps / 86400
```

If the `1000`-step run stays near the same steady-state throughput, the `8-13 day` estimate for `100k` steps is reasonable for this 4x A100 setup.
