#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" != "kuorobot01" ]]; then
    echo "This launcher is intended for kuorobot01." >&2
    exit 1
fi

PROJECT_ROOT="/u/xna8aw/workspace/live-robotics-lab/LRL_project"
MOLMOBOT_ROOT="${PROJECT_ROOT}/MolmoBot/MolmoBot"
PYTHON_ENV="/u/xna8aw/workspace/live-robotics-lab/envs/mlspaces"
STAGING_ROOT="/p/liverobotics/xna8aw/molmobot_nav_door_smoke"

CHECKPOINT="${STAGING_ROOT}/checkpoint"
DATASET="${STAGING_ROOT}/data"
OUTPUT="${STAGING_ROOT}/output/rtx4090_step1"
CACHE="${STAGING_ROOT}/cache"

mkdir -p \
    "${OUTPUT}" \
    "${CACHE}/hf" \
    "${CACHE}/torch" \
    "${CACHE}/triton" \
    "${CACHE}/inductor" \
    "${CACHE}/tmp"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${MOLMOBOT_ROOT}:${PYTHONPATH:-}"
export PATH="${PYTHON_ENV}/bin:${PATH}"
export HF_HOME="${CACHE}/hf"
export TORCH_HOME="${CACHE}/torch"
export TRITON_CACHE_DIR="${CACHE}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE}/inductor"
export XDG_CACHE_HOME="${CACHE}"
export TMPDIR="${CACHE}/tmp"
export WANDB_MODE=offline
export WANDB_PROJECT=molmobot-nav-door-smoke
export WANDB_ENTITY=dummy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

cd "${MOLMOBOT_ROOT}"

nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader

exec "${PYTHON_ENV}/bin/torchrun" \
    --nnodes=1 \
    --nproc-per-node=1 \
    --master_port=29503 \
    launch_scripts/train_molmobot.py \
    "${CHECKPOINT}" \
    --data_paths "${DATASET}" \
    --seq_len 512 \
    --device_batch_size 1 \
    --global_batch_size 1 \
    --action_preset RBY1_multitask \
    --camera_preset RBY1_full_with_head_gopro \
    --stats_path="${OUTPUT}/nav_door_norm_stats.yaml" \
    --no_val \
    --save_folder="${OUTPUT}/checkpoints" \
    --max_duration=1 \
    --stop_at=1 \
    --save_interval=1 \
    --save_interval_ephemeral=1 \
    --save_num_checkpoints_to_keep=1 \
    --checkpoint_retention_frequency=1 \
    --model.vision_backbone.vit.float32_attention=false
