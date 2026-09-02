#!/usr/bin/env bash
set -euo pipefail

GPU_CONSTRAINT="${1:-h100_94gb}"
NUM_GPUS="${2:-2}"

case "${GPU_CONSTRAINT}" in
    h100_94gb | a100_80gb) ;;
    *)
        echo "Unsupported GPU constraint: ${GPU_CONSTRAINT}" >&2
        echo "Usage: $0 [h100_94gb|a100_80gb] [num_gpus]" >&2
        exit 2
        ;;
esac

PROJECT_ROOT="/u/xna8aw/workspace/live-robotics-lab/LRL_project"
MOLMOBOT_ROOT="${PROJECT_ROOT}/MolmoBot/MolmoBot"
CHECKPOINT="/bigtemp/xna8aw/LRL/hf_models/MolmoBot-RBY1Multitask"
DATASET="/bigtemp/xna8aw/molmobot_data/nav_door_long_horizon_smoke"
OUTPUT="/bigtemp/xna8aw/molmobot_runs/nav_door_long_horizon_smoke_${GPU_CONSTRAINT}"
STATS="${OUTPUT}/nav_door_norm_stats.yaml"
JOB_NAME="molmobot-nav-door-${GPU_CONSTRAINT}-smoke"
MASTER_PORT="${MASTER_PORT:-29501}"

mkdir -p "${OUTPUT}" "${PROJECT_ROOT}/slurm_logs"

sbatch \
    --job-name="${JOB_NAME}" \
    --output="${PROJECT_ROOT}/slurm_logs/%x-%j.out" \
    --error="${PROJECT_ROOT}/slurm_logs/%x-%j.err" \
    --partition=gpu \
    --account=xna8aw_base \
    --constraint="${GPU_CONSTRAINT}" \
    --gres="gpu:${NUM_GPUS}" \
    --cpus-per-task=10 \
    --mem=200G \
    --time=02:00:00 \
    --wrap="bash -lc '
set -euo pipefail
cd \"${MOLMOBOT_ROOT}\"
module load miniforge/25.3.1-py3.12 cuda/12.8.1 gcc/12.4.0
source \"\$(conda info --base)/etc/profile.d/conda.sh\"
conda activate /u/xna8aw/workspace/live-robotics-lab/envs/mlspaces
export PYTHONPATH=\${PYTHONPATH:-.}
export WANDB_MODE=offline
export WANDB_PROJECT=molmobot-nav-door-smoke
export WANDB_ENTITY=dummy
torchrun \
    --nnodes=1 \
    --nproc-per-node=\"${NUM_GPUS}\" \
    --master_port=\"${MASTER_PORT}\" \
    launch_scripts/train_molmobot.py \
    \"${CHECKPOINT}\" \
    --data_paths \"${DATASET}\" \
    --seq_len 1024 \
    --device_batch_size 1 \
    --global_batch_size \"${NUM_GPUS}\" \
    --action_preset RBY1_multitask \
    --camera_preset RBY1_full_with_head_gopro \
    --stats_path=\"${STATS}\" \
    --no_val \
    --save_folder=\"${OUTPUT}\" \
    --max_duration=2 \
    --stop_at=2 \
    --save_interval=2 \
    --save_interval_ephemeral=1 \
    --save_num_checkpoints_to_keep=1 \
    --checkpoint_retention_frequency=2 \
    --model.vision_backbone.vit.float32_attention=false
'"
