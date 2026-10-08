#!/usr/bin/env bash
set -euo pipefail

# Fine-tune MolmoBot on a final opening-matched, opened-with-grasp split.
# Does not overwrite the original A-F house-split checkpoints.
#
# Usage:
#   bash scripts/sbatch_molmobot_opened_with_grasp_train.sh action_expert
#   bash scripts/sbatch_molmobot_opened_with_grasp_train.sh llm
#   bash scripts/sbatch_molmobot_opened_with_grasp_train.sh llm h100_94gb
#   DATASET_SPLIT=trajectory bash scripts/sbatch_molmobot_opened_with_grasp_train.sh llm

KIND="${1:?usage: $0 action_expert|llm [h100_94gb|a100_80gb]}"
GPU_CONSTRAINT="${2:-h100_94gb}"

case "${GPU_CONSTRAINT}" in
    h100_94gb | a100_80gb) ;;
    *)
        echo "Unsupported GPU constraint: ${GPU_CONSTRAINT}" >&2
        echo "Usage: $0 action_expert|llm [h100_94gb|a100_80gb]" >&2
        exit 2
        ;;
esac

PROJECT_ROOT="/u/xna8aw/workspace/live-robotics-lab/LRL_project"
MOLMOBOT_ROOT="${PROJECT_ROOT}/MolmoBot/MolmoBot"
CHECKPOINT="/bigtemp/xna8aw/LRL/hf_models/MolmoBot-RBY1Multitask"
DATASET_SPLIT="${DATASET_SPLIT:-house}"
case "${DATASET_SPLIT}" in
    house | trajectory) ;;
    *) echo "DATASET_SPLIT must be house or trajectory" >&2; exit 2 ;;
esac
DATASET="/bigtemp/xna8aw/nav_door/data/final/${DATASET_SPLIT}"
OUTPUT_ROOT="/bigtemp/xna8aw/nav_door/train/final/${DATASET_SPLIT}"

case "${KIND}" in
    action_expert | ae)
        KIND="action_expert"
        FT_LLM="false"
        MASTER_PORT="${MASTER_PORT:-29511}"
        ;;
    llm)
        FT_LLM="True"
        MASTER_PORT="${MASTER_PORT:-29512}"
        ;;
    *)
        echo "Unknown run kind: ${KIND}" >&2
        echo "Usage: $0 action_expert|llm [h100_94gb|a100_80gb]" >&2
        exit 2
        ;;
esac

OUTPUT="${OUTPUT_ROOT}/${KIND}"
STATS="${OUTPUT}/nav_door_norm_stats.yaml"
JOB_NAME="molmobot-final-${DATASET_SPLIT}-${KIND}"
EXP_NAME="nav_door_final_${DATASET_SPLIT}_${KIND}"

mkdir -p "${OUTPUT}" "${PROJECT_ROOT}/slurm_logs"

sbatch \
    --job-name="${JOB_NAME}" \
    --output="${PROJECT_ROOT}/slurm_logs/%x-%j.out" \
    --error="${PROJECT_ROOT}/slurm_logs/%x-%j.err" \
    --partition=gpu \
    --account=xna8aw_base \
    --constraint="${GPU_CONSTRAINT}" \
    --gres=gpu:1 \
    --cpus-per-task=10 \
    --mem=200G \
    --time=48:00:00 \
    --wrap="bash -lc '
set -eo pipefail
cd \"${MOLMOBOT_ROOT}\"
module load miniforge/25.3.1-py3.12 cuda/12.8.1 gcc/12.4.0
source \"\$(conda info --base)/etc/profile.d/conda.sh\"
set +u
conda activate /u/xna8aw/workspace/live-robotics-lab/envs/mlspaces
set -u
export PYTHONPATH=\"${PROJECT_ROOT}:${PROJECT_ROOT}/molmospaces:${MOLMOBOT_ROOT}:\${PYTHONPATH:-}\"
export WANDB_MODE=offline
export WANDB_PROJECT=molmobot-nav-door-final
export WANDB_ENTITY=dummy
python -u -m lrl_molmospaces_extensions.dataset.prepare_training --dataset \"${DATASET}\"
torchrun \
    --nnodes=1 \
    --nproc-per-node=1 \
    --master_port=\"${MASTER_PORT}\" \
    launch_scripts/train_molmobot.py \
    \"${CHECKPOINT}\" \
    --data_paths \"${DATASET}\" \
    --seq_len 1024 \
    --device_batch_size 1 \
    --global_batch_size 1 \
    --prefetch_factor 4 \
    --num_workers 4 \
    --action_preset RBY1_multitask \
    --camera_preset RBY1_full_with_head_gopro \
    --stats_path \"${STATS}\" \
    --exp_name \"${EXP_NAME}\" \
    --val_interval 500 \
    --val_max_examples 2000 \
    --save_folder=\"${OUTPUT}\" \
    --ft_llm=${FT_LLM} \
    --ft_vit=false \
    --ft_connector=false \
    --optimizer.action_expert_learning_rate=0.0001 \
    --optimizer.llm_learning_rate=1e-05 \
    --optimizer.vit_learning_rate=5e-06 \
    --optimizer.connector_learning_rate=5e-06 \
    --scheduler.action_expert_t_warmup=200 \
    --scheduler.llm_t_warmup=2000 \
    --scheduler.vit_t_warmup=200 \
    --scheduler.connector_t_warmup=200 \
    --max_duration=5000 \
    --stop_at=5000 \
    --save_interval=1000 \
    --save_interval_ephemeral=250 \
    --save_num_checkpoints_to_keep=2 \
    --checkpoint_retention_frequency=2000 \
    --model.vision_backbone.vit.float32_attention=false
'"
