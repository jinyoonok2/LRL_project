#!/usr/bin/env bash
#SBATCH --job-name=molmobot-final-llm-nav-eval
#SBATCH --partition=gpu
#SBATCH --account=xna8aw_base
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100_40gb|a100_80gb|h100_94gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-8%2
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
set -eo pipefail
PROJECT_ROOT=/u/xna8aw/workspace/live-robotics-lab/LRL_project
cd "$PROJECT_ROOT"
module load miniforge/25.3.1-py3.12 cuda/12.8.1 gcc/12.4.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /u/xna8aw/workspace/live-robotics-lab/envs/mlspaces
set -u
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/molmospaces:$PROJECT_ROOT/MolmoBot/MolmoBot:${PYTHONPATH:-}"
export MLSPACES_ASSETS_DIR="$PROJECT_ROOT/molmospaces/assets"
export MLSPACES_CACHE_DIR=/bigtemp/xna8aw/LRL/cache/molmo-spaces-resources
export MUJOCO_GL=egl
export WANDB_MODE=offline
export RBY1_MODEL_ACTION_SEED=42
python -u scripts/run_final_nav_eval.py --shard "$SLURM_ARRAY_TASK_ID" \
    --output "/bigtemp/xna8aw/nav_door/eval/final/trajectory/llm/full_validation_${SLURM_ARRAY_JOB_ID}"
