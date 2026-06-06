#!/usr/bin/env bash
set -euo pipefail

# Run the downloaded MolmoBot RBY1 multitask checkpoint on prepared RBY1 benchmarks.
# Usage:
#   ./scripts/run_rby1_multitask_eval.sh [pick|pnp|opening|door_opening] [idx] [output_root]
#   ./scripts/run_rby1_multitask_eval.sh all all
#   ./scripts/run_rby1_multitask_eval.sh pick,pnp 10,21,33
#
# This script is intended to run inside a Slurm GPU allocation or from an sbatch
# script. The all/all mode runs the default 30 selected indices for all benchmarks.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RBY1_EVAL_CONFIG="${RBY1_EVAL_CONFIG:-$PROJECT_ROOT/configs/rby1_eval.env}"

if [[ ! -f "$RBY1_EVAL_CONFIG" ]]; then
  echo "Missing RBY1 eval config: $RBY1_EVAL_CONFIG" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$RBY1_EVAL_CONFIG"

TASK_ARG="${1:-all}"
IDX_ARG="${2:-all}"
OUTPUT_ROOT_ARG="${3:-}"

export MLSPACES_CACHE_DIR="${MLSPACES_CACHE_DIR:-$CACHE_ROOT/molmo-spaces-resources}"
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$CACHE_ROOT/molmospaces/assets}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$CACHE_ROOT/huggingface/transformers}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export PYTHONPATH="$PROJECT_ROOT/MolmoBot/MolmoBot:$PROJECT_ROOT/molmospaces:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME"

# Conda CUDA activation scripts can read unset variables such as
# NVCC_PREPEND_FLAGS, so keep nounset disabled only while activating.
set +u
if command -v module >/dev/null 2>&1; then
  module load miniforge/25.3.1-py3.12 cuda/12.8.1 gcc/12.4.0
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
else
  echo "conda is not available. Run: module load miniforge/25.3.1-py3.12 cuda/12.8.1 gcc/12.4.0" >&2
  exit 1
fi
set -u

if [[ ! -f "$CHECKPOINT/model.pt" ]]; then
  echo "Missing checkpoint weights: $CHECKPOINT/model.pt" >&2
  exit 3
fi

if [[ ! -f "$CHECKPOINT/config.yaml" ]]; then
  echo "Missing checkpoint config: $CHECKPOINT/config.yaml" >&2
  exit 3
fi

configure_task() {
  local task="$1"

  case "$task" in
    pick)
      CONFIG="olmo.eval.configure_molmo_spaces:MolmoBotRBY1PickPnPEvalConfig"
      BENCHMARK="$BENCHMARK_ROOT/procthor-objaverse/rby1_benchmarks/pick_benchmark"
      ;;
    pnp)
      CONFIG="olmo.eval.configure_molmo_spaces:MolmoBotRBY1PickPnPEvalConfig"
      BENCHMARK="$BENCHMARK_ROOT/procthor-objaverse/rby1_benchmarks/pnp_benchmark"
      ;;
    opening)
      CONFIG="olmo.eval.configure_molmo_spaces:MolmoBotRBY1DoorPlusOpenEvalConfig"
      BENCHMARK="$BENCHMARK_ROOT/ithor/rby1_benchmarks/opening_benchmark"
      ;;
    door_opening)
      CONFIG="olmo.eval.configure_molmo_spaces:MolmoBotRBY1DoorPlusOpenEvalConfig"
      BENCHMARK="$BENCHMARK_ROOT/procthor-10k/rby1_benchmarks/door_opening_benchmark"
      ;;
    *)
      echo "Unknown task: $task. Use one of: pick, pnp, opening, door_opening, all." >&2
      exit 2
      ;;
  esac
}

run_one_episode() {
  local task="$1"
  local idx="$2"
  local output_root_arg="${3:-}"

  configure_task "$task"

  if [[ ! -f "$BENCHMARK/benchmark.json" ]]; then
    echo "Missing benchmark.json: $BENCHMARK" >&2
    echo "Run ./scripts/prepare_rby1_benchmark_assets.sh first." >&2
    exit 4
  fi

  local meta_env
  meta_env="$(mktemp)"
  local meta_args=(
    --benchmark_json "$BENCHMARK/benchmark.json"
    --task "$task"
    --idx "$idx"
    --benchmark_dir "$BENCHMARK"
    --checkpoint "$CHECKPOINT"
    --output_base_dir "$OUTPUT_BASE_DIR"
    --env_file "$meta_env"
  )

  if [[ -n "$output_root_arg" ]]; then
    meta_args+=(--output_root "$output_root_arg")
  fi

  python "$PROJECT_ROOT/scripts/rby1_eval_metadata.py" "${meta_args[@]}"
  # shellcheck disable=SC1090
  source "$meta_env"
  rm -f "$meta_env"

  echo "PROJECT_ROOT=$PROJECT_ROOT"
  echo "RBY1_EVAL_CONFIG=$RBY1_EVAL_CONFIG"
  echo "CONDA_ENV=$CONDA_ENV"
  echo "CHECKPOINT=$CHECKPOINT"
  echo "TASK=$task"
  echo "IDX=$idx"
  echo "TASK_DESCRIPTION=$TASK_DESCRIPTION"
  echo "OBJECT_LABEL=$OBJECT_LABEL"
  echo "EPISODE_SLUG=$EPISODE_SLUG"
  echo "CONFIG=$CONFIG"
  echo "BENCHMARK=$BENCHMARK"
  echo "OUTPUT_ROOT=$OUTPUT_ROOT"
  echo "RUN_INFO_PATH=$RUN_INFO_PATH"
  echo "RBY1_TASK_HORIZON_STEPS=${RBY1_TASK_HORIZON_STEPS:-benchmark_default}"

  cd "$PROJECT_ROOT/MolmoBot/MolmoBot"

  local eval_args=(
    --checkpoint_path "$CHECKPOINT"
    --benchmark_path "$BENCHMARK"
    --eval_config_cls "$CONFIG"
    --output_dir "$OUTPUT_ROOT"
    --num_workers 1
    --idx "$idx"
  )

  if [[ -n "$RBY1_TASK_HORIZON_STEPS" ]]; then
    eval_args+=(--task_horizon "$RBY1_TASK_HORIZON_STEPS")
  fi

  python launch_scripts/run_eval.py "${eval_args[@]}"
}

if [[ "$TASK_ARG" == "all" || "$TASK_ARG" == *","* || "$IDX_ARG" == "all" || "$IDX_ARG" == *","* ]]; then
  if [[ -n "$OUTPUT_ROOT_ARG" ]]; then
    echo "Custom output_root is only supported for single-episode runs." >&2
    exit 2
  fi

  if [[ "$TASK_ARG" == "all" ]]; then
    TASKS=("${DEFAULT_TASKS[@]}")
  else
    IFS=',' read -r -a TASKS <<< "$TASK_ARG"
    for task in "${TASKS[@]}"; do
      configure_task "$task"
    done
  fi

  if [[ "$IDX_ARG" == "all" ]]; then
    INDICES=("${DEFAULT_INDICES[@]}")
  else
    IFS=',' read -r -a INDICES <<< "$IDX_ARG"
  fi

  RUN_ID="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-$$}"
  LOG_DIR="$PROJECT_ROOT/logs/rby1_multitask_${RUN_ID}"
  OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-$DEFAULT_OUTPUT_BASE_DIR/${RUN_ID}}"
  SUMMARY_FILE="$LOG_DIR/summary.tsv"
  mkdir -p "$LOG_DIR" "$OUTPUT_BASE_DIR"

  echo "PROJECT_ROOT=$PROJECT_ROOT"
  echo "RBY1_EVAL_CONFIG=$RBY1_EVAL_CONFIG"
  echo "TASKS=${TASKS[*]}"
  echo "INDICES=${INDICES[*]}"
  echo "LOG_DIR=$LOG_DIR"
  echo "OUTPUT_BASE_DIR=$OUTPUT_BASE_DIR"
  echo -e "task\tidx\tstatus\texit_code\tlog_file" > "$SUMMARY_FILE"

  success_count=0
  failure_count=0

  for task in "${TASKS[@]}"; do
    for idx in "${INDICES[@]}"; do
      if ! [[ "$idx" =~ ^[0-9]+$ ]]; then
        echo "Invalid index: $idx" >&2
        exit 2
      fi

      log_file="$LOG_DIR/${task}_idx_${idx}.log"
      echo
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting task=$task idx=$idx"
      echo "Log: $log_file"

      if run_one_episode "$task" "$idx" > "$log_file" 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed task=$task idx=$idx"
        echo -e "$task\t$idx\tsuccess\t0\t$log_file" >> "$SUMMARY_FILE"
        success_count=$((success_count + 1))
      else
        exit_code=$?
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed task=$task idx=$idx exit_code=$exit_code"
        echo -e "$task\t$idx\tfailed\t$exit_code\t$log_file" >> "$SUMMARY_FILE"
        failure_count=$((failure_count + 1))
      fi
    done
  done

  echo
  echo "RBY1 multitask sweep complete."
  echo "Successes: $success_count"
  echo "Failures: $failure_count"
  echo "Summary: $SUMMARY_FILE"

  if [[ "$failure_count" -gt 0 ]]; then
    exit 1
  fi
else
  OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-$DEFAULT_OUTPUT_BASE_DIR}"
  run_one_episode "$TASK_ARG" "$IDX_ARG" "$OUTPUT_ROOT_ARG"
fi
