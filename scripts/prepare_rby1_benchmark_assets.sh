#!/usr/bin/env bash
set -euo pipefail

# Prepare selected RBY1 benchmark episode assets on the UVA CS server.
# Default: prepare 30 selected indices for all four RBY1 benchmark families:
# 0..9 plus 25, 75, ..., 975.
#
# Usage:
#   ./scripts/prepare_rby1_benchmark_assets.sh
#   ./scripts/prepare_rby1_benchmark_assets.sh 0 1 2
#   ./scripts/prepare_rby1_benchmark_assets.sh pick,pnp 10 21 33
#   RBY1_EVAL_CONFIG=configs/rby1_pick_pnp_expansion.env ./scripts/prepare_rby1_benchmark_assets.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RBY1_EVAL_CONFIG="${RBY1_EVAL_CONFIG:-$PROJECT_ROOT/configs/rby1_eval.env}"

if [[ ! -f "$RBY1_EVAL_CONFIG" && "$RBY1_EVAL_CONFIG" != /* ]]; then
  RBY1_EVAL_CONFIG="$PROJECT_ROOT/$RBY1_EVAL_CONFIG"
fi

if [[ ! -f "$RBY1_EVAL_CONFIG" ]]; then
  echo "Missing RBY1 eval config: $RBY1_EVAL_CONFIG" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$RBY1_EVAL_CONFIG"

export MLSPACES_CACHE_DIR="${MLSPACES_CACHE_DIR:-$CACHE_ROOT/molmo-spaces-resources}"
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$CACHE_ROOT/molmospaces/assets}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export PYTHONPATH="$PROJECT_ROOT/molmospaces:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

mkdir -p "$MLSPACES_CACHE_DIR" "$MLSPACES_ASSETS_DIR" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME"

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

cd "$PROJECT_ROOT/molmospaces"

is_task_arg() {
  [[ "$1" == "all" || "$1" == *","* || "$1" == "pick" || "$1" == "pnp" || "$1" == "opening" || "$1" == "door_opening" ]]
}

if [[ "$#" -gt 0 ]] && is_task_arg "$1"; then
  TASK_ARG="$1"
  shift
else
  TASK_ARG="all"
fi

if [[ "$TASK_ARG" == "all" ]]; then
  TASKS=("${DEFAULT_TASKS[@]}")
else
  IFS=',' read -r -a TASKS <<< "$TASK_ARG"
fi

if [[ "$#" -gt 0 ]]; then
  INDICES=("$@")
else
  INDICES=("${DEFAULT_INDICES[@]}")
fi

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "RBY1_EVAL_CONFIG=$RBY1_EVAL_CONFIG"
echo "MLSPACES_CACHE_DIR=$MLSPACES_CACHE_DIR"
echo "MLSPACES_ASSETS_DIR=$MLSPACES_ASSETS_DIR"
echo "CONDA_ENV=$CONDA_ENV"
echo "BENCHMARK_ROOT=$BENCHMARK_ROOT"
echo "TASKS=${TASKS[*]}"
echo "INDICES=${INDICES[*]}"

echo "Installing/verifying base MolmoSpaces resource metadata..."
python -m molmo_spaces.molmo_spaces_constants

for task in "${TASKS[@]}"; do
  case "$task" in
    pick)
      benchmark_dir="$BENCHMARK_ROOT/procthor-objaverse/rby1_benchmarks/pick_benchmark"
      ;;
    pnp)
      benchmark_dir="$BENCHMARK_ROOT/procthor-objaverse/rby1_benchmarks/pnp_benchmark"
      ;;
    opening)
      benchmark_dir="$BENCHMARK_ROOT/ithor/rby1_benchmarks/opening_benchmark"
      ;;
    door_opening)
      benchmark_dir="$BENCHMARK_ROOT/procthor-10k/rby1_benchmarks/door_opening_benchmark"
      ;;
    *)
      echo "Unknown task: $task. Use one of: pick, pnp, opening, door_opening, all." >&2
      exit 2
      ;;
  esac

  if [[ ! -f "$benchmark_dir/benchmark.json" ]]; then
    echo "Missing benchmark.json: $benchmark_dir" >&2
    exit 2
  fi

  echo
  echo "Preparing assets for: $benchmark_dir"
  python scripts/benchmarks/prepare_benchmark_assets.py \
    --benchmark_dir "$benchmark_dir" \
    --idx "${INDICES[@]}"
done

echo
echo "RBY1 benchmark asset preparation complete."
