#!/usr/bin/env bash
set -euo pipefail

# Submit one RBY1 recorded-trajectory replay YAML config to Slurm.
#
# Usage:
#   bash scripts/sbatch_rby1_replay_eval.sh configs/rby1_replay_val.yaml
#   RBY1_REPLAY_TIME=01:00:00 bash scripts/sbatch_rby1_replay_eval.sh \
#     configs/rby1_replay_val.yaml --configs RBY1PickDataGenConfig --max-episodes-per-config 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <config.yaml> [extra rby1_replay_runner.py eval args...]" >&2
  exit 2
fi

CONFIG="$1"
shift

if [[ "$CONFIG" != /* ]]; then
  CONFIG="$PROJECT_ROOT/$CONFIG"
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config: $CONFIG" >&2
  exit 2
fi

mkdir -p "$PROJECT_ROOT/slurm_logs"

CONFIG_STEM="$(basename "$CONFIG" .yaml)"
JOB_NAME="${RBY1_REPLAY_JOB_NAME:-replay-${CONFIG_STEM//_/-}}"
PARTITION="${RBY1_REPLAY_PARTITION:-gpu}"
ACCOUNT="${RBY1_REPLAY_ACCOUNT:-xna8aw_base}"
GRES="${RBY1_REPLAY_GRES:-gpu:1}"
CPUS="${RBY1_REPLAY_CPUS:-8}"
MEM="${RBY1_REPLAY_MEM:-96G}"
TIME_LIMIT="${RBY1_REPLAY_TIME:-02:00:00}"
CONDA_ENV="${CONDA_ENV:-/u/xna8aw/workspace/live-robotics-lab/envs/mlspaces}"

EXTRA_ARGS=""
if [[ "$#" -gt 0 ]]; then
  EXTRA_ARGS="$(printf ' %q' "$@")"
fi

sbatch_args=(
  --job-name="$JOB_NAME"
  --output="$PROJECT_ROOT/slurm_logs/%x-%j.out"
  --error="$PROJECT_ROOT/slurm_logs/%x-%j.err"
  --partition="$PARTITION"
  --account="$ACCOUNT"
  --gres="$GRES"
  --cpus-per-task="$CPUS"
  --mem="$MEM"
  --time="$TIME_LIMIT"
)

sbatch "${sbatch_args[@]}" <<SBATCH
#!/usr/bin/env bash
set -euo pipefail

cd "$PROJECT_ROOT"
set +u
module load miniforge/25.3.1-py3.12 cuda/12.8.1 gcc/12.4.0
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="\${PYTHONHASHSEED:-42}"
export CUBLAS_WORKSPACE_CONFIG="\${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

python scripts/rby1_replay_runner.py eval --config "$CONFIG"$EXTRA_ARGS
SBATCH
