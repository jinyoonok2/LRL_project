#!/usr/bin/env bash
set -euo pipefail

# Submit one RBY1 benchmark asset-preparation YAML config to Slurm.
#
# Usage:
#   bash scripts/sbatch_rby1_prepare.sh configs/rby1_pick_pnp_binary.yaml
#   RBY1_PREPARE_TIME=04:00:00 bash scripts/sbatch_rby1_prepare.sh configs/rby1_eval.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <config.yaml> [extra rby1_runner.py prepare args...]" >&2
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
JOB_NAME="${RBY1_JOB_NAME:-prep-${CONFIG_STEM//_/-}}"
PARTITION="${RBY1_PREPARE_PARTITION:-gpu}"
ACCOUNT="${RBY1_PREPARE_ACCOUNT:-xna8aw_base}"
GRES="${RBY1_PREPARE_GRES:-}"
CPUS="${RBY1_PREPARE_CPUS:-4}"
MEM="${RBY1_PREPARE_MEM:-32G}"
TIME_LIMIT="${RBY1_PREPARE_TIME:-02:00:00}"
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
  --cpus-per-task="$CPUS"
  --mem="$MEM"
  --time="$TIME_LIMIT"
)

if [[ -n "$GRES" ]]; then
  sbatch_args+=(--gres="$GRES")
fi

sbatch "${sbatch_args[@]}" <<SBATCH
#!/usr/bin/env bash
set -euo pipefail

cd "$PROJECT_ROOT"
set +u
module load miniforge/25.3.1-py3.12 cuda/12.8.1 gcc/12.4.0
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

python scripts/rby1_runner.py prepare --config "$CONFIG"$EXTRA_ARGS
SBATCH
