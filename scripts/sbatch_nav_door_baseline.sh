#!/usr/bin/env bash
# Submit the complete baseline: generation -> speed matching + points -> validation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${1:?usage: bash scripts/sbatch_nav_door_baseline.sh config.yaml [runner arguments]}"
shift
if [[ "${CONFIG}" != /* ]]; then CONFIG="${PROJECT_ROOT}/${CONFIG}"; fi
if [[ ! -f "${CONFIG}" ]]; then echo "Missing config: ${CONFIG}" >&2; exit 2; fi
EXTRA_ARGS=""
if [[ "$#" -gt 0 ]]; then EXTRA_ARGS="$(printf ' %q' "$@")"; fi
mkdir -p "${PROJECT_ROOT}/slurm_logs"

sbatch \
    --job-name=nav-door-baseline \
    --output="${PROJECT_ROOT}/slurm_logs/%x-%j.out" \
    --error="${PROJECT_ROOT}/slurm_logs/%x-%j.err" \
    --partition=gpu \
    --account=xna8aw_base \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem="${BASELINE_SLURM_MEM:-96G}" \
    --time="${BASELINE_SLURM_TIME:-48:00:00}" <<SBATCH
#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_ROOT}"
set +u
module load miniforge/25.3.1-py3.12 cuda/12.8.1 gcc/12.4.0
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate /u/xna8aw/workspace/live-robotics-lab/envs/mlspaces
set -u
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/molmospaces:\${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export RBY1_CUROBO_SERVER_URLS=local
python -u -m lrl_molmospaces_extensions.baseline.runner --config "${CONFIG}"${EXTRA_ARGS}
SBATCH
