#!/usr/bin/env bash
set -euo pipefail

HOST_SHORT="$(hostname -s)"
if [[ "${HOST_SHORT}" != "cheetah05" ]]; then
  echo "This debug launcher must run on cheetah05, not ${HOST_SHORT}." >&2
  echo "First connect with: ssh cheetah05" >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-/u/xna8aw/workspace/live-robotics-lab/LRL_project}"
MOLMOSPACES_DIR="${PROJECT_ROOT}/molmospaces"
PYTHON_BIN="${PYTHON_BIN:-/u/xna8aw/workspace/live-robotics-lab/envs/mlspaces/bin/python}"
CONFIG_NAME="${1:-RBY1NavPickAndPlaceFastDebugDataGenConfig}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1200}"

export PYTHONPATH="${MOLMOSPACES_DIR}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export RBY1_CUROBO_SERVER_URLS="${RBY1_CUROBO_SERVER_URLS:-local}"

cd "${MOLMOSPACES_DIR}"

echo "[nav-pnp-cheetah05] host=${HOST_SHORT}"
echo "[nav-pnp-cheetah05] config=${CONFIG_NAME}"
echo "[nav-pnp-cheetah05] timeout_sec=${TIMEOUT_SEC}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true

timeout "${TIMEOUT_SEC}s" "${PYTHON_BIN}" -u -m molmo_spaces.data_generation.main \
  "molmo_spaces.data_generation.config.object_manipulation_datagen_configs:${CONFIG_NAME}"
