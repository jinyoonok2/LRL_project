#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-nav-open}"
SCOPE="${2:-same-door}"
DOOR_ACTION="${3:-}"
ATTEMPTS_PER_DOOR="${4:-${RBY1_DOOR_STABILITY_ATTEMPTS_PER_DOOR:-10}}"
STABILITY_SHARD=""

case "${MODE}:${SCOPE}" in
    nav-open:same-door)
        CONFIG="RBY1NavDoorOpeningSinglePanelSameDoorTestConfig"
        ;;
    open-only:same-door)
        CONFIG="RBY1DoorOpeningSinglePanelSameDoorTestConfig"
        ;;
    nav-open:ten-doors)
        CONFIG="RBY1NavDoorOpeningSinglePanelTenDoorTestConfig"
        ;;
    open-only:ten-doors)
        CONFIG="RBY1DoorOpeningSinglePanelTenDoorTestConfig"
        ;;
    nav-open:all-doors)
        CONFIG="RBY1NavDoorOpeningAllSinglePanelDoorsTestConfig"
        ;;
    open-only:all-doors)
        CONFIG="RBY1DoorOpeningAllSinglePanelDoorsTestConfig"
        ;;
    nav-open:shard-[a-f])
        CONFIG="RBY1NavDoorOpeningAllSinglePanelDoorsTestConfig"
        STABILITY_SHARD="${SCOPE#shard-}"
        ;;
    open-only:shard-[a-f])
        CONFIG="RBY1DoorOpeningAllSinglePanelDoorsTestConfig"
        STABILITY_SHARD="${SCOPE#shard-}"
        ;;
    *)
        echo "Usage: $0 [nav-open|open-only] [same-door|ten-doors|all-doors|shard-a..shard-f] [push|pull] [attempts-per-door]" >&2
        exit 2
        ;;
esac

if [[ "${DOOR_ACTION}" != "push" && "${DOOR_ACTION}" != "pull" ]]; then
    echo "Door action must be either push or pull" >&2
    exit 2
fi

if [[ "$(hostname)" != "kuorobot01" ]]; then
    echo "This launcher is intended for kuorobot01." >&2
    exit 1
fi

PROJECT_ROOT="/u/xna8aw/workspace/live-robotics-lab/LRL_project"
MOLMOSPACES_ROOT="${PROJECT_ROOT}/molmospaces"
PYTHON_ENV="/u/xna8aw/workspace/live-robotics-lab/envs/mlspaces"
CACHE_ROOT="/dev/shm/xna8aw/molmospaces_kuorobot_cache"

mkdir -p \
    "${CACHE_ROOT}/xdg" \
    "${CACHE_ROOT}/torch" \
    "${CACHE_ROOT}/torch_extensions" \
    "${CACHE_ROOT}/matplotlib" \
    "${CACHE_ROOT}/jax" \
    "${CACHE_ROOT}/cuda" \
    "${CACHE_ROOT}/cupy" \
    "${CACHE_ROOT}/numba" \
    "${CACHE_ROOT}/tmp"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PROJECT_ROOT}:${MOLMOSPACES_ROOT}:${PYTHONPATH:-}"
export PATH="${PYTHON_ENV}/bin:${PATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export RBY1_CUROBO_SERVER_URLS=local
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export JAX_COMPILATION_CACHE_DIR="${CACHE_ROOT}/jax"
export CUDA_CACHE_PATH="${CACHE_ROOT}/cuda"
export CUPY_CACHE_DIR="${CACHE_ROOT}/cupy"
export NUMBA_CACHE_DIR="${CACHE_ROOT}/numba"
export TMPDIR="${CACHE_ROOT}/tmp"
export RBY1_DOOR_STABILITY_ATTEMPTS_PER_DOOR="${ATTEMPTS_PER_DOOR}"
export RBY1_DOOR_STABILITY_ACTION="${DOOR_ACTION}"
if [[ -n "${STABILITY_SHARD}" ]]; then
    export RBY1_DOOR_STABILITY_SHARD="${STABILITY_SHARD}"
else
    unset RBY1_DOOR_STABILITY_SHARD
fi

cd "${MOLMOSPACES_ROOT}"

echo "Mode: ${MODE}"
echo "Scope: ${SCOPE}"
echo "Attempts per door: ${ATTEMPTS_PER_DOOR}"
echo "Door action: ${DOOR_ACTION}"
echo "Shard: ${STABILITY_SHARD:-none}"
echo "Config: ${CONFIG}"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader

exec "${PYTHON_ENV}/bin/python" \
    -u \
    -m \
    lrl_molmospaces_extensions.datagen.main \
    "lrl_molmospaces_extensions.configs.door_opening:${CONFIG}"
