#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"

case "${MODE}" in
    smoke)
        CONFIG="RBY1NavDoorOpeningSimpleVisibleSmokeConfig"
        ;;
    production-a)
        CONFIG="RBY1NavDoorOpeningSimpleVisibleProductionAConfig"
        ;;
    production-b)
        CONFIG="RBY1NavDoorOpeningSimpleVisibleProductionBConfig"
        ;;
    production-c)
        CONFIG="RBY1NavDoorOpeningSimpleVisibleProductionCConfig"
        ;;
    production-d)
        CONFIG="RBY1NavDoorOpeningSimpleVisibleProductionDConfig"
        ;;
    production-e)
        CONFIG="RBY1NavDoorOpeningSimpleVisibleProductionEConfig"
        ;;
    production-f)
        CONFIG="RBY1NavDoorOpeningSimpleVisibleProductionFConfig"
        ;;
    production-all | production-b-to-f)
        CONFIG=""
        ;;
    *)
        echo "Usage: $0 [smoke|production-a|production-b|production-c|production-d|production-e|production-f|production-all|production-b-to-f]" >&2
        exit 2
        ;;
esac

if [[ "$(hostname)" != "kuorobot01" ]]; then
    echo "This launcher is intended for kuorobot01." >&2
    exit 1
fi

if [[ "${MODE}" == "production-all" || "${MODE}" == "production-b-to-f" ]]; then
    if [[ "${MODE}" == "production-all" ]]; then
        SHARDS=(a b c d e f)
    else
        SHARDS=(b c d e f)
    fi

    for shard in "${SHARDS[@]}"; do
        echo "Starting production shard ${shard^^}."
        set +e
        "${BASH_SOURCE[0]}" "production-${shard}"
        status=$?
        set -e
        if [[ ${status} -ne 0 ]]; then
            echo "Production shard ${shard^^} exited with status ${status}; continuing."
        fi
    done
    exit 0
fi

PROJECT_ROOT="/u/xna8aw/workspace/live-robotics-lab/LRL_project"
MOLMOSPACES_ROOT="${PROJECT_ROOT}/molmospaces"
PYTHON_ENV="/u/xna8aw/workspace/live-robotics-lab/envs/mlspaces"
CACHE_ROOT="/dev/shm/xna8aw/molmospaces_kuorobot_cache"
SMOKE_TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-1800}"

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

export CUDA_VISIBLE_DEVICES=0
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

cd "${MOLMOSPACES_ROOT}"

echo "Host: $(hostname)"
echo "Mode: ${MODE}"
echo "Config: ${CONFIG}"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
df -h /dev/shm

COMMAND=(
    "${PYTHON_ENV}/bin/python"
    -u
    -m
    lrl_molmospaces_extensions.datagen.main
    "lrl_molmospaces_extensions.configs.door_opening:${CONFIG}"
)

if [[ "${MODE}" == "smoke" ]]; then
    exec timeout \
        --signal=INT \
        --kill-after=60s \
        "${SMOKE_TIMEOUT_SECONDS}" \
        "${COMMAND[@]}"
fi

exec "${COMMAND[@]}"
