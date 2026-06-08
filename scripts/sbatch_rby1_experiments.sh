#!/usr/bin/env bash
set -euo pipefail

# Submit multiple RBY1 evaluation YAML configs to Slurm.
#
# Usage:
#   bash scripts/sbatch_rby1_experiments.sh \
#     configs/rby1_pick_pnp_binary.yaml \
#     configs/rby1_pick_pnp_horizon450.yaml \
#     configs/rby1_pick_pnp_horizon450_binary.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <config.yaml> [config.yaml ...]" >&2
  exit 2
fi

for config in "$@"; do
  echo "Submitting RBY1 eval config: $config"
  bash "$SCRIPT_DIR/sbatch_rby1_eval.sh" "$config"
done
