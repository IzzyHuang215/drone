#!/bin/bash
# Export plans from the sibling ese6180-generative-flows checkout (read-only). Submit from the
# drone repository root: sbatch experiments/flow_plans/export.sbatch
#SBATCH --job-name=drone-flow-export
#SBATCH --account=dbgoodma-goodman-laboratory
#SBATCH --partition=genoa-std-mem
#SBATCH -c 16
#SBATCH --mem=88G
#SBATCH -t 04:00:00
#SBATCH --output=runs/slurm/%x-%j.out
set -euo pipefail
DRONE=$PWD
FLOWS=${FLOWS:-$DRONE/../ese6180-generative-flows}
cd "$FLOWS"
echo "flows commit $(git rev-parse HEAD)"
# SCRIPT=export_feasible.py OUT=plans-feasible for the feasibility-guided plan set.
UV_OFFLINE=1 uv run --no-sync python "$DRONE/experiments/flow_plans/${SCRIPT:-export_plans.py}" \
  --out "$DRONE/runs/flow/${OUT:-plans}" --paths "${PATHS:-512}"
