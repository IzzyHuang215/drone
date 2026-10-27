#!/bin/bash
# Fly one planner seed's plans with every controller and condition (outputs that exist are kept).
#   PLAN_SEED=0 CONTROLLERS="pid generalist-s0" sbatch experiments/flow_plans/execute.sbatch
#SBATCH --job-name=drone-flow-exec
#SBATCH --account=dbgoodma-goodman-laboratory
#SBATCH --partition=genoa-std-mem
#SBATCH -c 8
#SBATCH --mem=44G
#SBATCH -t 12:00:00
#SBATCH --output=runs/slurm/%x-%j.out
source slurm/common.sh
export JAX_PLATFORMS=cpu PYTHONPATH=.
: "${PLAN_SEED:?}"
CONTROLLERS=${CONTROLLERS:-pid pid-ff generalist-s0 generalist-s1 generalist-s2 path-s0 path-s1 path-s2}
PLANS=${PLANS:-runs/flow/plans}  # PLANS=runs/flow/plans-feasible OUT=runs/flow/exec-feasible
OUT=${OUT:-runs/flow/exec}
CONDITIONS=${CONDITIONS:-nominal realistic held-out}
mkdir -p "$OUT"
for controller in $CONTROLLERS; do
  case $controller in
    generalist-s*) ckpt=(--checkpoint runs/suite/generalist-robust/all-${controller#generalist-}/policy.zip) ;;
    path-s*) ckpt=(--checkpoint runs/suite/path/specialist-robust-${controller#path-}/policy.zip) ;;
    *) ckpt=() ;;
  esac
  for condition in $CONDITIONS; do
    out=$OUT/$controller-$condition-p$PLAN_SEED.csv
    [[ -f $out ]] && continue
    uv run --locked --extra gpu python -m hover.suite.plans --plans "$PLANS" --seed-file "$PLAN_SEED" \
      --controller "$controller" "${ckpt[@]}" --condition "$condition" --output "$out" \
      --per-plan "${out%.csv}.npz" > /dev/null
  done
done
echo done
