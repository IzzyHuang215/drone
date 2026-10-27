#!/bin/bash
# Submit the whole portfolio. Safe to rerun: finished runs are skipped, unfinished ones resume.
#
#   bash slurm/suite-campaign.sh            # everything
#   SEEDS="0" bash slurm/suite-campaign.sh  # one seed
#
# Stages (warm starts are SLURM dependencies):
#   specialist/<task>-s<seed>        nominal, from scratch
#   specialist-real/<task>-s<seed>   realistic robustness, warm-started from the nominal policy
#   generalist/all-s<seed>           every task in one policy, nominal
#   generalist-real/all-s<seed>      generalist fine-tuned with realistic robustness
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/slurm
SEEDS=${SEEDS:-0 1 2}
declare -A STEPS=(
  [hover]=3000000 [hover-wide]=5000000 [reach]=5000000 [recover]=5000000
  [disturbance]=5000000 [circle]=10000000 [figure8]=10000000 [landing]=5000000
  [waypoints]=10000000
)
GENERALIST_STEPS=${GENERALIST_STEPS:-40000000}

submit() {  # submit NAME TASKS ROBUSTNESS STEPS SEED DEPENDENCY TRAIN_ARGS -> prints job id
  local name=$1 tasks=$2 rob=$3 steps=$4 seed=$5 dep=$6 args=$7
  if [[ -f runs/suite/$name/final-real-first_principles.json ]]; then
    return 0
  fi
  local job_name=suite-${name//\//-} queued
  queued=$(squeue -u "$USER" -h -n "$job_name" -o %i | head -1)
  if [[ -n $queued ]]; then  # already queued or running: reuse it as the dependency
    echo "$queued"
    return 0
  fi
  local opts=(--parsable -J "$job_name")
  [[ -n $dep ]] && opts+=(--dependency="afterok:$dep")
  NAME=$name TASKS=$tasks ROBUSTNESS=$rob STEPS=$steps SEED=$seed TRAIN_ARGS=$args \
    sbatch "${opts[@]}" slurm/suite-train.sbatch 2>/dev/null | tail -1
}

for seed in $SEEDS; do
  for task in "${!STEPS[@]}"; do
    base=specialist/$task-s$seed
    job=$(submit "$base" "$task" none "${STEPS[$task]}" "$seed" "" "")
    echo "$base ${job:-done}"
    real=specialist-real/$task-s$seed
    job=$(submit "$real" "$task" real "${STEPS[$task]}" "$seed" "$job" \
      "--init-from runs/suite/$base/policy.zip")
    echo "$real ${job:-done}"
  done
  base=generalist/all-s$seed
  job=$(submit "$base" all none "$GENERALIST_STEPS" "$seed" "" "--n-worlds 256 --batch-size 4096")
  echo "$base ${job:-done}"
  real=generalist-real/all-s$seed
  job=$(submit "$real" all real "$GENERALIST_STEPS" "$seed" "$job" \
    "--n-worlds 256 --batch-size 4096 --init-from runs/suite/$base/policy.zip")
  echo "$real ${job:-done}"
done
