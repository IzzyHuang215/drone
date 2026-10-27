#!/bin/bash
# Train one suite policy on CPU, then evaluate it on held-out seeds under four conditions:
# {no, realistic} robustness x {training model so_rpy, unseen first_principles model}.
# Resubmitting with the same NAME resumes from the latest checkpoint.
#
#   mkdir -p runs/slurm
#   NAME=specialist/circle-s0 TASKS=circle STEPS=10000000 sbatch slurm/suite-train.sbatch
#   NAME=generalist/real-s0 TASKS=all ROBUSTNESS=real STEPS=60000000 \
#     TRAIN_ARGS="--init-from runs/suite/generalist/none-s0/policy.zip" sbatch slurm/suite-train.sbatch
#SBATCH --job-name=drone-suite
#SBATCH --account=dbgoodma-goodman-laboratory
#SBATCH --partition=genoa-std-mem
#SBATCH -c 8
#SBATCH --mem=44G
#SBATCH -t 24:00:00
#SBATCH --output=runs/slurm/%x-%j.out
source slurm/common.sh
export JAX_PLATFORMS=cpu  # simulation and policy both run on CPU (see docs/hpc-setup.md)
: "${NAME:?set NAME, e.g. specialist/circle-s0}" "${TASKS:?set TASKS (names or all)}"
ROBUSTNESS=${ROBUSTNESS:-none}
STEPS=${STEPS:-5000000}
SEED=${SEED:-0}
EPISODES=${EPISODES:-100}
read -r -a train_args <<< "${TRAIN_ARGS:-}"
suite() { uv run --locked --extra gpu python -m hover.suite "$@"; }
RUN=runs/suite/$NAME
mkdir -p "$(dirname "$RUN")"
echo "name=$NAME tasks=[$TASKS] robustness=$ROBUSTNESS steps=$STEPS seed=$SEED args=[${TRAIN_ARGS:-}]"
if [[ -f $RUN/config.json ]]; then
  suite train --resume "$RUN" --steps "$STEPS"
else
  # shellcheck disable=SC2086
  suite train --output "$RUN" --tasks $TASKS --robustness "$ROBUSTNESS" --steps "$STEPS" \
    --seed "$SEED" "${train_args[@]}"
fi
for rob in none real; do
  for dyn in so_rpy first_principles; do
    # shellcheck disable=SC2086
    suite evaluate --checkpoint "$RUN/policy.zip" --tasks $TASKS --robustness "$rob" \
      --dynamics "$dyn" --episodes "$EPISODES" --output "$RUN/final-$rob-$dyn.json" > /dev/null
  done
done
echo "done: $RUN"
