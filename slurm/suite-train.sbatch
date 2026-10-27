#!/bin/bash
# Train one suite policy on CPU, then evaluate it on held-out seeds under five conditions
# (see the loop at the end). DYNAMICS=mix trains half the worlds on each physics model.
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
DYNAMICS=${DYNAMICS:-so_rpy}
STEPS=${STEPS:-5000000}
SEED=${SEED:-0}
EPISODES=${EPISODES:-100}
read -r -a train_args <<< "${TRAIN_ARGS:-}"
suite() { uv run --locked --extra gpu python -m hover.suite "$@"; }
RUN=runs/suite/$NAME
mkdir -p "$(dirname "$RUN")"
echo "name=$NAME tasks=[$TASKS] robustness=$ROBUSTNESS dynamics=$DYNAMICS steps=$STEPS seed=$SEED args=[${TRAIN_ARGS:-}]"
if [[ -f $RUN/config.json ]]; then
  suite train --resume "$RUN" --steps "$STEPS"
else
  # shellcheck disable=SC2086
  suite train --output "$RUN" --tasks $TASKS --robustness "$ROBUSTNESS" --steps "$STEPS" \
    --dynamics "$DYNAMICS" --seed "$SEED" "${train_args[@]}"
fi
# Evaluation conditions: robustness, dynamics model, drone. cf2x_P250 is a separately
# identified Crazyflie 2.x that no policy trains on (held-out drone).
for cond in "none so_rpy cf2x_L250" "real so_rpy cf2x_L250" "real first_principles cf2x_L250" \
    "real so_rpy cf2x_P250" "real first_principles cf2x_P250"; do
  read -r rob dyn drone <<< "$cond"
  out=$RUN/final-$rob-$dyn.json
  [[ $drone == cf2x_L250 ]] || out=$RUN/final-$rob-$dyn-$drone.json
  [[ -f $out ]] && continue
  # shellcheck disable=SC2086
  suite evaluate --checkpoint "$RUN/policy.zip" --tasks $TASKS --robustness "$rob" \
    --dynamics "$dyn" --drone "$drone" --episodes "$EPISODES" --output "$out" > /dev/null
done
echo "done: $RUN"
