#!/bin/bash
# Multi-seed PPO experiments inside one allocation. Resubmitting the same command resumes
# every unfinished run from its latest checkpoint, so a timed-out job loses at most
# --checkpoint-every steps per run.
#
#   mkdir -p runs/slurm
#   EXPERIMENT=ppo-1m STEPS=1000000 SEEDS="0 1 2" sbatch slurm/experiments.sbatch
#
# Extra `hover train` flags go in TRAIN_ARGS. Training reports on tuning seeds 10000+; the
# final table uses held-out seeds starting at FINAL_SEED that tuning never sees.
#SBATCH --job-name=drone-experiments
#SBATCH --account=dbgoodma-goodman-laboratory
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:1
#SBATCH -c 28
#SBATCH --mem=224G
#SBATCH -t 2-00:00:00
#SBATCH --output=runs/slurm/%x-%j.out
source slurm/common.sh
: "${EXPERIMENT:?set EXPERIMENT to a run-group name}"
STEPS=${STEPS:-1000000}
SEEDS=${SEEDS:-0 1 2}
FINAL_SEED=${FINAL_SEED:-20000}
FINAL_EPISODES=${FINAL_EPISODES:-50}
read -r -a train_args <<< "${TRAIN_ARGS:-}"
ROOT=runs/$EXPERIMENT
mkdir -p "$ROOT"
echo "steps=$STEPS seeds=[$SEEDS] train_args=[${TRAIN_ARGS:-}]" | tee -a "$ROOT/jobs.log"

for seed in $SEEDS; do
  run=$ROOT/seed-$seed
  if [[ -f $run/config.json ]]; then
    args=(--resume "$run" --steps "$STEPS")
  else
    args=(--output "$run" --steps "$STEPS" --seed "$seed" "${train_args[@]}")
  fi
  hover train "${args[@]}" > "$ROOT/seed-$seed.log" 2>&1 &
done
status=0
for job in $(jobs -p); do wait "$job" || status=1; done
[[ $status == 0 ]] || { echo "a training run failed; see $ROOT/seed-*.log"; exit 1; }

final=(--eval-first-seed "$FINAL_SEED" --eval-episodes "$FINAL_EPISODES")
hover baselines "${final[@]}" > "$ROOT/final-baselines.json"
for seed in $SEEDS; do
  hover evaluate --checkpoint "$ROOT/seed-$seed/policy.zip" "${final[@]}" \
    > "$ROOT/seed-$seed/final-evaluation.json"
done
uv run --locked --extra gpu python -m hover.report "$ROOT" | tee "$ROOT/results.md"
