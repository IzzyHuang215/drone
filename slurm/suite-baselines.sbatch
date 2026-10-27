#!/bin/bash
# PID and zero-action baselines on every suite task under the four evaluation conditions.
#SBATCH --job-name=drone-suite-baselines
#SBATCH --account=dbgoodma-goodman-laboratory
#SBATCH --partition=genoa-std-mem
#SBATCH -c 8
#SBATCH --mem=44G
#SBATCH -t 04:00:00
#SBATCH --output=runs/slurm/%x-%j.out
source slurm/common.sh
export JAX_PLATFORMS=cpu
OUT=runs/suite/baselines
mkdir -p "$OUT"
for baseline in pid zero; do
  for rob in none real; do
    for dyn in so_rpy first_principles; do
      uv run --locked --extra gpu python -m hover.suite evaluate --baseline "$baseline" \
        --tasks all --robustness "$rob" --dynamics "$dyn" --episodes "${EPISODES:-100}" \
        --output "$OUT/$baseline-$rob-$dyn.json" > /dev/null
    done
  done
done
echo done
