#!/bin/bash
# PID and zero-action baselines on every suite task under the five evaluation conditions.
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
  for cond in "none so_rpy cf2x_L250" "real so_rpy cf2x_L250" "real first_principles cf2x_L250" \
      "real so_rpy cf2x_P250" "real first_principles cf2x_P250"; do
    read -r rob dyn drone <<< "$cond"
    out=$OUT/$baseline-$rob-$dyn.json
    [[ $drone == cf2x_L250 ]] || out=$OUT/$baseline-$rob-$dyn-$drone.json
    [[ -f $out ]] && continue
    uv run --locked --extra gpu python -m hover.suite evaluate --baseline "$baseline" \
      --tasks all --robustness "$rob" --dynamics "$dyn" --drone "$drone" \
      --episodes "${EPISODES:-100}" --output "$out" > /dev/null
  done
done
echo done
