#!/bin/bash
#SBATCH --job-name=drone-flow-tune
#SBATCH --account=dbgoodma-goodman-laboratory
#SBATCH --partition=genoa-std-mem
#SBATCH -c 8
#SBATCH --mem=44G
#SBATCH -t 02:00:00
#SBATCH --output=runs/slurm/%x-%j.out
source slurm/common.sh
export JAX_PLATFORMS=cpu PYTHONPATH=.
uv run --locked --extra gpu python experiments/flow_plans/tune_pid_ff.py
