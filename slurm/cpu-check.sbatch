#!/bin/bash
# CPU validation: lint, tests, baselines and simulator throughput.
# Submit from the repository root: mkdir -p runs/slurm && sbatch slurm/cpu-check.sbatch
#SBATCH --job-name=drone-cpu-check
#SBATCH --account=dbgoodma-goodman-laboratory
#SBATCH --partition=genoa-std-mem
#SBATCH -c 8
#SBATCH --mem=44G
#SBATCH -t 01:00:00
#SBATCH --output=runs/slurm/%x-%j.out
source slurm/common.sh
OUT=runs/cpu-check-${SLURM_JOB_ID}
mkdir -p "$OUT"
uv run --locked --extra gpu ruff check hover tests
uv run --locked --extra gpu ruff format --check hover tests
uv run --locked --extra gpu pytest -q
hover smoke
hover baselines | tee "$OUT/baselines.json"
for n in 1 16 256 4096; do
  hover benchmark --sim-device cpu --n-worlds "$n" --steps $((n < 256 ? 5000 : 200000))
done | tee "$OUT/benchmark.jsonl"
