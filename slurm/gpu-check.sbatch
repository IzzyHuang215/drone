#!/bin/bash
# GPU validation and CPU-vs-GPU benchmarks. Fails if either framework cannot use the GPU.
# Submit from the repository root: mkdir -p runs/slurm && sbatch slurm/gpu-check.sbatch
# PARCC's filter requires exactly 28 CPUs and <=224G per dgx-b200 GPU (14 / 112G on
# b200-mig90, which also needs --qos=mig); check with `sbatch --test-only` first.
#SBATCH --job-name=drone-gpu-check
#SBATCH --account=dbgoodma-goodman-laboratory
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:1
#SBATCH -c 28
#SBATCH --mem=224G
#SBATCH -t 02:00:00
#SBATCH --output=runs/slurm/%x-%j.out
source slurm/common.sh
OUT=runs/gpu-check-${SLURM_JOB_ID}
mkdir -p "$OUT"
nvidia-smi | tee "$OUT/nvidia-smi.txt"
hover devices --sim-device gpu --policy-device cuda | tee "$OUT/devices.json"
uv run --locked --extra gpu pytest -q -rs

for device in cpu gpu; do
  for n in 1 64 1024 16384; do
    hover benchmark --sim-device "$device" --n-worlds "$n" --steps $((n < 64 ? 5000 : 500000))
  done
done | tee "$OUT/benchmark.jsonl"

# Short PPO runs to compare end-to-end training throughput (reported as
# throughput.steady_steps_per_second in each evaluation.json, excluding the first rollout).
train() {
  local name=$1; shift
  hover train --output "$OUT/train-$name" --checkpoint-every 0 --torch-threads 8 "$@"
}
train single-cpu-cpu --steps 16384
train single-gpu-cpu --steps 16384 --sim-device gpu
vec=(--n-steps 64 --batch-size 4096 --steps 262144)
train vec256-cpu-cpu --n-worlds 256 "${vec[@]}"
train vec256-gpu-cpu --n-worlds 256 "${vec[@]}" --sim-device gpu
train vec256-cpu-cuda --n-worlds 256 "${vec[@]}" --policy-device cuda
train vec256-gpu-cuda --n-worlds 256 "${vec[@]}" --sim-device gpu --policy-device cuda
train vec4096-gpu-cuda --n-worlds 4096 --n-steps 16 --batch-size 16384 --steps 1048576 \
  --sim-device gpu --policy-device cuda
nvidia-smi --query-gpu=memory.used --format=csv
for run in "$OUT"/train-*; do
  printf '%s ' "$(basename "$run")"
  python3 -c 'import json,sys; t=json.load(open(sys.argv[1]))["throughput"]; print(round(t["steady_steps_per_second"]), round(t["wall_seconds"], 1))' "$run/evaluation.json"
done | tee "$OUT/train-throughput.txt"
