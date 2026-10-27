#!/bin/bash
# Render one GIF per task: POLICIES/<task>-s0/policy.zip (default specialist-robust) flying on
# the held-out drone with realistic noise, plus the PID baseline for comparison.
#SBATCH --job-name=drone-suite-videos
#SBATCH --account=dbgoodma-goodman-laboratory
#SBATCH --partition=genoa-std-mem
#SBATCH -c 8
#SBATCH --mem=44G
#SBATCH -t 04:00:00
#SBATCH --output=runs/slurm/%x-%j.out
source slurm/common.sh
export JAX_PLATFORMS=cpu MUJOCO_GL=egl PYOPENGL_PLATFORM=egl  # offscreen; Mesa works without a GPU
GROUP=${GROUP:-specialist-robust}
OUT=runs/suite/videos/$GROUP
mkdir -p "$OUT"
common=(--robustness real --drone cf2x_P250 --dynamics "${DYNAMICS:-so_rpy}" --fps 20 --size 480 360)
for task in ${TASKS:-hover hover-wide reach recover disturbance circle figure8 landing waypoints}; do
  policy=runs/suite/$GROUP/$task-s0/policy.zip
  [[ $GROUP == generalist-robust ]] && policy=runs/suite/$GROUP/all-s0/policy.zip
  uv run --locked --extra gpu python -m hover.suite render --checkpoint "$policy" --tasks "$task" \
    "${common[@]}" --video "$OUT/$task.gif" > "$OUT/$task.json"
  uv run --locked --extra gpu python -m hover.suite render --baseline pid --tasks "$task" \
    "${common[@]}" --video "$OUT/$task-pid.gif" > "$OUT/$task-pid.json"
done
echo done
