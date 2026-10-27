# Shared setup for the PARCC job scripts; source from the repository root.
set -euo pipefail
# Compute nodes cannot install packages: run `uv sync --locked --extra gpu` on a login
# node first. Every uv command passes --extra gpu, since syncing without it would
# uninstall the JAX CUDA plugin from the shared .venv.
export UV_OFFLINE=1
# JAX and PyTorch may share one GPU: allocate on demand instead of JAX's 75% preallocation
# (hover/__init__.py sets the same default).
export XLA_PYTHON_CLIENT_PREALLOCATE=false
hover() { uv run --locked --extra gpu python -m hover "$@"; }
echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date -Is); commit $(git rev-parse HEAD)"
uname -m
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "no GPU visible"
