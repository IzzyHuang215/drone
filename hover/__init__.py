"""Simulation-only hover training tools."""

import os

# JAX otherwise reserves most GPU memory on first use, starving PyTorch on a shared GPU.
# Set before any submodule imports JAX; an explicit user setting still wins.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
