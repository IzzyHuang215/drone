# Handoff: Crazyflow hover experiments on NVIDIA HPC

## Goal and starting point

Set up this repo on the user's HPC, verify NVIDIA acceleration, and run reproducible
hover-control experiments before any hardware purchase.

Repo: https://github.com/IzzyHuang215/drone
Starting implementation: `d83a205` on `main` (pull current main).
Read `docs/hover-training.md` and `docs/hover-validation.md` first.

Current stack: Python 3.12, uv + `uv.lock`, Crazyflow 0.4.0, Stable-Baselines3 2.9.2 PPO.
`hover/env.py` implements the task; `hover/__main__.py` implements smoke/train/evaluate.
**Both simulator and PPO are explicitly CPU-only today. Installing CUDA packages alone
will not enable GPU training.** There is one simulator world, NumPy observations,
and host/device transfers every step; the current loop is not a batched GPU trainer.

## First actions on the cluster

1. Inspect the cluster's scheduler, GPU partition/account, allowed job durations,
   scratch storage, network access, and environment modules. Use an allocated compute
   node for tests/training; do not run training on the login node. Obtain missing
   account/partition details from the user rather than guessing them.
2. Record `uname -m`, `nvidia-smi`, Python, driver and GPU model/VRAM inside that
   allocation. Determine a supported JAX/CUDA combination from the official docs below.
   A CUDA version shown by `nvidia-smi` is a driver capability, not proof that the
   corresponding toolkit or Python runtime is installed.
3. Clone/pull the repo, then establish the unchanged CPU baseline:

   ```sh
   uv sync --locked
   uv run python -m hover smoke
   uv run pytest -q
   uv run python -m hover train --steps 4096 --output runs/hpc-cpu-check
   ```

4. Add an optional, platform-appropriate GPU dependency configuration and update the
   lockfile reproducibly. Keep the Mac CPU setup working. Crazyflow documents its
   GPU extra as `crazyflow[gpu]` with CUDA 12 on Linux x86-64; verify compatibility
   with the actual cluster and locked JAX version before selecting packages.
   Avoid an untracked pip overlay that the next `uv sync` silently removes.
5. Inside the GPU job, verify `jax.devices()` and `torch.cuda.is_available()` independently.
   Fail clearly if requested acceleration is unavailable. Test an actual Crazyflow
   `Sim(..., device="gpu")` rollout with finite states, not just device enumeration.

## Implementation work needed

- Add configurable simulator and policy devices, preserving CPU defaults and recording
  the selected devices in `config.json`. Current CPU selections appear in environment
  construction, PPO construction/loading, and configuration output.
- Benchmark before assuming a GPU helps: this tiny PPO network may be faster on CPU.
  Moving a single world to GPU while copying each observation to NumPy can be slower.
- For simulation throughput, expose Crazyflow's batched worlds and implement a correct
  vector-environment adapter, or use a suitable upstream JAX training path. Preserve
  per-world resets, terminal observations, and time-limit bootstrapping. Do not treat
  one vector of worlds as one Gymnasium environment. Keep the simple CPU path usable.
- If JAX and PyTorch share a GPU, inspect their memory allocation behavior and prevent
  competing preallocation; document any environment settings used.
- Add scheduler job scripts for the actual cluster, durable experiment/log directories,
  and periodic checkpoint/resume support before long jobs. The current CLI saves only
  at the end and has no resume flag; evaluation's `--checkpoint` does not resume training.
- Extend tests for the new device/vector behavior; run CPU tests plus a GPU smoke test.

## Experiments and success criteria

The current task is fitted `so_rpy` attitude-response dynamics for `cf2x_L250`, not raw
motor control. Target `(0,0,1)` m; 100 Hz actions / 500 Hz simulation; five-second episodes;
ideal-state observations and small randomized upright starts. No hardware interface.

Local seed-0 results on five evaluation seeds (10000–10004):

| Policy | Mean position error | Crash / boundary exit rate |
| --- | ---: | ---: |
| Untrained | 0.2772 m | 0% |
| PPO, 4,096 steps | 0.6056 m | 100% |
| PPO, 32,768 steps | 0.8908 m | 100% |

Neither learned policy converged. Zero actions already request nominal hover thrust
and attitude, so the untrained policy is a useful baseline. GPU availability is not a
fix for poor learning. Check action scaling, reward/termination incentives, observation
scaling, rollout length, and exploration before spending a large compute allocation.

After the short validation, start with 100k–1M steps and at least three training seeds,
within the user's allocation budget. Compare against zero-action hover and a classical
position-feedback baseline. Evaluate on separate, repeatable initial states; reserve
additional seeds for final reporting rather than tuning repeatedly on the same five.
Report position error, episode duration, crash rate, throughput excluding initial JAX
compilation, total wall time, versions, devices, settings, and checkpoint paths.
Do not label a run successful based on reward alone or claim physical flight readiness.

Deliver: reproducible GPU setup, working job script, checkpoint/resume instructions,
validated device placement, and a concise experiment-results table. Keep large outputs
in durable cluster storage (`runs/` is Git-ignored). Local Mac checkpoints are not in
Git; reproduce them if needed. Commit source/docs and report the branch/commit and
commands to the user; do not change hardware plans as part of this setup.

## Official references

- [Crazyflow installation and GPU support](https://learnsyslab.github.io/crazyflow/get-started/installation/)
- [JAX installation and driver requirements](https://docs.jax.dev/en/latest/installation.html)
