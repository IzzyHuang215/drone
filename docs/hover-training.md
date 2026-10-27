# Train a hovering policy in Crazyflow

This workflow is simulation-only and needs no drone, radio, or localization hardware.
It uses Crazyflow 0.4.0 and Stable-Baselines3 2.9.2, with all transitive dependencies
pinned in `uv.lock`. Python 3.12 is selected by `.python-version`.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the repo root:

```sh
uv sync --locked
uv run python -m hover smoke
uv run pytest -q
```

The first run compiles JAX kernels and may take longer. The default is headless CPU
execution, tested on macOS arm64. Training explicitly uses PyTorch CPU, not MPS.
NVIDIA GPU support (Linux x86-64 only) is an optional extra; see [hpc-setup.md](hpc-setup.md).
For much faster CPU training, use batched worlds (`--n-worlds 64 --batch-size 1024`).
No simulator source checkout or hardware firmware is needed.

MuJoCo may print notices about missing optional `warp` / `mujoco_warp` packages;
these are not required for this CPU workflow. Its model-attachment warning also
appears with the locked upstream versions. The environment checker warns about
unbounded observation limits because velocity and angular velocity are not artificially
clipped. These warnings were present in the passing validation run.

## Training

```sh
# Short pipeline check; this is not enough to claim a learned hover controller.
uv run python -m hover train --steps 4096 --output runs/hover-smoke

# Starting point for a longer experiment; increase the budget after checking metrics.
uv run python -m hover train --steps 100000 --output runs/hover-100k

# Reload the saved policy and test on a separate, repeatable set of starting states.
uv run python -m hover evaluate --checkpoint runs/hover-100k/policy.zip
```

Each training directory contains `policy.zip`, `config.json` (arguments, package
versions and PPO settings), `progress.csv`, and `evaluation.json` (untrained and
trained policies evaluated on the same seeds). Outputs are ignored by Git.
Use a new output directory for each experiment. PPO rounds requested steps up to
complete 256-step rollouts; actual steps are recorded in evaluation results.

Evaluation uses deterministic actions and seeds starting at 10000, separate from the
training seed (default 0). It reports mean Euclidean position error, episode return,
mean duration, and crash / boundary-exit rate. Look for lower position error and
longer, crash-free episodes across several seeds; PPO reward alone is insufficient.
A 4096-step check may perform worse than its untrained baseline.

## Task definition

- Drone: brushed Crazyflie `cf2x_L250`; this does not commit the hardware choice.
- Dynamics: Crazyflow's fitted second-order attitude-response model (`so_rpy`).
  It approximates stabilized attitude response rather than learning raw motor control.
- Physics: 500 Hz; policy: 100 Hz; episodes: 5 simulated seconds.
- Target: fixed world position `(0, 0, 1)` metres.
- Reset: upright at target plus independent uniform ±0.2 m offsets and ±0.1 m/s
  velocity; repeated seeds reproduce the same starting state.
- Observation (13 floats): position minus target, quaternion (xyzw), world velocity,
  and body angular velocity. These are ideal simulator states, not noisy sensors.
- Actions (4 floats in `[-1, 1]`): roll, pitch and yaw targets bounded to ±0.35 rad,
  and collective thrust bounded by the drone model's per-motor limits times four.
  Zero thrust action maps to gravity-compensating nominal hover thrust.
- Reward per step: `1 - position_error² - 0.05*speed² - 0.01*sum(action²)`,
  with an additional -10 penalty on termination.
- Termination: altitude below 0.05 m or position error above 2 m; time limit is a
  truncation so PPO can bootstrap correctly. Non-finite simulation states fail loudly.

The task has no obstacle perception, hardware interface, domain randomization, motor
latency, or sensor noise. Start with this baseline, then add more realistic dynamics
and held-out disturbances before considering physical transfer.

## Upstream references

- [Crazyflow installation](https://learnsyslab.github.io/crazyflow/get-started/installation/)
- [Crazyflow source](https://github.com/learnsyslab/crazyflow)
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

## Recorded validation

See [the validation report](hover-validation.md) for the tested commands and measured results.
