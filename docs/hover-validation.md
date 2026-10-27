# Hover setup validation — October 27, 2026

Validated on macOS arm64, Python 3.12.1, CPU only.

- `uv sync --locked --offline`: passes after the initial dependency download.
- `uv run pytest -q`: **3 passed**, with four upstream/checker warnings described in the training guide.
- `uv run ruff check hover tests` and `uv run ruff format --check hover tests`: pass.
- `uv run python -m hover smoke`: finite 13-element observation; position error 0.1133 m after one simulated second of nominal thrust.
- Training, checkpoint serialization/reloading, CSV logging, and standalone checkpoint evaluation all ran successfully.

## Measured policy performance

Seed 0 training; deterministic evaluation on five starting states with seeds 10000–10004.
Episodes allow five simulated seconds. These measurements test ideal-state fitted dynamics,
not noisy sensor feedback or physical hardware. Runs below start independently from the same seed.

| Policy | Mean position error (m) | Mean duration (s) | Crash / boundary exit rate |
| --- | ---: | ---: | ---: |
| Untrained | 0.2772 | 5.000 | 0% |
| PPO, 4,096 steps | 0.6056 | 3.716 | 100% |
| PPO, 32,768 steps | 0.8908 | 3.028 | 100% |

Neither short run converged to a useful hover controller. The untrained policy outputs
near-zero actions, which already request nominal hover thrust and attitude; it is a
strong baseline for these small, upright starting offsets. Exploratory PPO updates can
initially degrade this behavior. The task is ready for longer training and tuning,
but this validation establishes a working experiment pipeline, not successful learned control.

Reproduce with:

```sh
uv run python -m hover train --steps 4096 --output runs/reproduce-4k
uv run python -m hover train --steps 32768 --output runs/reproduce-32k
uv run python -m hover evaluate --checkpoint runs/reproduce-4k/policy.zip
```

The original local checkpoints and raw metrics remain in `runs/hover-smoke/` and
`runs/hover-32k/`. They are deliberately excluded from source control. Full configuration
and versions are recorded in each run directory; the source and lockfile are committed.

Validated package versions:

- crazyflow: 0.4.0
- stable-baselines3: 2.9.2
- gymnasium: 1.4.0
- jax: 0.12.0.1
- torch: 2.14.1

## PARCC validation — October 27, 2026

Linux x86-64, Python 3.12.3, AMD EPYC 9374F CPU nodes and an NVIDIA B200 (driver 580.95.05).
Setup, devices and job scripts: [hpc-setup.md](hpc-setup.md).

- Unchanged commit `36b5ffb` on a CPU node: `uv sync --locked`, `hover smoke` (0.1133 m),
  `pytest -q` (3 passed) and the 4,096-step run reproduce the Mac numbers exactly
  (untrained 0.2772 m / 0%, trained 0.6056 m / 100% crash).
- Branch `hpc-gpu-setup` (`slurm/cpu-check.sbatch`): ruff passes, `pytest -q` 11 passed,
  1 skipped (the GPU test, run in `slurm/gpu-check.sbatch`).

### Why the earlier runs failed: too little data

With 64 batched worlds (`--n-worlds 64 --batch-size 1024`, other PPO settings unchanged),
2M-step runs reached 0.019–0.025 m on 20 tuning episodes (seeds 10000+) with no crashes,
for the default settings, `--log-std-init -2` and `--n-steps 512` alike. No reward, action or
observation scaling change was needed; the 4k–32k-step runs simply had not seen enough
experience. The defaults were kept for the experiments below.

### Results on held-out seeds

PPO on CPU (64 worlds, batch 1024, 8 torch threads, `slurm/experiments.sbatch` on
`genoa-std-mem`), seeds 0–2, each evaluated deterministically on 50 episodes from seed 20000
that were never used for tuning. Requested budgets round up to full 16,384-step rollouts.

| Policy | Steps | Mean position error (m) | Mean duration (s) | Crash / exit rate |
| --- | ---: | ---: | ---: | ---: |
| Zero action | — | 0.3196 | 5.000 | 0% |
| Position PD (kp 4, kd 3) | — | 0.0320 | 5.000 | 0% |
| PPO, 3 seeds | 114,688 | 0.580 ± 0.111 (0.427, 0.624, 0.688) | 4.50 | 2% / 52% / 100% |
| PPO, 3 seeds | 1,015,808 | 0.0347 ± 0.0051 (0.0275, 0.0378, 0.0387) | 5.000 | 0% |

At 1M steps every PPO seed hovers stably on unseen starts and is close to, but on average not
better than, the hand-tuned PD baseline. Steady training throughput was ~19,400–20,200
steps/s per run (three runs sharing 24 CPU cores), 50–52 s of training per 1M-step run and
3.5 min for the whole job including JIT compilation and evaluation. Checkpoints:
`runs/ppo-1m-cpu/seed-{0,1,2}/policy.zip` (and `checkpoints/`) on PARCC `/vast/projects`.

These numbers are ideal-state, fitted attitude dynamics in simulation. They do not show
physical flight readiness.
