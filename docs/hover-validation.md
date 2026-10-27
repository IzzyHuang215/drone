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
