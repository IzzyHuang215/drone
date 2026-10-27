# Running hover experiments on PARCC (NVIDIA B200)

Setup recorded 2026-10-27 on Penn's PARCC cluster. The Mac CPU workflow in
[hover-training.md](hover-training.md) is unchanged: `uv sync --locked` without extras.

## Cluster facts

| Item | Value |
| --- | --- |
| Scheduler | SLURM; account `dbgoodma-goodman-laboratory` |
| GPU partitions | `dgx-b200` (full B200, 183 GB, 7-day max), `b200-mig90`, `b200-mig45` (MIG slices, 2-day max) |
| CPU partitions | `genoa-std-mem` (AMD EPYC 9374F), `genoa-lrg-mem`, `turin` |
| GPU driver | 580.95.05, CUDA 13.0 capability (dgx008) |
| Architecture / Python | x86_64, Python 3.12.3 (uv-managed) |
| Network | login nodes have internet; compute nodes cannot install packages |

PARCC's submit filter requires *exactly* 28 CPUs and at most 224G per `dgx-b200` GPU
(14 / 112G on `b200-mig90`, 6 / 48G on `b200-mig45`). `b200-mig90` also needs `--qos=mig`.
Dry-run with `sbatch --test-only ...`; its start estimates have been far too pessimistic.
CPU nodes ship an older `libcuda` (CUDA 12.8) and print a harmless "driver too old"
warning from PyTorch; it does not describe the GPU nodes.

## Install (login node)

```sh
git clone git@github.com:IzzyHuang215/drone.git && cd drone
git switch hpc-gpu-setup
uv sync --locked --extra gpu
```

The `gpu` extra adds `jax[cuda13]==0.12.0.1` (Linux x86-64 only) to `uv.lock`. CUDA 13
matches the runtime wheels that the locked Linux PyTorch 2.14.1 build already installs,
so JAX and PyTorch share one set of NVIDIA libraries. Crazyflow's own `gpu` extra selects
CUDA 12; it is not used because PyTorch 2.14.1 has no CUDA 12.8+ wheel for Blackwell.
**Always pass `--extra gpu` to `uv run` on the cluster** (the job scripts do): a plain
`uv run`/`uv sync` makes the environment exact and uninstalls the JAX CUDA plugin.

Memory: `hover/__init__.py` defaults `XLA_PYTHON_CLIENT_PREALLOCATE=false`, so JAX allocates
GPU memory on demand rather than reserving 75% and starving PyTorch. PyTorch's caching
allocator also grows on demand. Both values are recorded in each run's `config.json`.

## Validation commands

```sh
mkdir -p runs/slurm
sbatch slurm/cpu-check.sbatch   # lint, tests, baselines, CPU simulator throughput
sbatch slurm/gpu-check.sbatch   # device checks, tests incl. GPU, CPU-vs-GPU benchmarks
```

`python -m hover devices --sim-device gpu --policy-device cuda` checks `jax.devices()` and
`torch.cuda.is_available()` independently, runs a batched Crazyflow rollout on the GPU and
fails if any requested device is missing or states are non-finite. Requesting
`--sim-device gpu` or `--policy-device cuda` anywhere else also fails instead of silently
falling back to CPU.

## Devices and batched worlds

`train`, `evaluate`, `smoke`, `baselines` and `benchmark` accept:

- `--sim-device {cpu,gpu}`: where Crazyflow's JAX state lives (default `cpu`).
- `--policy-device {cpu,cuda}`: PyTorch device for PPO (default `cpu`).
- `--n-worlds N`: `1` (default) keeps the original single Gymnasium env; `N > 1` uses
  `hover.vec_env.HoverVecEnv`, one Stable-Baselines3 `VecEnv` over Crazyflow's batched
  worlds. Each world is its own episode: finished worlds reset in the same step, the final
  observation goes to `info["terminal_observation"]`, and `info["TimeLimit.truncated"]` is
  set only for time limits, so PPO bootstraps truncations but not crashes. Reset states come
  from a seeded JAX key (reproducible, but a different stream from the single env).
- PPO settings: `--n-steps` (per world), `--batch-size`, `--n-epochs`, `--learning-rate`,
  `--log-std-init`, plus `--torch-threads`.

Evaluation always uses the single env with deterministic actions on repeatable seeds
(`--eval-first-seed`, default 10000, `--eval-episodes`). Seeds 10000+ are for tuning;
final tables use 20000+.

## Checkpoints and resume

Training saves `checkpoints/hover_<steps>_steps.zip` every `--checkpoint-every` steps
(default 100000, `0` disables) and `policy.zip` at the end. To continue an interrupted run,
or to extend its budget:

```sh
uv run --extra gpu python -m hover train --resume runs/<run> [--steps <new total>]
```

Resume restores settings from `config.json`, loads the newest checkpoint (starting over if
none was written), reseeds resets with `seed + steps` and logs to
`progress.resume-<steps>.csv` (SB3 would truncate `progress.csv`). Each resume is appended to
`config.json["resumes"]`. Resumed runs are not bit-identical to uninterrupted ones. A finished
run (with `evaluation.json`) is left alone. `evaluate --checkpoint` only evaluates.

## Experiments

```sh
EXPERIMENT=ppo-1m STEPS=1000000 SEEDS="0 1 2" TRAIN_ARGS="--n-worlds 64 ..." \
  sbatch slurm/experiments.sbatch
```

The script trains each seed in parallel inside one allocation, resumes unfinished runs if
resubmitted with the same `EXPERIMENT`, evaluates every policy and both baselines on 50 held-out
episodes from seed 20000 and writes `runs/<experiment>/results.md` via `python -m hover.report`.

## Measured: does the GPU help? (dgx013, job 9637992)

`slurm/gpu-check.sbatch` passed: `hover devices` found the B200 for both JAX (`cuda:0`) and
PyTorch, the batched GPU rollout stayed finite with states on the GPU, and `pytest` passed
12/12 including the CPU-vs-GPU rollout comparison.

Simulator only (`hover benchmark`, zero actions, world-steps/s after compilation; CPU = the
same allocation's 28 cores of a busy DGX node):

| Worlds | CPU | GPU |
| ---: | ---: | ---: |
| 1 | 363 | 198 |
| 64 | 60,227 | 22,824 |
| 1,024 | 381,829 | 347,491 |
| 16,384 | 953,157 | 3,588,572 |

End-to-end PPO training, steady steps/s excluding the first rollout (`--torch-threads 8`):

| Worlds | sim CPU, policy CPU | sim GPU, policy CPU | sim CPU, policy CUDA | sim GPU, policy CUDA |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 294 | 168 | — | — |
| 256 (batch 4096) | 22,798 | 15,252 | 20,846 | 13,128 |
| 4,096 (batch 16384) | — | — | — | 62,244 |

**Recommendation: train on CPU nodes.** The tiny 64x64 policy gains nothing from CUDA, and the
simulator only beats the CPU with thousands of worlds. GPU compilation also costs 10–28 s
versus 2–4 s on CPU. The 4,096-world configuration is the fastest overall, but with 1M steps it
makes only 16 PPO updates and did not learn (0.76 m, 100% crash). Using it would need retuned
PPO settings (fewer worlds per update or more epochs and much larger budgets). On
`genoa-std-mem` the 64-world setup reaches ~20,000 steps/s per run with three runs at once, so
a 1M-step, three-seed experiment takes about 3.5 minutes, and those nodes start much sooner.

Caveat for GPU+CUDA runs: JAX warns that the cuBLAS 13.1 pinned by PyTorch 2.14.1 can
silently corrupt results when cuBLAS kernels run concurrently with other kernels. The training
loop serializes JAX and PyTorch through host copies, and the GPU rollout matched the CPU
rollout, but a cuBLAS of at least 13.2 would need a PyTorch build that pins it. Avoid
`--sim-device gpu --policy-device cuda` for results that matter until then.

CPU experiment on held-out seeds, with commands and the results table:
[hover-validation.md](hover-validation.md#parcc-validation--october-27-2026). Run it with:

```sh
EXPERIMENT=ppo-1m-cpu STEPS=1000000 SEEDS="0 1 2" \
  TRAIN_ARGS="--n-worlds 64 --batch-size 1024 --torch-threads 8 --eval-episodes 20" \
  sbatch -p genoa-std-mem --gres=none -c 24 --mem=132G -t 04:00:00 slurm/experiments.sbatch
```
