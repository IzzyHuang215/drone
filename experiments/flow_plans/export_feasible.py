"""Regenerate the planner's guided plans with an added drone-feasibility penalty (read-only).

Same invocation as export_plans.py, writing a parallel plan set:

    cd ../ese6180-generative-flows
    uv run --no-sync python ../drone/experiments/flow_plans/export_feasible.py \
        --out ../drone/runs/flow/plans-feasible --paths 512

Intervention: every guided law (terminal refinement and the early / constant / late schedules,
at the planning study's calibrated levels and on the same test noise) uses the task penalty
plus ``AccelerationLimit``, which penalizes reference acceleration beyond A_MAX, 80% of the
horizontal acceleration a Crazyflie reaches at its 20 degree tilt limit, over the study's 6 s
traversal. Unguided samples and the optimiser are unchanged and are not re-exported.
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path("experiments/planning").resolve()))
from common import SEEDS, TASKS, basis, load_model, penalty, target  # noqa: E402
from evaluate import N_CAL, N_TEST, N_VAL, SETTINGS, SHAPES, STEPS  # noqa: E402
from execute import CONDITIONS, RES, TRACKERS, run_batch  # noqa: E402
from flowfb import guidance as gd  # noqa: E402
from flowfb import planning as pl  # noqa: E402
from flowfb import rollout as ro  # noqa: E402

DURATION = 6.0
A_MAX = 0.8 * 9.81 * math.tan(0.35)
DI_RUNS = [("tight", "nominal"), ("tight", "force_1.5"), ("loose", "nominal")]


@dataclass
class AccelerationLimit:
    """V(c) = 1/2 mean_s max(0, |p''(s)| / T^2 - a_max)^2 over the traversal time T."""

    basis: pl.PathBasis
    a_max: float = A_MAX
    duration: float = DURATION
    quadrature: int = 201

    def _acc(self, c):
        s = pl.grid(self.quadrature, c.dtype)
        j = torch.arange(1, self.basis.modes + 1, dtype=c.dtype)
        weights = (
            -((j * math.pi) ** 2) * torch.sin(math.pi * s[:, None] * j[None]) / self.duration**2
        )
        acc = torch.einsum("sj,njk->nsk", weights, c.reshape(c.shape[0], self.basis.modes, 2))
        return weights, acc

    def value(self, c, t=1.0):
        _, acc = self._acc(c)
        depth = (acc.norm(dim=-1) - self.a_max).clamp_min(0.0)
        return 0.5 * (depth**2).mean(1)

    def grad(self, c, t):
        weights, acc = self._acc(c)
        norm = acc.norm(dim=-1).clamp_min(1e-12)
        depth = (norm - self.a_max).clamp_min(0.0)
        push = (depth / norm)[..., None] * acc  # d(1/2 depth^2) / d acc
        return (torch.einsum("sj,nsk->njk", weights, push) / weights.shape[0]).reshape(
            c.shape[0], -1
        )


@dataclass
class Sum:
    terms: list

    def value(self, c, t=1.0):
        return sum(p.value(c, t) for p in self.terms)

    def grad(self, c, t):
        return sum(p.grad(c, t) for p in self.terms)


def check_gradient():
    b = basis()
    pen = AccelerationLimit(b)
    c = torch.randn(4, b.d, dtype=torch.float64) * 1.5
    g = pen.grad(c, 1.0)
    eps = 1e-6
    fd = torch.zeros_like(c)
    for i in range(b.d):
        e = torch.zeros_like(c)
        e[:, i] = eps
        fd[:, i] = (pen.value(c + e) - pen.value(c - e)) / (2 * eps)
    err = float((g - fd).abs().max() / fd.abs().max().clamp_min(1e-12))
    assert err < 1e-5, err
    return err


@torch.no_grad()
def regenerate_feasible(seed, stored):
    mix, model = target(), load_model(seed)
    gen = torch.Generator().manual_seed(211 + 19 * seed)  # same batches as evaluate.py / execute.py
    _, _, x0 = (torch.randn(n, mix.d, generator=gen) for n in (N_CAL, N_VAL, N_TEST))
    unguided = {
        "learned": ro.rollout(model, x0, 2 * STEPS)[-1],
        "ideal": ro.rollout(mix.field, x0, 2 * STEPS)[-1],
    }
    feasibility = AccelerationLimit(basis())
    plans = {}
    for row in stored.itertuples():
        if row.method == "unguided":
            continue
        pen, kind = Sum([penalty(row.task), feasibility]), TASKS[row.task][0]
        if row.method == "refinement":
            plans[row.task, row.method] = {
                k: pl.refine(v, pen, row.level, SETTINGS[kind]["refine_steps"])
                for k, v in unguided.items()
            }
        else:
            guide = gd.Guidance(pen, SHAPES[row.method](row.level))
            plans[row.task, row.method] = {
                "learned": ro.rollout(ro.closed_loop(model, guide), x0, 2 * STEPS)[-1],
                "ideal": ro.rollout(ro.closed_loop(mix.field, guide), x0, 2 * STEPS)[-1],
            }
    return plans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    print(f"gradient check: relative error {check_gradient():.1e}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    stored = pd.read_csv(RES / "evaluate_all.csv")
    b = basis()
    for seed in args.seeds:
        rows = stored[
            (stored.seed == seed)
            & stored.experiment.isin(["baseline", "objective"])
            & stored.mse.notna()
        ]
        arrays, cells = {}, []
        for (task, method), flows in regenerate_feasible(seed, rows).items():
            gain = float(rows[(rows.task == task) & (rows.method == method)].gain_mean.median())
            for flow, c in flows.items():
                c = c[: args.paths]
                key = f"{task}|{method}+feasible|{flow}"
                arrays[f"{key}|c"] = c.numpy()
                arrays[f"{key}|bending"] = b.bending_energy(c).numpy()
                for tracker, condition in DI_RUNS:
                    _, ok = run_batch(c, task, TRACKERS[tracker], CONDITIONS[condition], seed)
                    arrays[f"{key}|di_{tracker}_{condition}"] = ok.numpy()
                cells.append(
                    {
                        "task": task,
                        "method": f"{method}+feasible",
                        "flow": flow,
                        "sampler_gain": gain,
                    }
                )
        np.savez_compressed(args.out / f"seed{seed}.npz", **arrays)
        (args.out / f"seed{seed}.json").write_text(json.dumps(cells, indent=2))
        print(f"seed {seed}: {len(cells)} cells", flush=True)
    src = Path(args.out).parent / "plans" / "scene.json"
    if src.exists():
        (args.out / "scene.json").write_text(src.read_text())


if __name__ == "__main__":
    main()
