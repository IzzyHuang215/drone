"""Export generated plans from the ese6180-generative-flows planning study (read-only).

Run from the root of that repository with its environment, writing outside it:

    cd ../ese6180-generative-flows
    uv run --no-sync python ../drone/experiments/flow_plans/export_plans.py \
        --out ../drone/runs/flow/plans --paths 512

For each of its trained models (seeds) this regenerates, with the study's own code and test
noise, the plans of every calibrated sampling law (unguided, terminal refinement, early /
constant / late guidance; learned and ideal flow) plus the trajectory optimiser's plan, and runs
the study's double-integrator execution on the same plans (tight and loose trackers, nominal and
strongest random force) so that both executions are paired per plan. Nothing in the flows
repository is written.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path("experiments/planning").resolve()))
from common import MODES, SEEDS, START, GOAL, CENTRAL, BLOCK, TASKS, VIA_TOLERANCE, basis  # noqa: E402
from execute import CONDITIONS, N_EXEC, RES, TRACKERS, optimiser_plans, regenerate, run_batch  # noqa: E402

DI_RUNS = [("tight", "nominal"), ("tight", "force_1.5"), ("loose", "nominal")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    assert args.paths <= N_EXEC
    torch.set_default_dtype(torch.float64)
    args.out.mkdir(parents=True, exist_ok=True)
    stored = pd.read_csv(RES / "evaluate_all.csv")
    b = basis()
    scene = {
        "modes": MODES,
        "start": START,
        "goal": GOAL,
        "central": CENTRAL,
        "block": BLOCK,
        "tasks": {k: [v[0], v[1]] for k, v in TASKS.items()},
        "via_tolerance": VIA_TOLERANCE,
        "duration": TRACKERS["tight"].duration,
        "source_commit": None,
    }
    (args.out / "scene.json").write_text(json.dumps(scene, indent=2))
    optimiser = optimiser_plans()
    for seed in args.seeds:
        rows = stored[
            (stored.seed == seed)
            & stored.experiment.isin(["baseline", "objective"])
            & stored.mse.notna()
        ]
        plans = regenerate(seed, rows)
        arrays, cells = {}, []
        for (task, method), flows in plans.items():
            gain = float(rows[(rows.task == task) & (rows.method == method)].gain_mean.median())
            for flow, c in flows.items():
                c = c[: args.paths]
                key = f"{task}|{method}|{flow}"
                arrays[f"{key}|c"] = c.numpy()
                arrays[f"{key}|bending"] = b.bending_energy(c).numpy()
                for tracker, condition in DI_RUNS:
                    _, ok = run_batch(c, task, TRACKERS[tracker], CONDITIONS[condition], seed)
                    arrays[f"{key}|di_{tracker}_{condition}"] = ok.numpy()
                cells.append({"task": task, "method": method, "flow": flow, "sampler_gain": gain})
        for task, c in optimiser.items():
            c = c[None].expand(args.paths, -1)
            key = f"{task}|optimiser|optimiser"
            arrays[f"{key}|c"] = c.numpy()
            arrays[f"{key}|bending"] = b.bending_energy(c).numpy()
            for tracker, condition in DI_RUNS:
                _, ok = run_batch(c, task, TRACKERS[tracker], CONDITIONS[condition], seed)
                arrays[f"{key}|di_{tracker}_{condition}"] = ok.numpy()
            cells.append(
                {"task": task, "method": "optimiser", "flow": "optimiser", "sampler_gain": None}
            )
        np.savez_compressed(args.out / f"seed{seed}.npz", **arrays)
        (args.out / f"seed{seed}.json").write_text(json.dumps(cells, indent=2))
        print(f"seed {seed}: {len(cells)} cells", flush=True)


if __name__ == "__main__":
    main()
