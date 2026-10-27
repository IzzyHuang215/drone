"""Portfolio tables from suite evaluations: python -m hover.suite.report runs/suite.

Expects ``baselines/{pid,zero}-<robustness>-<dynamics>.json`` and, for every trained run
``<group>/<name>-s<seed>/``, the ``final-<robustness>-<dynamics>.json`` files written by
slurm/suite-train.sbatch. Runs that differ only in seed are aggregated (mean, min-max).
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from hover.suite import tasks as T

# (robustness, dynamics, drone, column); the file suffix drops the drone for cf2x_L250.
CONDITIONS = [
    ("none", "so_rpy", "cf2x_L250", "Nominal"),
    ("real", "so_rpy", "cf2x_L250", "Realistic"),
    ("real", "first_principles", "cf2x_L250", "Second model"),
    ("real", "so_rpy", "cf2x_P250", "Held-out drone"),
    ("real", "first_principles", "cf2x_P250", "Held-out drone, 2nd model"),
]


def suffix(rob, dyn, drone):
    return f"{rob}-{dyn}" + ("" if drone == "cf2x_L250" else f"-{drone}")


def load(path):
    return json.loads(path.read_text())["tasks"] if path.exists() else {}


def collect(root):
    """{policy label: {condition: [per-seed task dicts]}}."""
    policies = defaultdict(lambda: defaultdict(list))
    for name in ("pid", "zero"):
        label = {"pid": "PID baseline", "zero": "Zero action"}[name]
        for cond in CONDITIONS:
            policies[label][cond].append(
                load(root / "baselines" / f"{name}-{suffix(*cond[:3])}.json")
            )
    for run in sorted(root.glob("*/*-s*/")):
        if run.parent.name == "baselines" or not (run / "policy.zip").exists():
            continue
        label = f"{run.parent.name}/{re.sub(r'-s\d+$', '', run.name)}"
        for cond in CONDITIONS:
            result = load(run / f"final-{suffix(*cond[:3])}.json")
            if result:
                policies[label][cond].append(result)
    return policies


def cell(values, percent=True):
    values = [v for v in values if v is not None]
    if not values:
        return "—"
    mean = np.mean(values)
    if percent:
        text = f"{100 * mean:.0f}%"
        return (
            text
            if len(values) == 1
            else f"{text} ({100 * min(values):.0f}–{100 * max(values):.0f})"
        )
    return f"{mean:.3f}" if len(values) == 1 else f"{mean:.3f} ±{np.std(values):.3f}"


def main(root):
    root = Path(root)
    policies = collect(root)
    lines = [
        "Success rate per task (100 held-out episodes from seed 20000 per task and condition; "
        "ranges span training seeds). Mean tracking error in metres underneath.",
        "",
    ]
    for task in T.TASKS.values():
        rows = []
        for label, conditions in policies.items():
            if not any(task.name in r for runs in conditions.values() for r in runs):
                continue
            success, error = [], []
            for cond in CONDITIONS:
                runs = [r[task.name] for r in conditions[cond] if task.name in r]
                success.append(cell([r["success_rate"] for r in runs]))
                error.append(cell([r["mean_error_m"] for r in runs], percent=False))
            rows.append(f"| {label} | " + " | ".join(success) + " |")
            rows.append("| ↳ error (m) | " + " | ".join(error) + " |")
        if not rows:
            continue
        lines += [
            f"### {task.level}. {task.name}: {task.summary}",
            "",
            "| Policy | " + " | ".join(c[3] for c in CONDITIONS) + " |",
            "| --- | " + " | ".join("---:" for _ in CONDITIONS) + " |",
            *rows,
            "",
        ]
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/suite")
