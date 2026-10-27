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

CONDITIONS = [
    ("none", "so_rpy", "Nominal"),
    ("real", "so_rpy", "Realistic"),
    ("none", "first_principles", "Unseen model"),
    ("real", "first_principles", "Realistic + unseen"),
]


def load(path):
    return json.loads(path.read_text())["tasks"] if path.exists() else {}


def collect(root):
    """{policy label: {condition: [per-seed task dicts]}}."""
    policies = defaultdict(lambda: defaultdict(list))
    for name in ("pid", "zero"):
        label = {"pid": "PID baseline", "zero": "Zero action"}[name]
        for rob, dyn, _ in CONDITIONS:
            policies[label][(rob, dyn)].append(
                load(root / "baselines" / f"{name}-{rob}-{dyn}.json")
            )
    for run in sorted(root.glob("*/*-s*/")):
        if run.parent.name == "baselines" or not (run / "policy.zip").exists():
            continue
        label = f"{run.parent.name}/{re.sub(r'-s\d+$', '', run.name)}"
        for rob, dyn, _ in CONDITIONS:
            result = load(run / f"final-{rob}-{dyn}.json")
            if result:
                policies[label][(rob, dyn)].append(result)
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
            for rob, dyn, _ in CONDITIONS:
                runs = [r[task.name] for r in conditions[(rob, dyn)] if task.name in r]
                success.append(cell([r["success_rate"] for r in runs]))
                error.append(cell([r["mean_error_m"] for r in runs], percent=False))
            rows.append(f"| {label} | " + " | ".join(success) + " |")
            rows.append("| ↳ error (m) | " + " | ".join(error) + " |")
        if not rows:
            continue
        lines += [
            f"### {task.level}. {task.name}: {task.summary}",
            "",
            "| Policy | " + " | ".join(c[2] for c in CONDITIONS) + " |",
            "| --- | " + " | ".join("---:" for _ in CONDITIONS) + " |",
            *rows,
            "",
        ]
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/suite")
