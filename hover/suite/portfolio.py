"""Build the portfolio page: python -m hover.suite.portfolio runs/suite runs/suite/portfolio.

Reads the same evaluation files as hover.suite.report plus the GIFs and render records from
slurm/suite-videos.sbatch, and writes ``index.html`` and ``media/`` into the output folder.
"""

import json
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np

from hover.suite import tasks as T
from hover.suite.report import CONDITIONS, collect

ROWS = [  # (group, row label, chart series, color token)
    ("PID baseline", "PID baseline", "pid", "var(--ref)"),
    ("specialist", "Specialist, nominal training", None, None),
    ("specialist-real", "Specialist, + realistic noise", None, None),
    ("specialist-robust", "Specialist, + both physics models", "specialist", "var(--s1)"),
    ("specialist-robust-long", "Specialist, + both models, 30M steps", "specialist", "var(--s1)"),
    ("generalist", "Generalist, nominal training", None, None),
    ("generalist-robust", "Generalist, + both physics models", "generalist", "var(--s2)"),
]
TITLES = {
    "hover": "Hover",
    "hover-wide": "Hover from anywhere",
    "reach": "Reach a point",
    "recover": "Recover from a throw",
    "disturbance": "Hold in wind and pushes",
    "circle": "Fly a circle",
    "figure8": "Fly a figure-eight",
    "landing": "Land on a pad",
    "waypoints": "Waypoint course",
}
CRITERIA = {
    "final": "mean error under {e:.0f} cm over the last second, without crashing.",
    "tracking": "mean error under {e:.0f} cm after the first second, without crashing.",
    "landing": "touch down slower than 0.4 m/s, tilted less than 20° and within 20 cm of the pad.",
    "waypoints": "capture all {n} waypoints (within {r:.0f} cm) in order, without crashing.",
}


def criteria(task):
    if task.reference == "landing":
        return CRITERIA["landing"]
    if task.reference == "waypoints":
        return CRITERIA["waypoints"].format(n=task.n_waypoints, r=100 * task.waypoint_radius)
    return CRITERIA[task.success_metric].format(e=100 * task.success_error)


def cell(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "range": None}
    lo, hi = min(values), max(values)
    spread = None if len(values) < 2 else f"{100 * lo:.0f}–{100 * hi:.0f}"
    return {"mean": float(np.mean(values)), "range": spread}


def video_note(record, who):
    if not record:
        return ""
    parts = [f"{who} on the held-out drone with realistic noise, seed 20000"]
    if record.get("waypoints") is not None:
        parts.append(f"{record['waypoints']} waypoints captured")
    elif record.get("landed"):
        parts.append("landed softly")
    elif record.get("crashed"):
        parts.append("crashed")
    parts.append(f"mean error {100 * record['mean_position_error_m']:.1f} cm")
    return ", ".join(parts) + "."


def build(root, out):
    root, out = Path(root), Path(out)
    policies = collect(root)
    media = out / "media"
    media.mkdir(parents=True, exist_ok=True)
    tasks, hardest = [], CONDITIONS[-1]
    for task in T.TASKS.values():
        rows, chart = [], {}
        for group, label, series, color in ROWS:
            key = group if group == "PID baseline" else f"{group}/{task.name}"
            if group.startswith("generalist"):
                key = f"{group}/all"
            runs = policies.get(key)
            if not runs or not any(task.name in r for c in runs.values() for r in c):
                continue
            cells = [
                cell([r[task.name]["success_rate"] for r in runs[c] if task.name in r])
                for c in CONDITIONS
            ]
            rows.append({"label": label, "cells": cells, "color": color, "series": series})
            if series and cells[-1]["mean"] is not None:
                best = chart.get(series)
                chart[series] = max(best or 0.0, cells[-1]["mean"])
        for r in rows:  # mark the row the chart shows for each series
            r["best"] = bool(r["series"]) and r["cells"][-1]["mean"] == chart.get(r["series"])
        video = None
        vdir = root / "videos" / "specialist-robust"
        if (vdir / f"{task.name}.gif").exists():
            for suffix in ("", "-pid"):
                shutil.copy(vdir / f"{task.name}{suffix}.gif", media / f"{task.name}{suffix}.gif")

            def record(suffix, vdir=vdir, name=task.name):
                path = vdir / f"{name}{suffix}.json"
                return json.loads(path.read_text()) if path.exists() and path.stat().st_size else {}

            video = {
                "policy": f"media/{task.name}.gif",
                "pid": f"media/{task.name}-pid.gif",
                "policy_label": "Robust specialist",
                "policy_note": video_note(record(""), "Robust specialist (training seed 0)"),
                "pid_note": video_note(record("-pid"), "PID baseline"),
            }
        tasks.append(
            {
                "level": task.level,
                "name": task.name,
                "title": TITLES[task.name],
                "summary": task.summary,
                "criteria": criteria(task),
                "rows": rows,
                "chart": chart,
                "video": video,
                "note": "Each cell is the success rate over 100 held-out episodes; small numbers show "
                "the range across three training seeds.",
            }
        )

    def mean_of(series):
        values = [t["chart"].get(series) for t in tasks if t["chart"].get(series) is not None]
        return float(np.mean(values)) if values else None

    spec, gen, pid = mean_of("specialist"), mean_of("generalist"), mean_of("pid")
    facts = [{"value": str(len(tasks)), "label": "tasks, from hovering to a waypoint course"}]
    if spec is not None and pid is not None:
        facts.append(
            {
                "value": f"{100 * spec:.0f}% vs {100 * pid:.0f}%",
                "label": "robust specialists vs PID, averaged over tasks on the held-out drone",
            }
        )
    if gen is not None:
        facts.append(
            {
                "value": f"{100 * gen:.0f}%",
                "label": "one robust generalist policy flying all nine tasks, same test",
            }
        )
    data = {
        "conditions": [{"label": c[3]} for c in CONDITIONS],
        "tasks": tasks,
        "facts": facts,
        "findings": FINDINGS,
        "method": METHOD,
        "hardest": list(hardest[:3]),
    }
    template = (Path(__file__).parent / "portfolio_template.html").read_text()
    html = template.replace("__DATE__", date.today().strftime("%B %Y")).replace(
        "__DATA__", json.dumps(data).replace("</", "<\\/")
    )
    (out / "index.html").write_text(html)
    (out / "data.json").write_text(json.dumps(data, indent=2))
    return out / "index.html"


FINDINGS = [
    "Enough experience fixed the original hover task. The first PPO runs (4k–32k steps) crashed "
    "every episode; with 64 simulated worlds in parallel and 1M steps, hover error fell to 3.5 cm "
    "on unseen starts, with no change to the task.",
    "An integral of the position error had to be in the observation. Without it a PD "
    "controller succeeded in 0 of 16 realistic hover episodes, because a tired battery or a "
    "steady wind leaves a constant offset that only an integral term can remove.",
    "How a reward is shaped decides what gets learned. Rewarding closeness to the current waypoint "
    "taught the first policy to hover just outside each waypoint (16% success): capturing one "
    "switches to a distant target, which looked like a loss. Rewarding progress instead fixed it.",
    "Sensor noise was not the hard part; the physics model was. Policies trained on one model "
    "shrugged off noise, latency and thrust errors, but failed on a second physics model whose "
    "attitude loop is about 30 ms slower: figure-eight and landing dropped to 0–1%.",
    "Training across both physics models closed most of that gap. On a Crazyflie variant no "
    "policy ever saw, the robust specialists clearly beat the PID baseline on every task except the "
    "waypoint course. Randomizing response delay alone did not do it.",
]
METHOD = (
    "Simulator: Crazyflow 0.4.0 with Crazyflie 2.x models at 500 Hz physics and a 100 Hz policy. "
    "Training: PPO (Stable-Baselines3 2.9.2), a 128×128 tanh network, 64–256 worlds in parallel on "
    "CPU nodes, 3–40 million steps per policy, three training seeds each. Robust policies continue "
    "from the nominal ones with randomized thrust (±15%), trim, wind, gusts, drag, 0–30 ms latency, "
    "0–60 ms attitude lag, sensor noise, and half the worlds on each physics model. Evaluation: "
    "100 episodes per task and condition from seed 20000, never used in tuning; conditions add "
    "realistic noise, the second physics model, and cf2x_P250, a separately identified Crazyflie "
    "2.x configuration with a slower attitude loop that no policy or randomization range was "
    "fitted to. The PID baseline sees the same observation, noise and latency."
)


if __name__ == "__main__":
    print(build(*(sys.argv[1:] or ["runs/suite", "runs/suite/portfolio"])))
