"""Analyse flow-planner plans flown on the Crazyflie: python experiments/flow_plans/analyze.py

Reads runs/flow/exec/*.csv (+ per-plan .npz) and runs/flow/plans, prints markdown tables and
writes them to runs/flow/analysis/. Questions:

1. How often do plans succeed on the drone, per controller and condition, against the planning
   study's double integrator (tight tracker, nominal and strongest random force)?
2. Across (task, method, flow) cells, does the success lost on the drone go with sampler
   sensitivity (gain), bending energy, planned clearance, or the acceleration a plan demands?
3. Do learned-vs-ideal flow differences in outcome persist on the drone?
4. Per plan: how does success depend on demanded acceleration, per controller?
5. Coverage: does training the tracker on the planner's path family help (path specialist vs the
   zero-shot generalist)?
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("runs/flow")
LIMIT = 9.81 * np.tan(0.35)  # horizontal acceleration at the 20 deg tilt limit
BINS = [0, 1.0, 2.0, LIMIT, np.inf]
CONTROLLERS = ["pid", "pid-ff", "generalist", "path"]
NAMES = {
    "pid": "PID",
    "pid-ff": "PID + plan feedforward",
    "generalist": "Learned generalist (never saw these paths)",
    "path": "Learned path specialist",
}
CONDITIONS = ["nominal", "realistic", "held-out"]


def load_rows():
    rows = []
    for f in sorted((ROOT / "exec").glob("*.csv")):
        with f.open() as fh:
            for r in csv.DictReader(fh):
                for k, v in r.items():
                    try:
                        r[k] = float(v)
                    except (TypeError, ValueError):
                        pass
                r["family"] = (
                    r["controller"].split("-s")[0]
                    if r["controller"].startswith(("generalist", "path"))
                    else r["controller"]
                )
                rows.append(r)
    return rows


def plan_geometry():
    """Per planner seed and cell: per-plan peak acceleration and planned clearance."""
    scene = json.loads((ROOT / "plans" / "scene.json").read_text())
    s = np.linspace(0, 1, 401)
    j = np.arange(1, 5)
    out = {}
    for f in sorted((ROOT / "plans").glob("seed*.npz")):
        seed = int(f.stem[4:])
        data = np.load(f)
        for key in data.files:
            if not key.endswith("|c"):
                continue
            cell = key[:-2]
            c = data[key].reshape(-1, 4, 2)
            sin = np.sin(np.pi * s[:, None] * j)
            pts = (
                np.asarray(scene["start"])[None, None]
                + s[None, :, None]
                * (np.asarray(scene["goal"]) - np.asarray(scene["start"]))[None, None]
                + np.einsum("sj,njk->nsk", sin, c)
            )
            acc = -np.einsum("sj,njk->nsk", (j * np.pi) ** 2 * sin, c) / 6.0**2
            task = cell.split("|")[0]
            discs = [scene["central"]] + (
                [scene["block"]] if scene["tasks"][task][0] == "clearance" else []
            )
            gap = np.min(
                [np.linalg.norm(pts - np.asarray(cn), axis=-1) - r for cn, r in discs], 0
            ).min(1)
            out[seed, cell] = {
                "peak_acc": np.linalg.norm(acc, axis=-1).max(1),
                "clearance": gap,
                "di": data[cell + "|di_tight_nominal"],
            }
    return out


def mean(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def table(header, rows):
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |",
    ]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def pct(x):
    return "—" if x != x else f"{100 * x:.1f}%"


def main():
    rows = load_rows()
    geometry = plan_geometry()
    out = ROOT / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    report = []

    # 1. Overall success (cells weighted equally; learned controllers averaged over policy seeds).
    by = defaultdict(list)
    di_nom, di_force = [], []
    seen = set()
    for r in rows:
        by[r["family"], r["condition"]].append(r["success"])
        cell = (r["planner_seed"], r["task"], r["method"], r["flow"])
        if cell not in seen:
            seen.add(cell)
            di_nom.append(r["di_tight_nominal"])
            di_force.append(r["di_tight_force"])
    report.append("## 1. Success rate over all plans\n")
    report.append(
        f"Double integrator (planning study, tight tracker): nominal {pct(mean(di_nom))}, "
        f"strongest random force {pct(mean(di_force))}; {len(seen)} cells of 512 plans.\n"
    )
    report.append(
        table(
            ["Controller on the Crazyflie", *CONDITIONS],
            [
                [NAMES[c], *[pct(mean(by[c, k])) for k in CONDITIONS]]
                for c in CONTROLLERS
                if any(by[c, k] for k in CONDITIONS)
            ],
        )
    )

    # 1b. By method, realistic and held-out.
    report.append("\n\n## 1b. By sampling method (mean over tasks, flows and planner seeds)\n")
    methods = ["unguided", "refinement", "early", "constant", "late", "optimiser"]
    for cond in ["realistic", "held-out"]:
        rowsm = []
        for m in methods:
            sel = [r for r in rows if r["method"] == m and r["condition"] == cond]
            if not sel:
                continue
            cells = {(r["planner_seed"], r["task"], r["flow"]): r for r in sel}
            rowsm.append(
                [
                    m,
                    pct(mean([r["di_tight_nominal"] for r in cells.values()])),
                    f"{mean([r['peak_acc_median'] for r in cells.values()]):.2f}",
                    pct(mean([r["over_limit"] for r in cells.values()])),
                    *[
                        pct(mean([r["success"] for r in sel if r["family"] == c]))
                        for c in CONTROLLERS
                    ],
                ]
            )
        report.append(f"\n**{cond}**\n")
        report.append(
            table(
                [
                    "Method",
                    "Double integrator",
                    "Peak accel. (m/s²)",
                    "Beyond drone limit",
                    *[NAMES[c] for c in CONTROLLERS],
                ],
                rowsm,
            )
        )

    # 2. Association across (task, method, flow) cells, aggregated over planner seeds.
    report.append("\n\n## 2. What predicts the success lost on the drone?\n")
    report.append(
        "Spearman correlation across (task, method, flow) cells between the drop "
        "(double-integrator nominal success minus drone success) and each predictor. Sampler "
        "gain is measured for learned flows only, as in the planning study.\n"
    )
    assoc_rows, csv_rows = [], []
    for fam in CONTROLLERS:
        for cond in CONDITIONS:
            cells = defaultdict(list)
            for r in rows:
                if r["family"] == fam and r["condition"] == cond:
                    cells[r["task"], r["method"], r["flow"]].append(r)
            if len(cells) < 5:
                continue
            drop, preds = [], defaultdict(list)
            gain_drop, gains = [], []
            for (task, method, flow), rs in cells.items():
                d = mean([x["di_tight_nominal"] for x in rs]) - mean([x["success"] for x in rs])
                drop.append(d)
                geo = [geometry[int(x["planner_seed"]), f"{task}|{method}|{flow}"] for x in rs]
                preds["bending"].append(mean([x["bending"] for x in rs]))
                preds["peak acceleration"].append(
                    mean([float(np.median(g["peak_acc"])) for g in geo])
                )
                preds["beyond tilt limit"].append(mean([x["over_limit"] for x in rs]))
                preds["planned clearance (5%)"].append(
                    mean([float(np.quantile(g["clearance"], 0.05)) for g in geo])
                )
                if (
                    flow == "learned"
                    and isinstance(rs[0]["sampler_gain"], float)
                    and rs[0]["sampler_gain"] == rs[0]["sampler_gain"]
                ):
                    gain_drop.append(d)
                    gains.append(mean([x["sampler_gain"] for x in rs]))
            row = [NAMES[fam], cond, len(drop)]
            for name in [
                "bending",
                "peak acceleration",
                "beyond tilt limit",
                "planned clearance (5%)",
            ]:
                rho, p = stats.spearmanr(preds[name], drop)
                row.append(f"{rho:+.2f} (p={p:.2g})")
                csv_rows.append(
                    {
                        "controller": fam,
                        "condition": cond,
                        "predictor": name,
                        "spearman": rho,
                        "p": p,
                        "cells": len(drop),
                    }
                )
            rho, p = stats.spearmanr(gains, gain_drop)
            row.append(f"{rho:+.2f} (p={p:.2g}, n={len(gains)})")
            csv_rows.append(
                {
                    "controller": fam,
                    "condition": cond,
                    "predictor": "sampler gain",
                    "spearman": rho,
                    "p": p,
                    "cells": len(gains),
                }
            )
            assoc_rows.append(row)
    report.append(
        table(
            [
                "Controller",
                "Condition",
                "Cells",
                "Bending",
                "Peak accel.",
                "Beyond limit",
                "Planned clearance",
                "Sampler gain",
            ],
            assoc_rows,
        )
    )
    with (out / "association.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)

    # 3. Learned vs ideal flow: paired outcome mismatch per plan.
    report.append("\n\n## 3. Learned vs ideal flow: share of paired plans whose outcome differs\n")
    mism = defaultdict(list)
    for f in sorted((ROOT / "exec").glob("*.npz")):
        name = f.stem
        fam = next(c for c in ["pid-ff", "pid", "generalist", "path"] if name.startswith(c))
        cond = next(c for c in CONDITIONS if f"-{c}-p" in name)
        seed = int(name.rsplit("-p", 1)[1])
        data = np.load(f)
        for key in data.files:
            task, method, flow = key.split("|")
            if flow != "learned":
                continue
            ideal = f"{task}|{method}|ideal"
            if ideal in data.files:
                mism[fam, cond].append(float((data[key] != data[ideal]).mean()))
                if fam == "pid" and cond == "nominal":
                    g1, g2 = geometry[seed, key], geometry[seed, ideal]
                    mism["di", "nominal"].append(float((g1["di"] != g2["di"]).mean()))
    report.append(f"Double integrator, nominal: {pct(mean(mism['di', 'nominal']))}\n")
    report.append(
        table(
            ["Controller", *CONDITIONS],
            [
                [NAMES[c], *[pct(mean(mism[c, k])) for k in CONDITIONS]]
                for c in CONTROLLERS
                if mism[c, "nominal"]
            ],
        )
    )

    # 4. Per plan: success by demanded acceleration.
    report.append("\n\n## 4. Success by the acceleration a plan demands (per plan, all cells)\n")
    labels = [
        f"< {BINS[1]:.0f}",
        f"{BINS[1]:.0f}–{BINS[2]:.0f}",
        f"{BINS[2]:.0f}–{LIMIT:.1f}",
        f"> {LIMIT:.1f} (beyond tilt limit)",
    ]
    for cond in ["realistic", "held-out"]:
        acc = defaultdict(lambda: defaultdict(list))
        counts = np.zeros(4, int)
        di_bins = defaultdict(list)
        for f in sorted((ROOT / "exec").glob(f"*-{cond}-p*.npz")):
            fam = next(c for c in ["pid-ff", "pid", "generalist", "path"] if f.stem.startswith(c))
            seed = int(f.stem.rsplit("-p", 1)[1])
            data = np.load(f)
            for key in data.files:
                g = geometry[seed, key]
                b = np.digitize(g["peak_acc"], BINS[1:-1])
                for k in range(4):
                    acc[fam][k].extend(data[key][b == k].tolist())
                    if fam == "pid":
                        di_bins[k].extend(g["di"][b == k].tolist())
                        counts[k] += int((b == k).sum())
        report.append(f"\n**{cond}** (plans per bin: {', '.join(str(c) for c in counts)})\n")
        report.append(
            table(
                ["Peak acceleration (m/s²)", "Double integrator", *[NAMES[c] for c in CONTROLLERS]],
                [
                    [labels[k], pct(mean(di_bins[k])), *[pct(mean(acc[c][k])) for c in CONTROLLERS]]
                    for k in range(4)
                ],
            )
        )

    # 5. Intervention: the same guided laws with a drone-feasibility penalty added.
    feasible = ROOT / "exec-feasible"
    if feasible.exists():
        report.append("\n\n## 5. Adding a feasibility penalty to the planner's guidance\n")
        report.append(
            "Each guided law (same schedule, calibrated level and test noise) with "
            "the task penalty plus an acceleration limit of 2.86 m/s² over the 6 s traversal. "
            "Means over tasks, flows and planner seeds of paired cells.\n"
        )
        base = {
            (
                r["planner_seed"],
                r["task"],
                r["method"],
                r["flow"],
                r["controller"],
                r["condition"],
            ): r
            for r in rows
        }
        pairs = defaultdict(list)
        for f in sorted(feasible.glob("*.csv")):
            with f.open() as fh:
                for r in csv.DictReader(fh):
                    method = r["method"].replace("+feasible", "")
                    key = (
                        float(r["planner_seed"]),
                        r["task"],
                        method,
                        r["flow"],
                        r["controller"],
                        r["condition"],
                    )
                    if key in base:
                        fam = (
                            r["controller"].split("-s")[0]
                            if r["controller"].startswith(("generalist", "path"))
                            else r["controller"]
                        )
                        pairs[fam, r["condition"], method].append(
                            (
                                base[key],
                                {
                                    k: float(v)
                                    for k, v in r.items()
                                    if k
                                    in (
                                        "success",
                                        "di_tight_nominal",
                                        "over_limit",
                                        "via_ok",
                                        "goal_ok",
                                        "collision",
                                    )
                                },
                            )
                        )
        prow = []
        for (fam, cond, method), ps in sorted(pairs.items()):
            b, f2 = [p[0] for p in ps], [p[1] for p in ps]
            prow.append(
                [
                    NAMES[fam],
                    cond,
                    method,
                    len(ps),
                    f"{pct(mean([x['over_limit'] for x in b]))} → {pct(mean([x['over_limit'] for x in f2]))}",
                    f"{pct(mean([x['di_tight_nominal'] for x in b]))} → {pct(mean([x['di_tight_nominal'] for x in f2]))}",
                    f"{pct(mean([x['success'] for x in b]))} → {pct(mean([x['success'] for x in f2]))}",
                ]
            )
        report.append(
            table(
                [
                    "Controller",
                    "Condition",
                    "Method",
                    "Cells",
                    "Beyond tilt limit",
                    "Double integrator success",
                    "Drone success",
                ],
                prow,
            )
        )

    text = "\n".join(report)
    (out / "report.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
