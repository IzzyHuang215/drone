"""Fly plans exported from the flow-matching planner on the simulated Crazyflie.

    python -m hover.suite.plans --plans runs/flow/plans --seed-file 0 \
        --controller pid-ff --condition realistic --output runs/flow/exec/pid-ff-realistic-0.csv

Each plan is a sine-series path (see hover.suite.tasks) flown over T = 6 s from its start, on
the reference with the reference velocity, as in the planning study's execution experiment.
Success follows that study: no crash, no point of the executed trajectory inside an obstacle
disc, horizontal distance to the goal within the via tolerance at t = T, and for via tasks
passage within the tolerance of the via point. Positions are checked every 10 ms; at the
plans' speeds (< 2.5 m/s) the polyline between checks deviates by under 1 cm.
"""

import argparse
import csv
import dataclasses
import json
from pathlib import Path

import numpy as np

from hover.suite import tasks as T
from hover.suite.controllers import TrackingPID
from hover.suite.env import SuiteVecEnv, drone_constants

CONDITIONS = {  # name -> (robustness, dynamics, drone)
    "nominal": ("none", "so_rpy", "cf2x_L250"),
    "realistic": ("real", "so_rpy", "cf2x_L250"),
    "held-out": ("real", "first_principles", "cf2x_P250"),
}
DURATION = 6.0
EVAL_SEED = 30000  # noise and randomization draws; disjoint from training and suite evaluation
# Evaluation episodes are the path task, started exactly on the reference.
EVAL_TASK = dataclasses.replace(T.TASKS["path"], start_pos=0.0, start_vel=0.0, episode_seconds=7.0)


def path_acceleration(c, t, duration=DURATION, rotation=0.0):
    """Analytic planar reference acceleration for coefficients c (n, modes, 2) at time t.

    ``duration`` and ``rotation`` may be scalars or per-path arrays.
    """
    duration = np.broadcast_to(np.asarray(duration, np.float64), (len(c),))
    rotation = np.broadcast_to(np.asarray(rotation, np.float64), (len(c),))
    s = np.clip(t / duration, 0.0, 1.0)
    j = np.arange(1, T.PATH_MODES + 1)
    a = -np.einsum("nj,njk->nk", (j * np.pi) ** 2 * np.sin(np.pi * s[:, None] * j), c)
    a = np.where((t < duration)[:, None], a / duration[:, None] ** 2, 0.0)
    cos, sin = np.cos(rotation), np.sin(rotation)
    return np.stack([cos * a[:, 0] - sin * a[:, 1], sin * a[:, 0] + cos * a[:, 1]], -1)


class FeedforwardPID(TrackingPID):
    """The planning study's tracker (PD plus reference acceleration) mapped through attitude."""

    def predict(self, obs, deterministic=True, acc_ff=None):
        if acc_ff is None:
            return super().predict(obs, deterministic)
        obs = np.array(obs, np.float64)
        # Shift the position error so the PID's desired acceleration gains +acc_ff.
        obs[:, 0:2] -= acc_ff / self.kp
        return super().predict(obs, deterministic)


def make_controller(name, drone, dynamics, checkpoint=None):
    constants = drone_constants(drone, dynamics)
    if name == "pid":
        return TrackingPID(constants)
    if name == "pid-ff":
        gains = json.loads(Path("runs/flow/pid_ff_gains.json").read_text())
        return FeedforwardPID(constants, kp=gains["kp"], ki=gains["ki"], kd=gains["kd"])
    from stable_baselines3 import PPO

    return PPO.load(checkpoint, device="cpu")


def fly(env, controller, c, scene, task):
    """Fly n plans (n = env.num_envs); returns per-plan outcome arrays."""
    n = env.num_envs
    env.set_paths(c, duration=DURATION)
    env.seed(EVAL_SEED)
    obs = env.reset()
    steps = round(DURATION / env.dt)
    xy = np.zeros((steps, n, 2))
    err = np.zeros((steps, n))
    alive = np.ones(n, bool)
    crashed = np.zeros(n, bool)
    import jax

    origin = T.ORIGIN[:2]
    for k in range(steps):
        if isinstance(controller, FeedforwardPID):
            actions, _ = controller.predict(obs, acc_ff=path_acceleration(c, k * env.dt))
        else:
            actions, _ = controller.predict(obs, deterministic=True)
        pos = np.asarray(jax.device_get(env.sim.data.states.pos))[:, 0]
        obs, _, _, terminated, truncated, info = env.step_arrays(actions)
        new_pos = np.asarray(jax.device_get(env.sim.data.states.pos))[:, 0]
        # A finished world has already reset; keep its last pre-reset position.
        pos_now = np.where((terminated | truncated)[:, None], pos, new_pos)
        xy[k] = np.where(
            alive[:, None], pos_now[:, :2] - origin, xy[k - 1] if k else pos[:, :2] - origin
        )
        err[k] = np.where(alive, info["err"], err[k - 1] if k else 0.0)
        crashed |= alive & info["crashed"]
        alive &= ~(terminated | truncated)
    kind, via = scene["tasks"][task]
    discs = [scene["central"]] + ([scene["block"]] if kind == "clearance" else [])
    clearance = np.min(
        [np.linalg.norm(xy - np.asarray(center), axis=-1) - radius for center, radius in discs],
        axis=0,
    ).min(0)
    tol = scene["via_tolerance"]
    goal_error = np.linalg.norm(xy[-1] - np.asarray(scene["goal"]), axis=-1)
    ok = ~crashed & (clearance >= 0) & (goal_error <= tol)
    via_ok = np.ones(n, bool)
    if via is not None:
        via_ok = np.linalg.norm(xy - np.asarray(via), axis=-1).min(0) <= tol
        ok &= via_ok
    return {
        "success": ok,
        "crashed": crashed,
        "collision": clearance < 0,
        "goal_ok": goal_error <= tol,
        "via_ok": via_ok,
        "tracking_rms": np.sqrt((err**2).mean(0)),
        "tracking_max": err.max(0),
        "clearance": clearance,
    }


def peak_acceleration(c, duration=DURATION, samples=601):
    s = np.linspace(0, 1, samples)
    j = np.arange(1, T.PATH_MODES + 1)
    a = (
        -np.einsum("sj,njk->nsk", (j * np.pi) ** 2 * np.sin(np.pi * s[:, None] * j), c)
        / duration**2
    )
    return np.linalg.norm(a, axis=-1).max(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, default=Path("runs/flow/plans"))
    parser.add_argument("--seed-file", type=int, required=True, help="planner model seed")
    parser.add_argument(
        "--controller", required=True, help="pid, pid-ff or a label for --checkpoint"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--condition", choices=list(CONDITIONS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-plan", type=Path, help="also save per-plan success arrays (.npz)")
    args = parser.parse_args()
    rob, dyn, drone = CONDITIONS[args.condition]
    scene = json.loads((args.plans / "scene.json").read_text())
    data = np.load(args.plans / f"seed{args.seed_file}.npz")
    cells = json.loads((args.plans / f"seed{args.seed_file}.json").read_text())
    controller = make_controller(args.controller, drone, dyn, args.checkpoint)
    n = len(data[f"{cells[0]['task']}|{cells[0]['method']}|{cells[0]['flow']}|c"])
    env = SuiteVecEnv([EVAL_TASK], n, robustness=rob, dynamics=dyn, drone=drone)
    rows, per_plan = [], {}
    try:
        for cell in cells:
            key = f"{cell['task']}|{cell['method']}|{cell['flow']}"
            c = data[key + "|c"].reshape(n, T.PATH_MODES, 2)
            out = fly(env, controller, c, scene, cell["task"])
            per_plan[key] = out["success"]
            pa = peak_acceleration(c)
            rows.append(
                {
                    **cell,
                    "planner_seed": args.seed_file,
                    "controller": args.controller,
                    "condition": args.condition,
                    "paths": n,
                    **{k: float(np.mean(v)) for k, v in out.items()},
                    "di_tight_nominal": float(data[key + "|di_tight_nominal"].mean()),
                    "di_tight_force": float(data[key + "|di_tight_force_1.5"].mean()),
                    "di_loose_nominal": float(data[key + "|di_loose_nominal"].mean()),
                    "bending": float(data[key + "|bending"].mean()),
                    "peak_acc_median": float(np.median(pa)),
                    "over_limit": float((pa > 9.81 * np.tan(0.35)).mean()),
                }
            )
            print(key, f"{rows[-1]['success']:.3f}", flush=True)
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.per_plan:
        np.savez_compressed(args.per_plan, **per_plan)


if __name__ == "__main__":
    main()
