"""Seeded, batched evaluation: one deterministic episode per world, scored per task."""

import numpy as np

from hover.suite import tasks as T
from hover.suite.env import SuiteVecEnv

# Final-report evaluations start here; tuning uses 10000+.
FINAL_SEED = 20000


def task_success(task, errors, crashed, landed, waypoints):
    """Per-episode success, defined per reference type (errors: per-step error array)."""
    second = round(1.0 / SuiteVecEnv.dt)
    if task.reference == "landing":
        return bool(landed)
    if task.reference == "waypoints":
        return bool(not crashed and waypoints >= task.n_waypoints)
    if crashed:
        return False
    window = errors[second:] if task.success_metric == "tracking" else errors[-second:]
    return bool(np.mean(window) < task.success_error)


def run_episodes(policy, env, seed):
    """Run until every world has finished its first episode; return per-episode records."""
    env.seed(seed)
    obs = env.reset()
    n = env.num_envs
    active = np.ones(n, bool)
    errors = [[] for _ in range(n)]
    smooth = np.zeros(n)
    records = [None] * n
    while active.any():
        actions, _ = policy.predict(obs, deterministic=True)
        obs, _, _, terminated, truncated, info = env.step_arrays(actions)
        for i in np.flatnonzero(active):
            errors[i].append(info["err"][i])
        smooth[active] += info["smooth"][active]
        for i in np.flatnonzero(active & (terminated | truncated)):
            records[i] = {
                "task": env.tasks[info["task"][i]],
                "errors": np.asarray(errors[i]),
                "crashed": bool(info["crashed"][i]),
                "landed": bool(info["landed"][i]),
                "touch_speed": float(info["touch_speed"][i]),
                "waypoints": int(info["waypoints"][i]),
                "smoothness": smooth[i] / len(errors[i]),
            }
        active &= ~(terminated | truncated)
    return records


def summarize(records):
    by_task = {}
    for r in records:
        by_task.setdefault(r["task"].name, []).append(r)
    summary = {}
    for name, rows in by_task.items():
        task = rows[0]["task"]
        success = [
            task_success(task, r["errors"], r["crashed"], r["landed"], r["waypoints"]) for r in rows
        ]
        entry = {
            "episodes": len(rows),
            "success_rate": float(np.mean(success)),
            "crash_rate": float(np.mean([r["crashed"] for r in rows])),
            "mean_error_m": float(np.mean([r["errors"].mean() for r in rows])),
            "final_error_m": float(np.mean([r["errors"][-100:].mean() for r in rows])),
            "mean_duration_s": float(np.mean([len(r["errors"]) for r in rows]) * SuiteVecEnv.dt),
            "action_change": float(np.mean([r["smoothness"] for r in rows])),
        }
        if task.reference == "landing":
            speeds = [r["touch_speed"] for r in rows if np.isfinite(r["touch_speed"])]
            entry["touchdown_speed_mps"] = float(np.mean(speeds)) if speeds else None
        if task.reference == "waypoints":
            entry["mean_waypoints"] = float(np.mean([r["waypoints"] for r in rows]))
        summary[name] = entry
    return summary


def evaluate(policy, tasks, episodes, seed=FINAL_SEED, **env_kwargs):
    """Evaluate on each task separately with ``episodes`` seeded starts per task."""
    results = {}
    for task in tasks:
        env = SuiteVecEnv([task], episodes, **env_kwargs)
        try:
            results.update(
                summarize(run_episodes(policy, env, seed + 1000 * T.TASKS[task.name].level))
            )
        finally:
            env.close()
    return results
