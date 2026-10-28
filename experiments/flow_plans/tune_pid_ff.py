"""Tune the feedforward PID's gains on random paths from the training prior (never on plans).

    uv run --extra gpu python experiments/flow_plans/tune_pid_ff.py

Objective: mean tracking error over 256 prior paths in the realistic condition, a crash
counting as 1 m for the rest of its episode. Writes runs/flow/pid_ff_gains.json.
"""

import itertools
import json
from pathlib import Path

import jax
import numpy as np

from hover.suite.env import SuiteVecEnv, drone_constants
from hover.suite.plans import FeedforwardPID, path_acceleration
import dataclasses

from hover.suite import tasks as T

TASK = dataclasses.replace(T.TASKS["path"], start_pos=0.0, start_vel=0.0)
STEPS = 600


def score(controller, env):
    env.seed(12345)
    obs = env.reset()
    ep = jax.device_get(env._ep)
    c, tau, rot = np.asarray(ep["path_c"]), np.asarray(ep["path_T"]), np.asarray(ep["path_rot"])
    alive, total = np.ones(env.num_envs, bool), np.zeros(env.num_envs)
    for k in range(STEPS):
        actions, _ = controller.predict(obs, acc_ff=path_acceleration(c, k * env.dt, tau, rot))
        obs, _, _, terminated, truncated, info = env.step_arrays(actions)
        total += np.where(alive, info["err"], 1.0)
        alive &= ~(terminated | truncated)
    return float(total.mean() / STEPS), float(1 - alive.mean())


def main():
    env = SuiteVecEnv([TASK], 256, robustness="real")
    constants = drone_constants("cf2x_L250", "so_rpy")
    results = []
    for kp, kd, ki in itertools.product([16.0, 22.0, 30.0, 40.0, 55.0, 75.0], [6.0, 8.0, 11.0, 15.0], [4.0, 6.0, 9.0]):
        err, crash = score(FeedforwardPID(constants, kp=kp, ki=ki, kd=kd), env)
        results.append({"kp": kp, "kd": kd, "ki": ki, "mean_error_m": err, "crash_rate": crash})
        print(results[-1], flush=True)
    env.close()
    best = min(results, key=lambda r: r["mean_error_m"])
    out = Path("runs/flow")
    out.mkdir(parents=True, exist_ok=True)
    (out / "pid_ff_tuning.json").write_text(json.dumps(results, indent=2))
    (out / "pid_ff_gains.json").write_text(json.dumps(best, indent=2))
    print("best", best)


if __name__ == "__main__":
    main()
