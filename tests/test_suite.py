import json
import subprocess
import sys

import jax
import numpy as np
import pytest

from hover.suite import tasks as T
from hover.suite.controllers import TrackingPID, Zero
from hover.suite.env import OBS_DIM, SuiteVecEnv, drone_constants, reference
from hover.suite.evaluate import evaluate

PID = TrackingPID(drone_constants("cf2x_L250", "so_rpy"))


def test_every_task_runs_with_finite_observations():
    env = SuiteVecEnv(T.resolve(["all"]), 18)
    try:
        env.seed(0)
        obs = env.reset()
        assert obs.shape == (18, OBS_DIM)
        for _ in range(20):
            obs, rewards, dones, infos = env.step(PID.predict(obs)[0])
        assert np.isfinite(obs).all() and np.isfinite(rewards).all()
        assert rewards.flags.writeable
    finally:
        env.close()


def test_reset_is_seeded_including_robustness():
    env = SuiteVecEnv(T.resolve(["recover", "circle"]), 8, robustness="real")
    try:
        env.seed(5)
        first = env.reset()
        env.seed(5)
        np.testing.assert_array_equal(first, env.reset())
    finally:
        env.close()


def test_time_limit_is_truncation_and_crash_is_not():
    env = SuiteVecEnv([T.TASKS["hover"]], 2)
    try:
        env.seed(1)
        env.reset()
        states = env.sim.data.states
        env.sim.data = env.sim.data.replace(
            states=states.replace(pos=states.pos.at[1, 0, 2].set(0.02))
        )
        obs, _, dones, infos = env.step(np.zeros((2, 4), np.float32))
        assert dones.tolist() == [False, True]
        assert infos[1]["TimeLimit.truncated"] is False
        for _ in range(499):
            obs, _, dones, infos = env.step(np.zeros((2, 4), np.float32))
        assert dones[0] and infos[0]["TimeLimit.truncated"] is True
        assert infos[0]["terminal_observation"].shape == (OBS_DIM,)
    finally:
        env.close()


def test_runner_rebuilds_simulator_observations():
    from hover.suite.deploy import PolicyRunner

    env = SuiteVecEnv([T.TASKS["circle"]], 1)
    runner = PolicyRunner.__new__(PolicyRunner)
    runner.reset()
    try:
        env.seed(3)
        obs = env.reset()
        for step in range(60):
            s = jax.device_get(env.sim.data.states)
            t = np.full(1, float(np.asarray(env._ep["t"])[0]) * env.dt, np.float32)
            ref_pos, ref_vel = (np.asarray(x[0]) for x in reference(env._ep, t))
            ahead = np.asarray(reference(env._ep, t + T.LOOKAHEAD_SECONDS)[0][0])
            rebuilt = runner.observation(
                s.pos[0, 0], s.vel[0, 0], s.quat[0, 0], s.ang_vel[0, 0], ref_pos, ref_vel, ahead
            )
            np.testing.assert_allclose(rebuilt, obs[0], atol=2e-5, err_msg=f"step {step}")
            action = PID.predict(obs)[0]
            runner.prev_action, runner.started = action[0], True
            obs, _, dones, _ = env.step(action)
            assert not dones[0]
    finally:
        env.close()


def test_exported_policy_matches_torch(tmp_path):
    from stable_baselines3 import PPO

    from hover.suite.deploy import NumpyPolicy, PolicyRunner, export

    env = SuiteVecEnv([T.TASKS["hover"]], 4)
    try:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=8,
            batch_size=32,
            seed=0,
            policy_kwargs=dict(net_arch=[32, 32]),
        )
        model.learn(64)
        model.save(tmp_path / "policy.zip")
        export(tmp_path / "policy.zip", tmp_path / "policy.npz")
        env.seed(0)
        obs = env.reset()
        expected, _ = model.predict(obs, deterministic=True)
        np.testing.assert_allclose(NumpyPolicy(tmp_path / "policy.npz")(obs), expected, atol=1e-5)
        runner = PolicyRunner(tmp_path / "policy.npz")
        _, setpoint = runner(
            [0, 0, 1], [0, 0, 0], [0, 0, 0, 1], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 1]
        )
        assert abs(setpoint["roll"]) <= 0.35 and 0 < setpoint["thrust"] <= 0.48
    finally:
        env.close()


def test_pid_solves_easy_tasks_and_landing_touches_down_softly():
    results = evaluate(PID, T.resolve(["hover", "landing", "waypoints"]), 4, seed=10000)
    assert results["hover"]["success_rate"] == 1.0
    assert results["landing"]["success_rate"] == 1.0
    assert results["landing"]["touchdown_speed_mps"] < 0.4
    assert results["waypoints"]["mean_waypoints"] == 4
    zero = evaluate(Zero(), T.resolve(["landing"]), 4, seed=10000)
    assert zero["landing"]["success_rate"] < 1.0


def test_first_principles_transfer_model_hovers():
    pid = TrackingPID(drone_constants("cf2x_L250", "first_principles"))
    results = evaluate(pid, [T.TASKS["hover"]], 4, seed=10000, dynamics="first_principles")
    assert results["hover"]["crash_rate"] == 0.0


def run_suite(*args):
    return subprocess.run(
        [sys.executable, "-m", "hover.suite", *map(str, args)],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("robustness", ["none", "real"])
def test_cli_trains_resumes_and_evaluates(tmp_path, robustness):
    run = tmp_path / "run"
    small = [
        "--n-worlds",
        4,
        "--n-steps",
        16,
        "--batch-size",
        32,
        "--n-epochs",
        1,
        "--eval-episodes",
        2,
        "--net-arch",
        16,
        16,
    ]
    run_suite(
        "train",
        "--tasks",
        "hover",
        "landing",
        "--robustness",
        robustness,
        "--steps",
        64,
        "--checkpoint-every",
        64,
        "--output",
        run,
        *small,
    )
    (run / "evaluation.json").unlink()
    run_suite("train", "--resume", run, "--steps", 128)
    result = json.loads((run / "evaluation.json").read_text())
    assert result["actual_training_steps"] == 128
    assert set(result["tuning_evaluation"]) == {"hover", "landing"}
    report = json.loads(
        run_suite(
            "evaluate",
            "--checkpoint",
            run / "policy.zip",
            "--tasks",
            "hover",
            "--episodes",
            2,
            "--robustness",
            robustness,
        ).stdout
    )
    assert report["tasks"]["hover"]["episodes"] == 2
    assert "recover" in run_suite("list").stdout


def test_mixed_training_env_steps_both_models():
    from hover.suite.env import MixedVecEnv, training_env

    env = training_env([T.TASKS["hover"]], 6, dynamics="mix")
    try:
        assert isinstance(env, MixedVecEnv) and env.num_envs == 6
        assert [e.dynamics for e in env.envs] == ["so_rpy", "first_principles"]
        env.seed(4)
        first = env.reset()
        env.seed(4)
        np.testing.assert_array_equal(first, env.reset())
        obs, rewards, dones, infos = env.step(PID.predict(first)[0])
        assert obs.shape == (6, OBS_DIM) and len(infos) == 6 and np.isfinite(rewards).all()
    finally:
        env.close()
