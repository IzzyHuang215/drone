import jax
import jax.numpy as jnp
import numpy as np
import pytest


def has_jax_gpu():
    try:
        return bool(jax.devices("gpu"))
    except RuntimeError:
        return False


def set_world_state(env, world, pos, vel):
    states = env.sim.data.states
    states = states.replace(
        pos=states.pos.at[world, 0].set(jnp.asarray(pos, states.pos.dtype)),
        vel=states.vel.at[world, 0].set(jnp.asarray(vel, states.vel.dtype)),
    )
    env.sim.data = env.sim.data.replace(states=states)


def test_vector_worlds_match_single_environment():
    from hover.env import HoverEnv
    from hover.vec_env import HoverVecEnv

    single, batched = HoverEnv(), HoverVecEnv(3)
    try:
        batched.seed(3)
        batched.reset()
        obs, _ = single.reset(seed=0)
        set_world_state(batched, 0, obs[:3] + single.target, obs[7:10])
        actions = np.random.default_rng(0).uniform(-0.3, 0.3, size=(20, 4)).astype(np.float32)
        for action in actions:
            expected, reward, *_ = single.step(action)
            batch_actions = np.zeros((3, 4), np.float32)
            batch_actions[0] = action
            got, rewards, dones, infos = batched.step(batch_actions)
            np.testing.assert_allclose(got[0], expected, atol=1e-5)
            assert rewards[0] == pytest.approx(reward, abs=1e-4)
            assert not dones.any()
    finally:
        single.close()
        batched.close()


def test_time_limit_resets_each_world_and_bootstraps():
    from hover.vec_env import HoverVecEnv

    env = HoverVecEnv(4, episode_seconds=0.02)
    try:
        env.seed(1)
        env.reset()
        zeros = np.zeros((4, 4), np.float32)
        _, _, dones, _ = env.step(zeros)
        assert not dones.any()
        obs, rewards, dones, infos = env.step(zeros)
        assert dones.all()
        # PPO adds bootstrapped values to truncated rewards in place.
        assert rewards.flags.writeable and obs.flags.writeable
        for i, info in enumerate(infos):
            assert info["TimeLimit.truncated"] is True
            assert info["terminal_observation"].shape == (13,)
            assert not np.allclose(info["terminal_observation"], obs[i])
        _, _, dones, infos = env.step(zeros)
        assert not dones.any()
        assert all("terminal_observation" not in info for info in infos)
    finally:
        env.close()


def test_crash_terminates_only_that_world_without_bootstrapping():
    from hover.vec_env import HoverVecEnv

    env = HoverVecEnv(3)
    try:
        env.seed(2)
        env.reset()
        set_world_state(env, 1, [0.0, 0.0, 0.02], [0.0, 0.0, -1.0])
        _, rewards, dones, infos = env.step(np.zeros((3, 4), np.float32))
        assert dones.tolist() == [False, True, False]
        assert infos[1]["TimeLimit.truncated"] is False
        assert "terminal_observation" in infos[1]
        assert rewards[1] < -9
    finally:
        env.close()


def test_vector_reset_is_seeded_per_world():
    from hover.vec_env import HoverVecEnv

    env = HoverVecEnv(4)
    try:
        env.seed(7)
        first = env.reset()
        env.seed(7)
        second = env.reset()
        np.testing.assert_array_equal(first, second)
        assert len({tuple(row) for row in first[:, :3].round(6)}) == 4
        assert np.all(np.abs(first[:, :3]) <= 0.2 + 1e-6)
    finally:
        env.close()


def test_position_pd_baseline_beats_zero_action():
    from hover.__main__ import evaluate
    from hover.baselines import PositionPD, ZeroPolicy
    from hover.env import HoverEnv

    env = HoverEnv(episode_seconds=2.0)
    try:
        zero = evaluate(ZeroPolicy(), env, 2)
        pd = evaluate(PositionPD(), env, 2)
    finally:
        env.close()
    assert pd["crash_or_boundary_exit_rate"] == 0
    assert pd["mean_position_error_m"] < zero["mean_position_error_m"]


def test_requested_unavailable_devices_fail_clearly():
    import torch

    from hover.__main__ import require_torch_device
    from hover.env import require_jax_device

    with pytest.raises(ValueError):
        require_jax_device("tpu")
    if not has_jax_gpu():
        with pytest.raises(RuntimeError, match="no 'gpu' backend"):
            require_jax_device("gpu")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA was requested"):
            require_torch_device("cuda")


@pytest.mark.skipif(not has_jax_gpu(), reason="needs a JAX GPU backend")
def test_gpu_vector_rollout_matches_cpu():
    from hover.vec_env import HoverVecEnv

    cpu, gpu = HoverVecEnv(8, device="cpu"), HoverVecEnv(8, device="gpu")
    try:
        for env in (cpu, gpu):
            env.seed(11)
        np.testing.assert_allclose(cpu.reset(), gpu.reset(), atol=1e-6)
        actions = np.random.default_rng(1).uniform(-0.3, 0.3, size=(8, 4)).astype(np.float32)
        for _ in range(50):
            a, *_ = cpu.step(actions)
            b, *_ = gpu.step(actions)
        assert np.isfinite(b).all()
        assert {d.platform for d in gpu.sim.data.states.pos.devices()} == {"gpu"}
        np.testing.assert_allclose(a, b, atol=1e-3)
    finally:
        cpu.close()
        gpu.close()
