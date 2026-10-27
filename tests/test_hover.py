import numpy as np
from gymnasium.utils.env_checker import check_env


def test_hover_environment_contract():
    from hover.env import HoverEnv

    env = HoverEnv(episode_seconds=0.02)
    try:
        check_env(env, skip_render_check=True)
        first, _ = env.reset(seed=7)
        second, _ = env.reset(seed=7)
        np.testing.assert_array_equal(first, second)
        assert env.observation_space.contains(first)
        # The episode ends at the configured policy-step horizon.
        _, reward, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
        assert np.isfinite(reward)
        assert not terminated and not truncated
        _, _, _, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
        assert truncated
    finally:
        env.close()


def test_action_mapping_and_invalid_actions():
    from hover.env import HoverEnv

    env = HoverEnv()
    try:
        env.reset(seed=0)
        low = env.command(np.full(4, -1.0))
        high = env.command(np.full(4, 1.0))
        assert np.all(high >= low)
        assert low[-1] > 0
        assert high[-1] > 0
        import pytest

        with pytest.raises(ValueError):
            env.step(np.full(4, np.nan))
        with pytest.raises(ValueError):
            env.step(np.zeros(3))
    finally:
        env.close()
