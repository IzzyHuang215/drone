"""The hover task over Crazyflow's batched worlds, as one Stable-Baselines3 vector env.

Each world is an independent episode. Action mapping, physics, reward, termination and
per-world resets run in a single jitted call on the simulator device, followed by one
device-to-host transfer of the batch. Finished worlds reset in the same step, and their
final observation is returned in ``info["terminal_observation"]`` with
``info["TimeLimit.truncated"]`` so PPO bootstraps time limits but not crashes, matching
Stable-Baselines3's DummyVecEnv convention.

Reset states are drawn from a JAX key seeded through ``seed()``; they reproduce per seed
but are a different random stream from :class:`hover.env.HoverEnv`.
"""

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.sim import functional as F
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from hover.env import (
    MAX_ANGLE,
    PHYSICS_STEPS_PER_ACTION,
    TARGET,
    DT,
    horizon_steps,
    make_sim,
    thrust_limits,
)


def commands(actions, thrust_min, thrust_max, hover_thrust):
    """Batched JAX version of HoverEnv.command for already clipped actions."""
    scale = jnp.where(actions[:, 3] >= 0, thrust_max - hover_thrust, hover_thrust - thrust_min)
    thrust = hover_thrust + actions[:, 3] * scale
    return jnp.concatenate([actions[:, :3] * MAX_ANGLE, thrust[:, None]], axis=1)


def observe(data):
    states = data.states
    return jnp.concatenate(
        [states.pos[:, 0] - TARGET, states.quat[:, 0], states.vel[:, 0], states.ang_vel[:, 0]],
        axis=1,
    ).astype(jnp.float32)


class HoverVecEnv(VecEnv):
    dt = DT
    render_mode = None

    def __init__(self, n_worlds, episode_seconds=5.0, device="cpu"):
        if n_worlds < 1:
            raise ValueError("n_worlds must be positive")
        self.horizon = horizon_steps(episode_seconds)
        self.device = device
        self.sim = make_sim(n_worlds, device)
        limits = thrust_limits()
        sim, horizon = self.sim, self.horizon

        def randomize(data, default_data, mask, key):
            key_pos, key_vel = jax.random.split(key)
            shape = (n_worlds, 1, 3)
            pos = TARGET + jax.random.uniform(key_pos, shape, minval=-0.2, maxval=0.2)
            vel = jax.random.uniform(key_vel, shape, minval=-0.1, maxval=0.1)
            data = sim._reset(data, default_data, mask)
            m = mask[:, None, None]
            states = data.states.replace(
                pos=jnp.where(m, pos, data.states.pos).astype(data.states.pos.dtype),
                vel=jnp.where(m, vel, data.states.vel).astype(data.states.vel.dtype),
            )
            return data.replace(states=states)

        def reset_all(data, default_data, key):
            key, sub = jax.random.split(key)
            data = randomize(data, default_data, jnp.ones(n_worlds, bool), sub)
            return data, key, observe(data)

        def step(data, default_data, steps, key, actions):
            actions = jnp.clip(actions, -1, 1)
            data = F.attitude_control(data, commands(actions, *limits)[:, None, :])
            data = sim._step(data, n_steps=PHYSICS_STEPS_PER_ACTION)
            obs = observe(data)
            error = jnp.linalg.norm(obs[:, :3], axis=1)
            terminated = (data.states.pos[:, 0, 2] < 0.05) | (error > 2.0)
            steps = steps + 1
            truncated = steps >= horizon
            reward = (
                1.0
                - error**2
                - 0.05 * jnp.sum(obs[:, 7:10] ** 2, axis=1)
                - 0.01 * jnp.sum(actions**2, axis=1)
                - 10.0 * terminated
            )
            done = terminated | truncated
            key, sub = jax.random.split(key)
            data = randomize(data, default_data, done, sub)
            steps = jnp.where(done, 0, steps)
            return data, steps, key, (observe(data), obs, reward, terminated, truncated, error)

        self._reset_fn = jax.jit(reset_all)
        self._step_fn = jax.jit(step)
        self._steps = jax.device_put(jnp.zeros(n_worlds, jnp.int32), sim.device)
        self._key = None
        self._actions = None
        super().__init__(
            n_worlds,
            spaces.Box(-np.inf, np.inf, shape=(13,), dtype=np.float32),
            spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
        )

    def reset(self):
        seed = self._seeds[0] if self._seeds and self._seeds[0] is not None else None
        if seed is None:
            seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        self._key = jax.device_put(jax.random.key(seed), self.sim.device)
        self.sim.data, self._key, obs = self._reset_fn(
            self.sim.data, self.sim.default_data, self._key
        )
        self._steps = jnp.zeros_like(self._steps)
        self._reset_seeds()
        return np.array(obs)

    def step_async(self, actions):
        actions = np.asarray(actions, dtype=np.float32).reshape(self.num_envs, 4)
        if not np.isfinite(actions).all():
            raise ValueError("actions must be finite")
        self._actions = jax.device_put(actions, self.sim.device)

    def step_wait(self):
        if self._key is None:
            raise RuntimeError("reset() must be called before step()")
        self.sim.data, self._steps, self._key, outputs = self._step_fn(
            self.sim.data, self.sim.default_data, self._steps, self._key, self._actions
        )
        # device_get gives read-only views; SB3 edits rewards in place to bootstrap time limits.
        obs, final_obs, reward, terminated, truncated, error = (
            np.array(x) for x in jax.device_get(outputs)
        )
        if not np.isfinite(final_obs).all():
            raise FloatingPointError("Crazyflow produced a non-finite state")
        done = terminated | truncated
        infos = [{"position_error_m": float(e)} for e in error]
        for i in np.flatnonzero(done):
            infos[i]["terminal_observation"] = final_obs[i]
            infos[i]["TimeLimit.truncated"] = bool(truncated[i] and not terminated[i])
        return obs, reward.astype(np.float32, copy=False), done, infos

    def close(self):
        self.sim.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("worlds share one simulator; set attributes on the VecEnv")

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError("worlds share one simulator; call methods on the VecEnv")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
