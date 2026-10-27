"""A small, fully observed hover task backed by Crazyflow's fitted dynamics."""

import sys
from contextlib import redirect_stdout

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

# MuJoCo reports optional Warp imports on stdout. Keep CLI JSON machine-readable
# while preserving those diagnostics on stderr. CPU simulation does not need Warp.
with redirect_stdout(sys.stderr):
    from crazyflow.control import Control
    from crazyflow.drones import Drone
    from crazyflow.dynamics import Dynamics, load_params
    from crazyflow.sim import Sim


class HoverEnv(gym.Env):
    """Hold (0, 0, 1) m with roll, pitch, yaw and collective thrust commands.

    Observations: position error (3), xyzw quaternion (4), world velocity (3),
    body angular velocity (3). Policy frequency is 100 Hz, physics is 500 Hz.
    No position controller sits between the policy and attitude-response dynamics.
    """

    metadata = {"render_modes": []}
    target = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    dt = 0.01

    def __init__(self, episode_seconds=5.0):
        super().__init__()
        if not np.isfinite(episode_seconds) or episode_seconds < self.dt:
            raise ValueError("episode_seconds must be finite and at least 0.01")
        self.horizon = int(round(episode_seconds / self.dt))
        self.sim = Sim(
            n_worlds=1,
            n_drones=1,
            drone=Drone.cf2x_L250,
            dynamics=Dynamics.so_rpy,
            control=Control.attitude,
            freq=500,
            device="cpu",
        )
        params = load_params(Dynamics.so_rpy, Drone.cf2x_L250)
        self.thrust_min = float(params["thrust_min"]) * 4
        self.thrust_max = float(params["thrust_max"]) * 4
        self.hover_thrust = float(params["mass"]) * 9.81 / float(params["cmd_f_coef"])
        self.action_space = spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(13,), dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        pos = self.target + self.np_random.uniform(-0.2, 0.2, size=3)
        vel = self.np_random.uniform(-0.1, 0.1, size=3)
        states = self.sim.data.states.replace(
            pos=jnp.asarray(pos, dtype=jnp.float32).reshape(1, 1, 3),
            vel=jnp.asarray(vel, dtype=jnp.float32).reshape(1, 1, 3),
        )
        self.sim.data = self.sim.data.replace(states=states)
        self.steps = 0
        return self._observation(), {}

    def command(self, action):
        """Map normalized actions to bounded radians and collective Newtons."""
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("action must be four finite values")
        action = np.clip(action, -1, 1)
        scale = (
            self.thrust_max - self.hover_thrust
            if action[3] >= 0
            else self.hover_thrust - self.thrust_min
        )
        return np.array(
            [*(action[:3] * 0.35), self.hover_thrust + action[3] * scale], dtype=np.float32
        )

    def step(self, action):
        cmd = self.command(action)
        self.sim.attitude_control(jnp.asarray(cmd).reshape(1, 1, 4))
        self.sim.step(5)
        self.steps += 1
        obs = self._observation()
        if not np.isfinite(obs).all():
            raise FloatingPointError("Crazyflow produced a non-finite state")
        error = float(np.linalg.norm(obs[:3]))
        altitude = float(self.sim.data.states.pos[0, 0, 2])
        terminated = bool(altitude < 0.05 or error > 2.0)
        truncated = self.steps >= self.horizon
        reward = (
            1.0
            - error**2
            - 0.05 * float(obs[7:10] @ obs[7:10])
            - 0.01 * float(np.square(np.clip(action, -1, 1)).sum())
        )
        if terminated:
            reward -= 10.0
        return obs, reward, terminated, truncated, {"position_error_m": error}

    def _observation(self):
        states = self.sim.data.states
        return np.concatenate(
            [
                np.asarray(states.pos[0, 0]) - self.target,
                np.asarray(states.quat[0, 0]),
                np.asarray(states.vel[0, 0]),
                np.asarray(states.ang_vel[0, 0]),
            ]
        ).astype(np.float32)

    def close(self):
        self.sim.close()
