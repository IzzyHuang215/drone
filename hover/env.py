"""A small, fully observed hover task backed by Crazyflow's fitted dynamics."""

import sys
from contextlib import redirect_stdout

import gymnasium as gym
import jax
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

TARGET = np.array([0.0, 0.0, 1.0], dtype=np.float32)
DT = 0.01  # 100 Hz policy
PHYSICS_FREQ = 500
PHYSICS_STEPS_PER_ACTION = 5
MAX_ANGLE = 0.35  # rad, roll/pitch/yaw target bound
SIM_DEVICES = ("cpu", "gpu")


def require_jax_device(device):
    """Return the first JAX device of a platform, failing clearly if it is unavailable."""
    if device not in SIM_DEVICES:
        raise ValueError(f"simulator device must be one of {SIM_DEVICES}, got {device!r}")
    try:
        return jax.devices(device)[0]
    except RuntimeError as error:
        raise RuntimeError(
            f"JAX has no {device!r} backend here (available: {jax.devices()}). "
            "For NVIDIA GPUs on Linux, install with `uv sync --locked --extra gpu` "
            "and run inside a GPU allocation."
        ) from error


def horizon_steps(episode_seconds):
    if not np.isfinite(episode_seconds) or episode_seconds < DT:
        raise ValueError("episode_seconds must be finite and at least 0.01")
    return int(round(episode_seconds / DT))


def make_sim(n_worlds, device):
    require_jax_device(device)
    return Sim(
        n_worlds=n_worlds,
        n_drones=1,
        drone=Drone.cf2x_L250,
        dynamics=Dynamics.so_rpy,
        control=Control.attitude,
        freq=PHYSICS_FREQ,
        device=device,
    )


def thrust_limits():
    """Collective thrust command bounds and the gravity-compensating hover command."""
    params = load_params(Dynamics.so_rpy, Drone.cf2x_L250)
    thrust_min = float(params["thrust_min"]) * 4
    thrust_max = float(params["thrust_max"]) * 4
    hover_thrust = float(params["mass"]) * 9.81 / float(params["cmd_f_coef"])
    return thrust_min, thrust_max, hover_thrust


class HoverEnv(gym.Env):
    """Hold (0, 0, 1) m with roll, pitch, yaw and collective thrust commands.

    Observations: position error (3), xyzw quaternion (4), world velocity (3),
    body angular velocity (3). Policy frequency is 100 Hz, physics is 500 Hz.
    No position controller sits between the policy and attitude-response dynamics.
    """

    metadata = {"render_modes": []}
    target = TARGET
    dt = DT

    def __init__(self, episode_seconds=5.0, device="cpu"):
        super().__init__()
        self.horizon = horizon_steps(episode_seconds)
        self.device = device
        self.sim = make_sim(1, device)
        self.thrust_min, self.thrust_max, self.hover_thrust = thrust_limits()
        self.action_space = spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(13,), dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        pos = self.target + self.np_random.uniform(-0.2, 0.2, size=3)
        vel = self.np_random.uniform(-0.1, 0.1, size=3)
        states = self.sim.data.states.replace(
            pos=jax.device_put(
                jnp.asarray(pos, dtype=jnp.float32).reshape(1, 1, 3), self.sim.device
            ),
            vel=jax.device_put(
                jnp.asarray(vel, dtype=jnp.float32).reshape(1, 1, 3), self.sim.device
            ),
        )
        self.sim.data = self.sim.data.replace(states=states)
        self.steps = 0
        return self._observation(), {}

    def command(self, action):
        """Map normalized actions to bounded radians and collective thrust commands."""
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
            [*(action[:3] * MAX_ANGLE), self.hover_thrust + action[3] * scale], dtype=np.float32
        )

    def step(self, action):
        cmd = self.command(action)
        self.sim.attitude_control(
            jax.device_put(jnp.asarray(cmd).reshape(1, 1, 4), self.sim.device)
        )
        self.sim.step(PHYSICS_STEPS_PER_ACTION)
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
        # One device-to-host transfer per step, also when the simulator runs on a GPU.
        states = jax.device_get(self.sim.data.states)
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
