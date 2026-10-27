"""Run a trained suite policy outside the simulator, e.g. on a laptop flying a real Crazyflie.

``export`` writes the deterministic policy network as plain arrays plus every constant needed
to reproduce the training interface (``policy.npz``). ``PolicyRunner`` needs only NumPy: give it
the drone's state estimate and the reference each 10 ms control step and it returns the
normalized action and the attitude/thrust setpoint for the onboard controller. It rebuilds the
observation exactly as the simulator does, including the previous action and the integral term.
"""

import json
from pathlib import Path

import numpy as np

from hover.env import DT, MAX_ANGLE
from hover.suite.env import (
    INTEGRAL_LIMIT,
    INTEGRAL_ZONE,
    OBSERVATION,
    action_to_command,
    build_observation,
    drone_constants,
    integrate_error,
)
from hover.suite.tasks import LOOKAHEAD_SECONDS


def export(checkpoint, output, drone="cf2x_L250", dynamics="so_rpy"):
    """Write the actor of an SB3 PPO checkpoint and the interface constants to an .npz file."""
    import torch
    from stable_baselines3 import PPO

    policy = PPO.load(checkpoint, device="cpu").policy
    linears = [m for m in policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    activations = {type(m).__name__ for m in policy.mlp_extractor.policy_net} - {"Linear"}
    if activations != {"Tanh"}:
        raise ValueError(f"only Tanh MLP policies can be exported, found {activations}")
    layers = [*linears, policy.action_net]
    arrays = {}
    for i, layer in enumerate(layers):
        arrays[f"w{i}"] = layer.weight.detach().numpy().astype(np.float32)
        arrays[f"b{i}"] = layer.bias.detach().numpy().astype(np.float32)
    meta = {
        "checkpoint": str(checkpoint),
        "layers": len(layers),
        "activation": "tanh",
        "observation": OBSERVATION,
        "action": ["roll", "pitch", "yaw", "thrust"],
        "action_range": [-1.0, 1.0],
        "max_angle_rad": MAX_ANGLE,
        "dt": DT,
        "lookahead_seconds": LOOKAHEAD_SECONDS,
        "integral_zone_m": INTEGRAL_ZONE,
        "integral_limit": INTEGRAL_LIMIT,
        "drone": drone,
        "dynamics": dynamics,
        "constants": drone_constants(drone, dynamics),
    }
    np.savez(output, meta=json.dumps(meta), **arrays)


class NumpyPolicy:
    """The exported deterministic actor: tanh MLP, then the action head, clipped to [-1, 1]."""

    def __init__(self, path):
        data = np.load(Path(path))
        self.meta = json.loads(str(data["meta"]))
        n = self.meta["layers"]
        self.layers = [(data[f"w{i}"], data[f"b{i}"]) for i in range(n)]

    def __call__(self, obs):
        x = np.asarray(obs, np.float32)
        for w, b in self.layers[:-1]:
            x = np.tanh(x @ w.T + b)
        w, b = self.layers[-1]
        return np.clip(x @ w.T + b, -1.0, 1.0)

    def predict(self, obs, deterministic=True):
        return self(obs), None


class PolicyRunner:
    """Stateful 100 Hz control step around a NumpyPolicy.

    Each call takes the state estimate (position m, velocity m/s, xyzw quaternion, body angular
    velocity rad/s) and the reference (position and velocity now, and the position
    ``lookahead_seconds`` ahead or the next waypoint). Call ``reset()`` before each flight.
    """

    def __init__(self, path):
        self.policy = NumpyPolicy(path)
        self.constants = self.policy.meta["constants"]
        self.reset()

    def reset(self):
        self.prev_action = np.zeros(4, np.float32)
        self.integral = np.zeros(3, np.float32)
        self.started = False

    def observation(self, pos, vel, quat, gyro, ref_pos, ref_vel, ref_ahead):
        pos, ref_pos = np.asarray(pos, np.float32), np.asarray(ref_pos, np.float32)
        err = pos - ref_pos
        if self.started:  # the simulator integrates after each step, not at reset
            self.integral = integrate_error(self.integral, err).astype(np.float32)
        return build_observation(
            err,
            np.asarray(vel, np.float32) - np.asarray(ref_vel, np.float32),
            np.asarray(quat, np.float32),
            np.asarray(gyro, np.float32),
            self.prev_action,
            np.asarray(ref_ahead, np.float32) - ref_pos,
            pos[2],
            self.integral,
        )

    def __call__(self, pos, vel, quat, gyro, ref_pos, ref_vel, ref_ahead):
        """Return (normalized action, {roll, pitch, yaw: rad; thrust: N})."""
        obs = self.observation(pos, vel, quat, gyro, ref_pos, ref_vel, ref_ahead)
        action = self.policy(obs)
        self.prev_action = action.astype(np.float32)
        self.started = True
        c = self.constants
        roll, pitch, yaw, thrust = action_to_command(
            action, c["thrust_min"], c["thrust_max"], c["hover_thrust"]
        )
        return action, {
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "thrust": float(thrust),
        }
