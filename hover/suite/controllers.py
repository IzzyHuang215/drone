"""Classical reference controllers on the suite observation, with an SB3-like ``predict``."""

import numpy as np

from hover.env import MAX_ANGLE
from hover.suite.env import GRAVITY


class TrackingPID:
    """Cascaded PID reference tracking through the onboard attitude controller.

    Desired world acceleration ``a = -kp * error - ki * integral - kd * velocity_error``
    (the integral is the observation's anti-windup integral, the same one a policy sees) becomes a
    tilt of the thrust vector, rotated into the drone's current heading (so it works from any
    yaw), and a collective thrust scaled from the hover command. The yaw target is 0.
    Uses only the observation, so it sees the same noise and latency as a learned policy.
    """

    def __init__(self, constants, kp=4.0, ki=2.0, kd=3.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.thrust_min = constants["thrust_min"]
        self.thrust_max = constants["thrust_max"]
        self.hover = constants["hover_thrust"]

    def predict(self, obs, deterministic=True):
        obs = np.atleast_2d(np.asarray(obs, np.float64))
        rot = obs[:, 6:15].reshape(-1, 3, 3)
        acc = -self.kp * obs[:, :3] - self.ki * obs[:, 26:29] - self.kd * obs[:, 3:6]
        yaw = np.arctan2(rot[:, 1, 0], rot[:, 0, 0])
        ax = np.cos(yaw) * acc[:, 0] + np.sin(yaw) * acc[:, 1]
        ay = -np.sin(yaw) * acc[:, 0] + np.cos(yaw) * acc[:, 1]
        lift = np.maximum(GRAVITY + acc[:, 2], 0.1 * GRAVITY)
        pitch = np.clip(np.arctan2(ax, lift), -MAX_ANGLE, MAX_ANGLE)
        roll = np.clip(np.arctan2(-ay * np.cos(pitch), lift), -MAX_ANGLE, MAX_ANGLE)
        thrust = self.hover * lift / (GRAVITY * np.cos(roll) * np.cos(pitch))
        thrust = np.clip(thrust, self.thrust_min, self.thrust_max)
        thrust_action = np.where(
            thrust >= self.hover,
            (thrust - self.hover) / (self.thrust_max - self.hover),
            (thrust - self.hover) / (self.hover - self.thrust_min),
        )
        actions = np.stack(
            [roll / MAX_ANGLE, pitch / MAX_ANGLE, np.zeros_like(roll), thrust_action], 1
        )
        return np.clip(actions, -1, 1).astype(np.float32), None


class Zero:
    def predict(self, obs, deterministic=True):
        return np.zeros((np.atleast_2d(obs).shape[0], 4), np.float32), None
