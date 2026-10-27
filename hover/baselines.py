"""Non-learned reference policies with the same ``predict`` interface as an SB3 model."""

import numpy as np

from hover.env import MAX_ANGLE, thrust_limits

GRAVITY = 9.81


class ZeroPolicy:
    """Always command nominal hover thrust and level attitude."""

    def predict(self, obs, deterministic=True):
        return np.zeros(np.shape(obs)[:-1] + (4,), np.float32), None


class PositionPD:
    """Cascaded PD position feedback onto attitude and collective thrust.

    Desired world acceleration ``a = -kp * position_error - kd * velocity`` is turned into
    roll/pitch targets (at zero yaw, +pitch accelerates along +x and +roll along -y) and a
    collective command scaled from the hover command so the thrust vector tracks
    ``a + g``. Commands are then normalized exactly as HoverEnv.command expects.
    """

    def __init__(self, kp=4.0, kd=3.0, max_tilt=MAX_ANGLE):
        self.kp, self.kd, self.max_tilt = kp, kd, max_tilt
        self.thrust_min, self.thrust_max, self.hover_thrust = thrust_limits()

    def predict(self, obs, deterministic=True):
        single = np.ndim(obs) == 1
        obs = np.atleast_2d(np.asarray(obs, np.float64))
        acc = -self.kp * obs[:, :3] - self.kd * obs[:, 7:10]
        lift = np.maximum(GRAVITY + acc[:, 2], 0.1 * GRAVITY)
        pitch = np.clip(np.arctan2(acc[:, 0], lift), -self.max_tilt, self.max_tilt)
        roll = np.clip(np.arctan2(-acc[:, 1] * np.cos(pitch), lift), -self.max_tilt, self.max_tilt)
        thrust = self.hover_thrust * lift / (GRAVITY * np.cos(roll) * np.cos(pitch))
        thrust = np.clip(thrust, self.thrust_min, self.thrust_max)
        above = thrust >= self.hover_thrust
        thrust_action = np.where(
            above,
            (thrust - self.hover_thrust) / (self.thrust_max - self.hover_thrust),
            (thrust - self.hover_thrust) / (self.hover_thrust - self.thrust_min),
        )
        actions = np.stack(
            [roll / MAX_ANGLE, pitch / MAX_ANGLE, np.zeros_like(roll), thrust_action], axis=1
        ).astype(np.float32)
        actions = np.clip(actions, -1, 1)
        return (actions[0] if single else actions), None
