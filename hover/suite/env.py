"""Batched Crazyflow worlds running any mix of suite tasks, as one Stable-Baselines3 VecEnv.

Each world samples a task, start state, reference and (optionally) sim-to-real perturbations
at every reset. Action mapping, latency, disturbances, physics, reference tracking, rewards and
same-step resets run in one jitted call on the simulator device, followed by one transfer of
the batch to the host. Finished worlds report ``terminal_observation`` and, for time limits only,
``TimeLimit.truncated``, as in Stable-Baselines3's DummyVecEnv.

Observation (29 floats; see ``OBSERVATION``), all computable on a real drone from its state
estimate, the reference and the previous command:

* position and velocity minus the reference (6)
* body-to-world rotation matrix, row-major (9)
* body angular velocity (3)
* previous normalized action (4)
* reference position 0.5 s ahead minus now, or the next waypoint minus the current one (3)
* altitude (1)
* integrated position error (3), the integral term of a PID: accumulated only within
  ``INTEGRAL_ZONE`` of the reference and clipped to +-``INTEGRAL_LIMIT`` m*s. Without it neither a
  policy nor a PD controller can cancel constant biases such as a weak battery or steady wind.

Actions (4 floats in [-1, 1]) are the same as HoverEnv: roll, pitch and yaw targets (+-0.35 rad)
and collective thrust, with 0 meaning the model's hover thrust. They go to the drone's onboard
attitude controller, matching the Crazyflie's attitude-setpoint interface.
"""

import sys
from contextlib import redirect_stdout

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from hover.env import DT, MAX_ANGLE, PHYSICS_FREQ, PHYSICS_STEPS_PER_ACTION, require_jax_device
from hover.suite import tasks as T

with redirect_stdout(sys.stderr):
    from crazyflow.control import Control
    from crazyflow.drones import Drone
    from crazyflow.dynamics import Dynamics, load_params
    from crazyflow.sim import Sim
    from crazyflow.sim import functional as F

OBSERVATION = (
    ["err_x", "err_y", "err_z", "err_vx", "err_vy", "err_vz"]
    + [f"R{i}{j}" for i in range(3) for j in range(3)]
    + ["gyro_x", "gyro_y", "gyro_z", "prev_roll", "prev_pitch", "prev_yaw", "prev_thrust"]
    + ["ahead_x", "ahead_y", "ahead_z", "altitude", "int_x", "int_y", "int_z"]
)
OBS_DIM = len(OBSERVATION)
DRONES = [d.value for d in Drone]
# so_rpy is Crazyflow's attitude-response model identified from flight data (training default);
# first_principles runs a firmware-style attitude controller on a rigid body with rotor dynamics
# and is used as an unseen model to measure transfer.
DYNAMICS = ("so_rpy", "first_principles")
GRAVITY = 9.81
GROUND_CRASH_Z = 0.05
TOUCHDOWN_Z = 0.03
SOFT_SPEED = 0.4
SOFT_TILT = np.radians(20)
GUST_TIME_CONSTANT = 1.0
INTEGRAL_ZONE = 0.3
INTEGRAL_LIMIT = 1.0


def drone_constants(drone, dynamics):
    """Thrust bounds, hover command and hover rotor speed for one drone model."""
    params = load_params(Dynamics(dynamics), Drone(drone))
    mass = float(params["mass"])
    weight = mass * GRAVITY
    hover = weight / float(params.get("cmd_f_coef", 1.0))
    hover_rpm = 0.0
    if "rpm2thrust" in params:
        a, b, c = (float(x) for x in params["rpm2thrust"])
        hover_rpm = (-b + np.sqrt(b * b - 4 * c * (a - weight / 4))) / (2 * c)
    return {
        "mass": mass,
        "weight": weight,
        "thrust_min": float(params["thrust_min"]) * 4,
        "thrust_max": float(params["thrust_max"]) * 4,
        "hover_thrust": hover,
        "hover_rpm": hover_rpm,
    }


def action_to_command(actions, thrust_min, thrust_max, hover_thrust):
    """Normalized actions -> [roll, pitch, yaw (rad), collective thrust command]."""
    xp = jnp if isinstance(actions, jax.Array) else np
    actions = xp.clip(actions, -1, 1)
    scale = xp.where(actions[..., 3] >= 0, thrust_max - hover_thrust, hover_thrust - thrust_min)
    thrust = hover_thrust + actions[..., 3] * scale
    return xp.concatenate([actions[..., :3] * MAX_ANGLE, thrust[..., None]], axis=-1)


def quat_to_matrix(q):
    """xyzw unit quaternions (..., 4) -> rotation matrices (..., 3, 3), body to world."""
    xp = jnp if isinstance(q, jax.Array) else np
    x, y, z, w = (q[..., i] for i in range(4))
    return xp.stack(
        [
            xp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            xp.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            xp.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def build_observation(err_pos, err_vel, quat, gyro, prev_action, ahead, altitude, integral):
    """Assemble observations; shared by simulation and deployment (NumPy or JAX)."""
    xp = jnp if isinstance(quat, jax.Array) else np
    rot = quat_to_matrix(quat).reshape(*quat.shape[:-1], 9)
    return xp.concatenate(
        [err_pos, err_vel, rot, gyro, prev_action, ahead, altitude[..., None], integral], axis=-1
    ).astype(xp.float32)


def integrate_error(integral, err_pos, dt=DT):
    """PID-style integral with conditional integration (anti-windup) and clipping."""
    xp = jnp if isinstance(err_pos, jax.Array) else np
    inside = xp.linalg.norm(err_pos, axis=-1, keepdims=True) < INTEGRAL_ZONE
    return xp.clip(integral + xp.where(inside, err_pos * dt, 0.0), -INTEGRAL_LIMIT, INTEGRAL_LIMIT)


def quat_multiply(a, b):
    ax, ay, az, aw = (a[..., i] for i in range(4))
    bx, by, bz, bw = (b[..., i] for i in range(4))
    return jnp.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        -1,
    )


def _ramp_angle(t, omega):
    """Phase and phase rate of a loop that speeds up linearly over RAMP_SECONDS."""
    r = T.RAMP_SECONDS
    angle = jnp.where(t < r, omega * t * t / (2 * r), omega * (t - r / 2))
    rate = omega * jnp.minimum(t / r, 1.0)
    return angle, rate


def reference(ep, t):
    """Reference position and velocity for every world at time t (s)."""
    kind = ep["ref"][:, None]
    zero = jnp.zeros_like(ep["anchor"])
    anchor = ep["anchor"]
    radius = ep["radius"][:, None]
    angle, rate = _ramp_angle(t, ep["omega"])
    angle, rate = (ep["sign"] * angle)[:, None], (ep["sign"] * rate)[:, None]
    phase = ep["phase"][:, None]
    th = phase + angle
    circle = anchor + radius * jnp.concatenate(
        [jnp.cos(th) - jnp.cos(phase), jnp.sin(th) - jnp.sin(phase), jnp.zeros_like(th)], -1
    )
    circle_v = radius * rate * jnp.concatenate([-jnp.sin(th), jnp.cos(th), jnp.zeros_like(th)], -1)
    eight = anchor + radius * jnp.concatenate(
        [jnp.sin(angle), jnp.sin(angle) * jnp.cos(angle), jnp.zeros_like(angle)], -1
    )
    eight_v = (
        radius
        * rate
        * jnp.concatenate([jnp.cos(angle), jnp.cos(2 * angle), jnp.zeros_like(angle)], -1)
    )
    # Descend at the episode's speed to FLARE_HEIGHT, then at FLARE_SPEED to the pad.
    fast = jnp.maximum(ep["height"] - T.FLARE_HEIGHT, 0.0) / ep["descent"]
    flare_start = jnp.minimum(ep["height"], T.FLARE_HEIGHT)
    height = jnp.where(
        t < fast,
        ep["height"] - ep["descent"] * t,
        jnp.maximum(flare_start - T.FLARE_SPEED * (t - fast), 0.0),
    )
    sink = jnp.where(t < fast, ep["descent"], jnp.where(height > 0, T.FLARE_SPEED, 0.0))
    height, sink = height[:, None], sink[:, None]
    landing = anchor + jnp.concatenate([jnp.zeros_like(height)] * 2 + [height], -1)
    landing_v = jnp.concatenate([jnp.zeros_like(height)] * 2 + [-sink], -1)
    # Planar sine-series path, rotated about the vertical through ORIGIN.
    tau = ep["path_T"]
    sp = jnp.clip(t / tau, 0.0, 1.0)[:, None]
    j = jnp.arange(1, T.PATH_MODES + 1, dtype=jnp.float32)
    chord = jnp.asarray(T.PATH_GOAL, jnp.float32) - jnp.asarray(T.PATH_START, jnp.float32)
    planar = (
        jnp.asarray(T.PATH_START, jnp.float32)
        + sp * chord
        + jnp.einsum("nj,njk->nk", jnp.sin(jnp.pi * sp * j), ep["path_c"])
    )
    moving = (t < tau)[:, None]
    planar_v = jnp.where(
        moving,
        (chord + jnp.einsum("nj,njk->nk", j * jnp.pi * jnp.cos(jnp.pi * sp * j), ep["path_c"]))
        / tau[:, None],
        0.0,
    )
    cos, sin = jnp.cos(ep["path_rot"])[:, None], jnp.sin(ep["path_rot"])[:, None]

    def rotate(xy):
        return jnp.concatenate(
            [cos * xy[:, :1] - sin * xy[:, 1:], sin * xy[:, :1] + cos * xy[:, 1:]], -1
        )

    path = anchor + jnp.concatenate([rotate(planar), jnp.zeros_like(height)], -1)
    path_v = jnp.concatenate([rotate(planar_v), jnp.zeros_like(height)], -1)
    n = ep["n_wp"]
    idx = jnp.minimum(ep["wp_idx"], n - 1)
    wps = ep["wps"]
    waypoint = jnp.take_along_axis(wps, idx[:, None, None], 1)[:, 0]
    choose = [
        (kind == T.CIRCLE, circle, circle_v),
        (kind == T.FIGURE8, eight, eight_v),
        (kind == T.LANDING, landing, landing_v),
        (kind == T.WAYPOINTS, waypoint, zero),
        (kind == T.PATH, path, path_v),
    ]
    pos, vel = anchor, zero
    for mask, p, v in choose:
        pos, vel = jnp.where(mask, p, pos), jnp.where(mask, v, vel)
    return pos, vel


def lookahead(ep, t, pos_now):
    """Where the reference will be LOOKAHEAD_SECONDS ahead, relative to now."""
    ahead, _ = reference(ep, t + T.LOOKAHEAD_SECONDS)
    nxt = jnp.minimum(ep["wp_idx"] + 1, ep["n_wp"] - 1)
    next_wp = jnp.take_along_axis(ep["wps"], nxt[:, None, None], 1)[:, 0]
    is_wp = (ep["ref"] == T.WAYPOINTS)[:, None]
    return jnp.where(is_wp, next_wp, ahead) - pos_now


class SuiteVecEnv(VecEnv):
    dt = DT
    render_mode = None

    def __init__(
        self,
        tasks,
        n_worlds,
        robustness="none",
        device="cpu",
        drone="cf2x_L250",
        dynamics="so_rpy",
        smoothness=0.05,
    ):
        if n_worlds < 1:
            raise ValueError("n_worlds must be positive")
        if dynamics not in DYNAMICS:
            raise ValueError(f"dynamics must be one of {DYNAMICS}")
        self.tasks = list(tasks)
        self.robustness = T.ROBUSTNESS[robustness]
        self.device, self.drone, self.dynamics = device, drone, dynamics
        require_jax_device(device)
        self.sim = Sim(
            n_worlds=n_worlds,
            n_drones=1,
            drone=Drone(drone),
            dynamics=Dynamics(dynamics),
            control=Control.attitude,
            freq=PHYSICS_FREQ,
            device=device,
        )
        self.constants = c = drone_constants(drone, dynamics)
        table, columns = T.task_table(self.tasks)
        self.horizons = np.array([round(t.episode_seconds / DT) for t in self.tasks])
        self.smoothness = smoothness
        self._fns = self._build(n_worlds, jnp.asarray(table), columns, c, self.robustness)
        self._key = self._ep = None
        self.set_paths(None)
        self._actions = None
        super().__init__(
            n_worlds,
            spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32),
            spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
        )

    def _build(self, n, table, columns, c, rob):
        sim, col = self.sim, {name: i for i, name in enumerate(columns)}
        n_tasks, lat = table.shape[0], rob.latency_steps
        weight, k_wp = c["weight"], T.MAX_WAYPOINTS
        uniform = jax.random.uniform

        def field(rows, name):
            return rows[:, col[name]]

        def sample_episode(key, mask, ep, override):
            k_path = jax.random.fold_in(key, 7)  # separate stream: other tasks' draws unchanged
            keys = iter(jax.random.split(key, 24))
            task = jax.random.randint(next(keys), (n,), 0, n_tasks)
            rows = table[task]

            def box(half, shape=(n, 3)):
                return uniform(next(keys), shape, minval=-1.0, maxval=1.0) * half

            def between(lo, hi):
                return field(rows, lo) + uniform(next(keys), (n,)) * (
                    field(rows, hi) - field(rows, lo)
                )

            kind = field(rows, "reference").astype(jnp.int32)
            point = jnp.stack(
                [field(rows, "point_xy"), field(rows, "point_xy"), field(rows, "point_z")], -1
            )
            anchor = jnp.where(
                (kind == T.POINT)[:, None], T.ORIGIN + box(point), jnp.asarray(T.ORIGIN)
            )
            # The landing pad sits on the ground below the origin.
            anchor = jnp.where((kind == T.LANDING)[:, None], 0.0, anchor)
            wp_box = jnp.stack(
                [field(rows, "waypoint_xy"), field(rows, "waypoint_xy"), field(rows, "waypoint_z")],
                -1,
            )[:, None]
            new = {
                "task": task,
                "ref": kind,
                "t": jnp.zeros(n, jnp.int32),
                "horizon": jnp.round(field(rows, "episode_seconds") / DT).astype(jnp.int32),
                "anchor": anchor,
                "radius": between("radius_min", "radius_max"),
                "omega": 2 * jnp.pi / between("period_min", "period_max"),
                "sign": jnp.where(uniform(next(keys), (n,)) < 0.5, -1.0, 1.0),
                "phase": uniform(next(keys), (n,), maxval=2 * jnp.pi),
                "height": between("height_min", "height_max"),
                "descent": between("descent_min", "descent_max"),
                **sample_path(k_path, rows, override),
                "wps": T.ORIGIN
                + uniform(next(keys), (n, k_wp, 3), minval=-1.0, maxval=1.0) * wp_box,
                "wp_idx": jnp.zeros(n, jnp.int32),
                "n_wp": field(rows, "n_waypoints").astype(jnp.int32),
                "wp_radius": field(rows, "waypoint_radius"),
                "max_error": field(rows, "max_error"),
                "push_p": field(rows, "push_rate") * DT,
                "push_speed": field(rows, "push_speed"),
                "gust_std": weight * jnp.sqrt(field(rows, "gust") ** 2 + rob.gust**2),
                "wind_force": weight
                * box((field(rows, "wind") + rob.wind)[:, None])
                * jnp.array([1.0, 1.0, 0.3]),
                "drag": uniform(next(keys), (n,)) * rob.drag,
                "thrust_scale": 1.0 + box(rob.thrust_scale, (n,)),
                "att_bias": box(rob.attitude_bias) * jnp.array([1.0, 1.0, 0.0]),
                "latency": jax.random.randint(next(keys), (n,), 0, lat + 1),
                "att_filter": jnp.zeros((n, 3)),
                "gust": jnp.zeros((n, 3)),
                "buf": jnp.zeros((n, lat + 1, 4)),
                "prev_action": jnp.zeros((n, 4)),
                "integral": jnp.zeros((n, 3)),
            }
            ref_pos, ref_vel = reference(new, jnp.zeros(n))
            start = {
                "pos": ref_pos + box(field(rows, "start_pos")[:, None]),
                "vel": ref_vel + box(field(rows, "start_vel")[:, None]),
                "ang_vel": box(field(rows, "start_ang_vel")[:, None]),
            }
            start["pos"] = start["pos"].at[:, 2].set(jnp.maximum(start["pos"][:, 2], 0.2))
            axis_angle = uniform(next(keys), (n,), maxval=2 * jnp.pi)
            tilt = uniform(next(keys), (n,)) * field(rows, "start_tilt")
            yaw = box(field(rows, "start_yaw"), (n,))
            q_tilt = jnp.stack(
                [
                    jnp.cos(axis_angle) * jnp.sin(tilt / 2),
                    jnp.sin(axis_angle) * jnp.sin(tilt / 2),
                    jnp.zeros(n),
                    jnp.cos(tilt / 2),
                ],
                -1,
            )
            q_yaw = jnp.stack([jnp.zeros(n), jnp.zeros(n), jnp.sin(yaw / 2), jnp.cos(yaw / 2)], -1)
            start["quat"] = quat_multiply(q_yaw, q_tilt)
            # Drawn last so that adding them left every earlier draw (and so every evaluation
            # episode's start) unchanged.
            new["att_tau"] = uniform(next(keys), (n,)) * rob.attitude_lag
            new["att_gain"] = 1.0 + box(rob.attitude_gain, (n,))
            if ep is None:
                return new, start
            pick = lambda a, b: jnp.where(mask.reshape((n,) + (1,) * (a.ndim - 1)), a, b)  # noqa: E731
            return jax.tree.map(pick, new, ep), start

        def sample_path(key, rows, override):
            k_c1, k_c, k_t, k_r = jax.random.split(key, 4)
            scale = (
                field(rows, "path_std")[:, None, None]
                / jnp.arange(1, T.PATH_MODES + 1)[None, :, None]
            )
            coef = jax.random.normal(k_c, (n, T.PATH_MODES, 2)) * scale
            c1 = (2 * jax.random.uniform(k_c1, (n,)) - 1) * field(rows, "path_c1")
            coef = coef.at[:, 0, 1].add(c1)
            tau = field(rows, "path_T_min") + jax.random.uniform(k_t, (n,)) * (
                field(rows, "path_T_max") - field(rows, "path_T_min")
            )
            rot = field(rows, "path_rotate") * jax.random.uniform(k_r, (n,), maxval=2 * jnp.pi)
            use = override["use"]
            return {
                "path_c": jnp.where(use, override["c"], coef),
                "path_T": jnp.where(use, override["T"], tau),
                "path_rot": jnp.where(use, override["rot"], rot),
            }

        def apply_start(data, default_data, mask, start):
            data = sim._reset(data, default_data, mask)
            s = data.states
            m = mask[:, None, None]
            states = s.replace(
                pos=jnp.where(m, start["pos"][:, None], s.pos).astype(s.pos.dtype),
                vel=jnp.where(m, start["vel"][:, None], s.vel).astype(s.vel.dtype),
                quat=jnp.where(m, start["quat"][:, None], s.quat).astype(s.quat.dtype),
                ang_vel=jnp.where(m, start["ang_vel"][:, None], s.ang_vel).astype(s.ang_vel.dtype),
            )
            if c["hover_rpm"]:
                rpm = jnp.full_like(s.rotor_vel, c["hover_rpm"])
                states = states.replace(rotor_vel=jnp.where(m, rpm, s.rotor_vel))
            return data.replace(states=states)

        def observe(data, ep, key):
            s = data.states
            t = ep["t"] * DT
            ref_pos, ref_vel = reference(ep, t)
            pos, vel = s.pos[:, 0], s.vel[:, 0]
            k1, k2, k3, k4 = jax.random.split(key, 4)
            noisy_pos = pos + rob.pos_noise * jax.random.normal(k1, pos.shape)
            noisy_vel = vel + rob.vel_noise * jax.random.normal(k2, vel.shape)
            quat = s.quat[:, 0] + rob.attitude_noise * jax.random.normal(k3, (n, 4))
            quat = quat / jnp.linalg.norm(quat, axis=-1, keepdims=True)
            gyro = s.ang_vel[:, 0] + rob.gyro_noise * jax.random.normal(k4, (n, 3))
            return build_observation(
                noisy_pos - ref_pos,
                noisy_vel - ref_vel,
                quat,
                gyro,
                ep["prev_action"],
                lookahead(ep, t, ref_pos),
                noisy_pos[:, 2],
                ep["integral"],
            )

        def reset_all(data, default_data, key, override):
            key, k_ep, k_obs = jax.random.split(key, 3)
            ep, start = sample_episode(k_ep, None, None, override)
            data = apply_start(data, default_data, jnp.ones(n, bool), start)
            return data, ep, key, observe(data, ep, k_obs)

        def step(data, default_data, ep, key, actions, override):
            key, k_gust, k_push, k_dir, k_reset, k_obs, k_final = jax.random.split(key, 7)
            actions = jnp.clip(actions, -1, 1)
            buf = jnp.concatenate([actions[:, None], ep["buf"][:, :-1]], 1)
            applied = buf[jnp.arange(n), ep["latency"]]
            cmd = action_to_command(applied, c["thrust_min"], c["thrust_max"], c["hover_thrust"])
            # Attitude-response uncertainty: gain error and a first-order lag on the targets.
            alpha = DT / (ep["att_tau"] + DT)
            att = ep["att_filter"] + alpha[:, None] * (
                cmd[:, :3] * ep["att_gain"][:, None] - ep["att_filter"]
            )
            cmd = jnp.concatenate(
                [att + ep["att_bias"], cmd[:, 3:] * ep["thrust_scale"][:, None]], 1
            )
            pos_before = data.states.pos[:, 0]
            data = F.attitude_control(data, cmd[:, None])
            decay = DT / GUST_TIME_CONSTANT
            gust = ep["gust"] * (1 - decay) + ep["gust_std"][:, None] * jnp.sqrt(
                2 * decay
            ) * jax.random.normal(k_gust, (n, 3))
            vel = data.states.vel[:, 0]
            force = ep["wind_force"] + gust - ep["drag"][:, None] * vel
            data = data.replace(states=data.states.replace(force=force[:, None]))
            data = sim._step(data, n_steps=PHYSICS_STEPS_PER_ACTION)
            direction = jax.random.normal(k_dir, (n, 3))
            direction = direction / jnp.linalg.norm(direction, axis=-1, keepdims=True)
            pushed = jax.random.uniform(k_push, (n,)) < ep["push_p"]
            kick = jnp.where(pushed[:, None], direction * ep["push_speed"][:, None], 0.0)
            data = data.replace(states=data.states.replace(vel=data.states.vel + kick[:, None]))

            s = data.states
            pos, vel, quat = s.pos[:, 0], s.vel[:, 0], s.quat[:, 0]
            t = ep["t"] + 1
            ep = {**ep, "t": t, "gust": gust, "buf": buf, "att_filter": att}
            ref_pos, ref_vel = reference(ep, t * DT)
            is_wp = ep["ref"] == T.WAYPOINTS
            en_route = is_wp & (ep["wp_idx"] < ep["n_wp"])
            # Distance closed toward the current waypoint this step (potential-based shaping).
            progress = jnp.linalg.norm(pos_before - ref_pos, axis=-1) - jnp.linalg.norm(
                pos - ref_pos, axis=-1
            )
            reached = en_route & (jnp.linalg.norm(pos - ref_pos, axis=-1) < ep["wp_radius"])
            ep["wp_idx"] = ep["wp_idx"] + reached
            ref_pos, ref_vel = reference(ep, t * DT)
            err = jnp.linalg.norm(pos - ref_pos, axis=-1)
            err_vel = jnp.sum((vel - ref_vel) ** 2, -1)
            up = quat_to_matrix(quat)[:, 2, 2]
            altitude = pos[:, 2]
            speed = jnp.linalg.norm(vel, axis=-1)
            is_landing = ep["ref"] == T.LANDING
            touch = is_landing & (altitude < TOUCHDOWN_Z)
            horizontal = jnp.linalg.norm((pos - ep["anchor"])[:, :2], axis=-1)
            soft = (speed < SOFT_SPEED) & (up > np.cos(SOFT_TILT)) & (horizontal < 0.2)
            landed = touch & soft
            crashed = (
                (~is_landing & (altitude < GROUND_CRASH_Z))
                | (touch & ~soft)
                | (err > ep["max_error"])
            )
            terminated = crashed | landed
            truncated = t >= ep["horizon"]
            # Landing has no per-step survival reward, so touching down beats hovering.
            alive = jnp.where(is_landing, 0.0, 1.0)
            # While waypoints remain, reward progress instead of closeness: closeness would make
            # capturing a waypoint (which switches to a distant target) look like a loss.
            tracking = jnp.where(
                en_route,
                20.0 * progress,
                alive * 0.5 * jnp.exp(-((err / 0.1) ** 2))
                - 0.5 * jnp.tanh(err)
                - 0.05 * jnp.minimum(err_vel, 10.0),
            )
            reward = (
                alive
                + tracking
                - 0.01 * jnp.sum(actions**2, -1)
                - self.smoothness * jnp.sum((actions - ep["prev_action"]) ** 2, -1)
                - 0.1 * (1 - up)
                + 5.0 * reached
                + 10.0 * landed
                - 10.0 * crashed
            )
            ep["prev_action"] = actions
            ep["integral"] = integrate_error(ep["integral"], pos - ref_pos)
            final_obs = observe(data, ep, k_final)
            info = {
                "err": err,
                "crashed": crashed,
                "landed": landed,
                "touch_speed": jnp.where(touch, speed, jnp.nan),
                "waypoints": ep["wp_idx"],
                "task": ep["task"],
                "smooth": jnp.sum((actions - buf[:, 1]) ** 2, -1),
                "altitude": altitude,
            }
            done = terminated | truncated
            new_ep, start = sample_episode(k_reset, done, ep, override)
            data = apply_start(data, default_data, done, start)
            obs = observe(data, new_ep, k_obs)
            return data, new_ep, key, (obs, final_obs, reward, terminated, truncated, info)

        return jax.jit(reset_all), jax.jit(step)

    def reset(self):
        seed = self._seeds[0] if self._seeds and self._seeds[0] is not None else None
        if seed is None:
            seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        key = jax.device_put(jax.random.key(seed), self.sim.device)
        self.sim.data, self._ep, self._key, obs = self._fns[0](
            self.sim.data, self.sim.default_data, key, self._override
        )
        self._reset_seeds()
        return np.array(obs)

    def step_arrays(self, actions):
        """Advance all worlds; returns NumPy arrays without building per-world info dicts."""
        if self._key is None:
            raise RuntimeError("reset() must be called before step()")
        actions = jax.device_put(
            np.asarray(actions, np.float32).reshape(self.num_envs, 4), self.sim.device
        )
        self.sim.data, self._ep, self._key, out = self._fns[1](
            self.sim.data, self.sim.default_data, self._ep, self._key, actions, self._override
        )
        obs, final_obs, reward, terminated, truncated, info = jax.tree.map(
            np.array, jax.device_get(out)
        )
        if not np.isfinite(final_obs).all():
            raise FloatingPointError("Crazyflow produced a non-finite state")
        return obs, final_obs, reward, terminated, truncated & ~terminated, info

    def set_paths(self, coefficients, duration=6.0, rotation=0.0):
        """Fly given path coefficients (n_worlds, PATH_MODES, 2) instead of the task prior.

        Applies to every reset until called again; None restores the prior.
        """
        n = self.num_envs if hasattr(self, "num_envs") else self.sim.n_worlds
        use = coefficients is not None
        coef = np.zeros((n, T.PATH_MODES, 2), np.float32)
        if use:
            coef = np.asarray(coefficients, np.float32).reshape(n, T.PATH_MODES, 2)
        put = lambda x: jax.device_put(jnp.asarray(x), self.sim.device)  # noqa: E731
        self._override = {
            "use": put(use),
            "c": put(coef),
            "T": put(np.full(n, duration, np.float32)),
            "rot": put(np.full(n, rotation, np.float32)),
        }

    def step_async(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if not np.isfinite(actions).all():
            raise ValueError("actions must be finite")
        self._actions = actions

    def step_wait(self):
        obs, final_obs, reward, terminated, truncated, info = self.step_arrays(self._actions)
        done = terminated | truncated
        infos = [{"position_error_m": float(e)} for e in info["err"]]
        for i in np.flatnonzero(done):
            infos[i]["terminal_observation"] = final_obs[i]
            infos[i]["TimeLimit.truncated"] = bool(truncated[i])
        return obs, reward.astype(np.float32), done, infos

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


class MixedVecEnv(VecEnv):
    """Several SuiteVecEnvs (e.g. different dynamics models) stepped as one vector env."""

    render_mode = None

    def __init__(self, envs):
        self.envs = list(envs)
        self.sizes = [e.num_envs for e in self.envs]
        first = self.envs[0]
        super().__init__(sum(self.sizes), first.observation_space, first.action_space)

    def reset(self):
        seed = self._seeds[0] if self._seeds and self._seeds[0] is not None else None
        if seed is None:
            seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        for i, env in enumerate(self.envs):
            env.seed(seed + 7919 * i)
        self._reset_seeds()
        return np.concatenate([env.reset() for env in self.envs])

    def step_async(self, actions):
        for env, part in zip(self.envs, np.split(np.asarray(actions), np.cumsum(self.sizes)[:-1])):
            env.step_async(part)

    def step_wait(self):
        results = [env.step_wait() for env in self.envs]
        obs, rewards, dones, infos = zip(*results)
        return (
            np.concatenate(obs),
            np.concatenate(rewards),
            np.concatenate(dones),
            [info for part in infos for info in part],
        )

    def close(self):
        for env in self.envs:
            env.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


def training_env(tasks, n_worlds, dynamics="so_rpy", **kwargs):
    """One model, or ``dynamics="mix"``: half the worlds on each of DYNAMICS."""
    if dynamics != "mix":
        return SuiteVecEnv(tasks, n_worlds, dynamics=dynamics, **kwargs)
    share = n_worlds // len(DYNAMICS)
    return MixedVecEnv(SuiteVecEnv(tasks, share, dynamics=d, **kwargs) for d in DYNAMICS)
