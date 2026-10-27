"""The task curriculum, from holding a point to flying a waypoint course.

Every task shares one observation and action interface, so a policy trained on an easy task
can warm-start the next one and a single generalist policy can be trained on a mixture.
A task is data: how episodes start, how the reference moves, and which disturbances act.
Sim-to-real robustness (parameter randomization, sensor noise, latency) is a separate
profile applied on top of any task; see ``ROBUSTNESS``.
"""

from dataclasses import dataclass, fields

import numpy as np

# Reference motion types. All references are anchored at ORIGIN except landing (ground pad).
HOLD, POINT, CIRCLE, FIGURE8, LANDING, WAYPOINTS = range(6)
REFERENCES = {
    "hold": HOLD,
    "point": POINT,
    "circle": CIRCLE,
    "figure8": FIGURE8,
    "landing": LANDING,
    "waypoints": WAYPOINTS,
}
ORIGIN = np.array([0.0, 0.0, 1.0], np.float32)
MAX_WAYPOINTS = 4
LOOKAHEAD_SECONDS = 0.5
# Trajectories ramp their speed up over this time so they can start from hover.
RAMP_SECONDS = 1.0
# Landing references slow to FLARE_SPEED for the last FLARE_HEIGHT metres.
FLARE_HEIGHT = 0.25
FLARE_SPEED = 0.15


@dataclass(frozen=True)
class TaskSpec:
    name: str
    level: int
    summary: str
    reference: str
    episode_seconds: float = 5.0
    # Start state, relative to the reference at t = 0. Uniform within +-value; tilt about a
    # random horizontal axis, yaw about z.
    start_pos: float = 0.2
    start_vel: float = 0.1
    start_tilt: float = 0.0
    start_yaw: float = 0.0
    start_ang_vel: float = 0.0
    # "point": target offset from ORIGIN, uniform within +-(xy, xy, z).
    point_xy: float = 0.0
    point_z: float = 0.0
    # "circle" / "figure8": size (m) and loop period (s), uniform in [min, max].
    radius_min: float = 0.5
    radius_max: float = 0.5
    period_min: float = 6.0
    period_max: float = 6.0
    # "landing": start height above the pad and reference descent speed.
    height_min: float = 1.0
    height_max: float = 1.0
    descent_min: float = 0.3
    descent_max: float = 0.3
    # "waypoints": number, sampling box around ORIGIN, and capture radius.
    n_waypoints: int = MAX_WAYPOINTS
    waypoint_xy: float = 1.0
    waypoint_z: float = 0.4
    waypoint_radius: float = 0.15
    # Task disturbances, as fractions of the drone's weight: Ornstein-Uhlenbeck gusts
    # (std) and a constant per-episode wind (max); pushes are random velocity kicks.
    gust: float = 0.0
    wind: float = 0.0
    push_rate: float = 0.0
    push_speed: float = 0.0
    # Leaving the reference by more than this terminates the episode as a failure.
    max_error: float = 2.0
    # Success (besides not crashing): mean error below success_error over the final second
    # ("final") or over the whole episode after the first second ("tracking"). Landing and
    # waypoints have their own criteria (soft touchdown; every waypoint captured).
    success_error: float = 0.1
    success_metric: str = "final"


TASKS = {
    task.name: task
    for task in [
        TaskSpec(
            "hover",
            1,
            "Hold a point from small upright offsets (the original task).",
            "hold",
        ),
        TaskSpec(
            "hover-wide",
            2,
            "Hold a point from larger offsets, velocities, tilts and any heading.",
            "hold",
            start_pos=0.5,
            start_vel=0.5,
            start_tilt=np.radians(20),
            start_yaw=np.pi,
            start_ang_vel=1.0,
        ),
        TaskSpec(
            "reach",
            3,
            "Fly to a random target up to 1 m away in each direction and stop there.",
            "point",
            episode_seconds=6.0,
            point_xy=1.0,
            point_z=0.5,
            max_error=3.0,
        ),
        TaskSpec(
            "recover",
            4,
            "Catch a thrown drone: up to 60 deg tilt, 6 rad/s spin and 2 m/s velocity.",
            "hold",
            start_pos=0.3,
            start_vel=2.0,
            start_tilt=np.radians(60),
            start_yaw=np.pi,
            start_ang_vel=6.0,
            max_error=3.0,
        ),
        TaskSpec(
            "disturbance",
            5,
            "Hold position in gusty wind with random 0.75 m/s pushes.",
            "hold",
            episode_seconds=8.0,
            gust=0.1,
            wind=0.1,
            push_rate=0.3,
            push_speed=0.75,
            success_error=0.25,
            success_metric="tracking",
        ),
        TaskSpec(
            "circle",
            6,
            "Track a horizontal circle (0.3-1 m radius, 5-8 s per loop, either direction).",
            "circle",
            episode_seconds=10.0,
            start_pos=0.1,
            radius_min=0.3,
            radius_max=1.0,
            period_min=5.0,
            period_max=8.0,
            success_error=0.15,
            success_metric="tracking",
        ),
        TaskSpec(
            "figure8",
            7,
            "Track a figure-eight (0.3-1 m half-width, 6-10 s per loop).",
            "figure8",
            episode_seconds=12.0,
            start_pos=0.1,
            radius_min=0.3,
            radius_max=1.0,
            period_min=6.0,
            period_max=10.0,
            success_error=0.15,
            success_metric="tracking",
        ),
        TaskSpec(
            "landing",
            8,
            "Descend 0.5-1.5 m onto a pad, slowing near the ground, and touch down gently.",
            "landing",
            episode_seconds=8.0,
            start_pos=0.3,
            start_vel=0.2,
            height_min=0.5,
            height_max=1.5,
            descent_min=0.25,
            descent_max=0.5,
        ),
        TaskSpec(
            "waypoints",
            9,
            "Fly through four random waypoints in order within 12 s.",
            "waypoints",
            episode_seconds=12.0,
            start_pos=0.1,
            max_error=3.0,
        ),
    ]
}


@dataclass(frozen=True)
class Robustness:
    """Sim-to-real randomization, resampled per episode (noise per step)."""

    name: str
    thrust_scale: float = 0.0  # +-fraction: mass, battery sag, motor and thrust-map error
    attitude_bias: float = 0.0  # rad, +-: trim / IMU mounting error
    wind: float = 0.0  # fraction of weight, max constant wind
    gust: float = 0.0  # fraction of weight, gust std
    drag: float = 0.0  # N per (m/s), max linear drag
    latency_steps: int = 0  # max action delay in 10 ms policy steps
    attitude_lag: float = 0.0  # s, max time constant of a first-order lag on attitude targets
    attitude_gain: float = 0.0  # +-fraction: achieved vs commanded attitude
    pos_noise: float = 0.0  # m, std
    vel_noise: float = 0.0  # m/s, std
    attitude_noise: float = 0.0  # quaternion component std
    gyro_noise: float = 0.0  # rad/s, std


ROBUSTNESS = {
    "none": Robustness("none"),
    # Scaled to a Crazyflie 2.x with an external positioning system (e.g. Lighthouse) and an
    # offboard computer sending setpoints over Crazyradio.
    "real": Robustness(
        "real",
        thrust_scale=0.15,
        attitude_bias=np.radians(2.0),
        wind=0.05,
        gust=0.03,
        drag=0.02,
        latency_steps=3,
        # The attitude loop of the real drone is the least certain part of the model: the
        # second simulator model responds ~30 ms slower, so cover about twice that.
        attitude_lag=0.06,
        attitude_gain=0.15,
        pos_noise=0.01,
        vel_noise=0.05,
        attitude_noise=0.01,
        gyro_noise=0.05,
    ),
}
# Wider timing uncertainty: up to 50 ms command latency and a 100 ms attitude lag.
ROBUSTNESS["wide"] = Robustness(
    **{**ROBUSTNESS["real"].__dict__, "name": "wide", "latency_steps": 5, "attitude_lag": 0.1}
)

# Numeric task table for the jitted environment: one row per task, one column per field.
NUMERIC_FIELDS = [
    f.name
    for f in fields(TaskSpec)
    if f.name not in ("name", "level", "summary", "reference", "success_metric")
]


def task_table(tasks):
    rows = [
        [REFERENCES[t.reference]] + [float(getattr(t, name)) for name in NUMERIC_FIELDS]
        for t in tasks
    ]
    return np.asarray(rows, np.float32), ["reference", *NUMERIC_FIELDS]


def resolve(names):
    """Task specs from names; "all" is every task in curriculum order."""
    if names == ["all"]:
        return list(TASKS.values())
    unknown = [n for n in names if n not in TASKS]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; choose from {list(TASKS)} or 'all'")
    return [TASKS[n] for n in names]
