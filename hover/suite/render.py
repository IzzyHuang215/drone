"""Watch one suite episode in the MuJoCo viewer, or save it as a GIF, with the reference drawn.

Green sphere: where the drone should be now. Faint dots: the reference path. Orange spheres:
waypoints still to capture (green once captured). Disc: landing pad.
"""

import time

import jax
import mujoco
import numpy as np

from hover.suite.env import SuiteVecEnv, reference


def _path(ep, horizon, dt, samples=80):
    times = np.linspace(0, horizon * dt, samples)
    return np.stack([np.asarray(reference(ep, np.full(1, t, np.float32))[0][0]) for t in times])


def _markers(env, path, task):
    ep = jax.device_get(env._ep)
    now, _ = reference(env._ep, env._ep["t"] * env.dt)
    marks = [dict(pos=np.asarray(now[0]), size=np.full(3, 0.025), rgba=(0.1, 0.9, 0.3, 0.9))]
    if task.reference == "waypoints":
        for i, wp in enumerate(ep["wps"][0][: task.n_waypoints]):
            done = i < ep["wp_idx"][0]
            color = (0.1, 0.9, 0.3, 0.35) if done else (1.0, 0.55, 0.1, 0.45)
            marks.append(dict(pos=wp, size=np.full(3, task.waypoint_radius), rgba=color))
    elif task.reference == "landing":
        marks.append(
            dict(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=ep["anchor"][0] + np.array([0, 0, 0.002]),
                size=np.array([0.15, 0.15, 0.002]),
                rgba=(0.9, 0.2, 0.2, 0.8),
            )
        )
    if task.reference not in ("hold", "point", "waypoints"):
        marks += [dict(pos=p, size=np.full(3, 0.008), rgba=(1, 1, 1, 0.5)) for p in path[::2]]
    return marks


def render(policy, task, seed, video=None, fps=25, width=640, height=480, **env_kwargs):
    env = SuiteVecEnv([task], 1, **env_kwargs)
    try:
        env.seed(seed)
        obs = env.reset()
        horizon = round(task.episode_seconds / env.dt)
        path = _path(env._ep, horizon, env.dt)
        points = np.concatenate([path, jax.device_get(env.sim.data.states.pos)[0]])
        center = (points.max(0) + points.min(0)) / 2
        extent = float(np.max(np.linalg.norm(points - center, axis=1)))
        camera = {
            "distance": max(0.8, 2.5 * extent + 0.4),
            "azimuth": 135.0,
            "elevation": -25.0,
            "lookat": center,
        }
        mode = "rgb_array" if video else "human"
        stride = max(round(1 / (env.dt * fps)), 1)
        frames, errors = [], []

        def draw():
            sim = env.sim
            if sim.viewer is not None:
                viewer = sim.viewer._get_viewer(mode)
                for marker in _markers(env, path, task):
                    viewer.add_marker(**marker)
            frame = sim.render(mode=mode, width=width, height=height, cam_config=camera)
            if video:
                frames.append(frame)

        draw()  # creates the viewer; markers start with the next frame
        if video:
            frames.clear()
            draw()
        info = None
        for step in range(horizon):
            start = time.perf_counter()
            actions, _ = policy.predict(obs, deterministic=True)
            obs, _, _, terminated, truncated, info = env.step_arrays(actions)
            errors.append(float(info["err"][0]))
            if terminated[0] or truncated[0]:
                break  # the world has already reset; do not draw the next episode
            if not video:
                draw()
                time.sleep(max(env.dt - (time.perf_counter() - start), 0))
            elif step % stride == 0:
                draw()
        if video:
            import imageio.v2 as imageio

            imageio.mimsave(video, frames, duration=1000 / fps, loop=0)
        return {
            "task": task.name,
            "seed": seed,
            "duration_seconds": len(errors) * env.dt,
            "crashed": bool(info["crashed"][0]),
            "landed": bool(info["landed"][0]),
            "waypoints": int(info["waypoints"][0]) if task.reference == "waypoints" else None,
            "mean_position_error_m": float(np.mean(errors)),
            "video": str(video) if video else None,
        }
    finally:
        env.close()
