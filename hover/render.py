"""Watch one hover episode in the MuJoCo viewer, or save it as a GIF.

The live viewer needs a display (e.g. a laptop); ``--video`` renders offscreen and also
works on headless cluster nodes with ``MUJOCO_GL=egl``.
"""

import time

import numpy as np

from hover.env import TARGET

# Free camera looking at the target from the side and slightly above.
CAMERA = {"distance": 0.6, "azimuth": 135.0, "elevation": -15.0, "lookat": TARGET.astype(float)}


def render_episode(policy, env, seed, video=None, fps=25, width=640, height=480):
    """Roll out ``policy`` deterministically; show it live or write every k-th frame."""
    obs, _ = env.reset(seed=seed)
    stride = max(int(round(1 / (env.dt * fps))), 1)
    mode = "rgb_array" if video else "human"
    frames, errors = [], []

    def draw():
        frame = env.sim.render(mode=mode, width=width, height=height, cam_config=CAMERA)
        if video:
            frames.append(frame)

    draw()
    for step in range(env.horizon):
        start = time.perf_counter()
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        errors.append(info["position_error_m"])
        if video:
            if step % stride == 0:
                draw()
        else:
            draw()
            # Play back in real time (simulated 0.01 s per policy step).
            time.sleep(max(env.dt - (time.perf_counter() - start), 0))
        if terminated or truncated:
            break
    if video:
        import imageio.v2 as imageio

        imageio.mimsave(video, frames, duration=1000 / fps, loop=0)
    return {
        "seed": seed,
        "duration_seconds": len(errors) * env.dt,
        "crashed": bool(terminated),
        "mean_position_error_m": float(np.mean(errors)),
        "video": str(video) if video else None,
    }
