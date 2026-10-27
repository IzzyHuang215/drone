"""Run with python -m hover {smoke,train,evaluate,baselines,benchmark,devices,render}."""

import argparse
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import CSVOutputFormat, HumanOutputFormat, Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor

from hover.baselines import PositionPD, ZeroPolicy
from hover.env import SIM_DEVICES, HoverEnv, require_jax_device

EVAL_FIRST_SEED = 10000
PACKAGES = ["crazyflow", "stable-baselines3", "gymnasium", "jax", "jaxlib", "torch"]


def evaluate(model, env, episodes, first_seed=EVAL_FIRST_SEED):
    """Compare deterministic policies on a repeatable set of held-out resets."""
    errors, returns, lengths, crashes = [], [], [], []
    for seed in range(first_seed, first_seed + episodes):
        obs, _ = env.reset(seed=seed)
        episode_errors, total = [], 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_errors.append(info["position_error_m"])
            total += reward
            if terminated or truncated:
                break
        errors.append(float(np.mean(episode_errors)))
        returns.append(total)
        lengths.append(len(episode_errors) * env.dt)
        crashes.append(terminated)
    return {
        "episodes": episodes,
        "first_seed": first_seed,
        "mean_position_error_m": float(np.mean(errors)),
        "mean_episode_return": float(np.mean(returns)),
        "mean_duration_seconds": float(np.mean(lengths)),
        "crash_or_boundary_exit_rate": float(np.mean(crashes)),
    }


def require_torch_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch CUDA was requested but torch.cuda.is_available() is False "
            f"(torch {torch.__version__}, built for CUDA {torch.version.cuda}). "
            "Run inside a GPU allocation."
        )
    return torch.device(device)


def device_report(sim_device, policy_device):
    jax_device = require_jax_device(sim_device)
    torch_device = require_torch_device(policy_device)
    report = {
        "sim_device": sim_device,
        "jax_device": str(jax_device),
        "jax_device_kind": jax_device.device_kind,
        "policy_device": policy_device,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_build": torch.version.cuda,
    }
    if torch_device.type == "cuda":
        report["torch_device_name"] = torch.cuda.get_device_name(torch_device)
    return report


def run_environment():
    """Versions, host and scheduler details worth keeping with every run."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                check=True,
                cwd=Path(__file__).parent,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    names = [
        "SLURM_JOB_ID",
        "SLURM_JOB_PARTITION",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "JAX_PLATFORMS",
        "CUDA_VISIBLE_DEVICES",
    ]
    return {
        "host": platform.node(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "git_commit": commit,
        "git_dirty": dirty,
        "env": {name: os.environ[name] for name in names if name in os.environ},
        "versions": {name: importlib.metadata.version(name) for name in PACKAGES},
    }


def make_training_env(args):
    if args.n_worlds == 1:
        return Monitor(HoverEnv(args.episode_seconds, device=args.sim_device))
    from hover.vec_env import HoverVecEnv

    return VecMonitor(HoverVecEnv(args.n_worlds, args.episode_seconds, device=args.sim_device))


class Throughput(BaseCallback):
    """Wall-clock training speed, excluding the first rollout and its JAX compilation."""

    def __init__(self):
        super().__init__()
        self.start = self.warm = None
        self.warm_steps = self.start_steps = 0

    def _on_training_start(self):
        self.start, self.start_steps = time.perf_counter(), self.num_timesteps

    def _on_rollout_end(self):
        if self.warm is None:
            self.warm, self.warm_steps = time.perf_counter(), self.num_timesteps

    def _on_step(self):
        return True

    def summary(self):
        end = time.perf_counter()
        steps = self.num_timesteps - self.warm_steps
        return {
            "wall_seconds": end - self.start,
            "steps": self.num_timesteps - self.start_steps,
            "steady_steps_per_second": steps / (end - self.warm) if steps > 0 else None,
            "first_rollout_seconds": self.warm - self.start,
        }


def latest_checkpoint(output):
    found = [
        (int(m.group(1)), path)
        for path in (output / "checkpoints").glob("hover_*_steps.zip")
        if (m := re.fullmatch(r"hover_(\d+)_steps\.zip", path.name))
    ]
    return max(found)[1] if found else None


def train(args):
    resume = args.resume is not None
    if resume:
        output = args.resume
        config = json.loads((output / "config.json").read_text())
        if (output / "evaluation.json").exists():
            print(f"{output} already finished; see evaluation.json")
            return
        for key in [
            "seed",
            "episode_seconds",
            "eval_episodes",
            "eval_first_seed",
            "n_worlds",
            "sim_device",
            "policy_device",
            "checkpoint_every",
        ]:
            setattr(args, key, config[key])
        args.steps = args.steps or config["steps"]
        settings = config["ppo"]
    else:
        args.steps = args.steps or 100000
        output = args.output or Path("runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output.mkdir(parents=True, exist_ok=False)
        settings = dict(
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            policy_kwargs=dict(net_arch=[64, 64], log_std_init=args.log_std_init),
        )
        config = {
            **{k: v for k, v in vars(args).items() if k not in ("command", "resume")},
            "output": str(output),
            "checkpoint": None,
            "ppo": settings,
            "devices": device_report(args.sim_device, args.policy_device),
            "environment": run_environment(),
            "resumes": [],
        }
    require_torch_device(args.policy_device)
    env = make_training_env(args)
    eval_env = HoverEnv(args.episode_seconds, device=args.sim_device)
    try:
        checkpoint = latest_checkpoint(output) if resume else None
        if checkpoint:
            model = PPO.load(checkpoint, env=env, device=args.policy_device)
            # Fresh reset states, not a replay of the first segment's; not bit-identical.
            model.get_env().seed(args.seed + model.num_timesteps)
            config["resumes"].append(
                {
                    "checkpoint": str(checkpoint),
                    "from_steps": model.num_timesteps,
                    "steps": args.steps,
                    "devices": device_report(args.sim_device, args.policy_device),
                    "environment": run_environment(),
                }
            )
            log_name = f"progress.resume-{model.num_timesteps}.csv"
        else:
            # No checkpoint yet (also after a crash before the first one): start over.
            model = PPO("MlpPolicy", env, seed=args.seed, device=args.policy_device, **settings)
            log_name = "progress.csv"
        (output / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
        untrained_path = output / "untrained_evaluation.json"
        if not untrained_path.exists():
            before = evaluate(model, eval_env, args.eval_episodes, args.eval_first_seed)
            untrained_path.write_text(json.dumps(before, indent=2) + "\n")
        before = json.loads(untrained_path.read_text())
        # SB3 truncates its CSV on open, so each resumed segment logs to its own file.
        model.set_logger(
            Logger(
                str(output),
                [HumanOutputFormat(sys.stdout), CSVOutputFormat(str(output / log_name))],
            )
        )
        callbacks = [Throughput()]
        if args.checkpoint_every:
            callbacks.append(
                CheckpointCallback(
                    max(args.checkpoint_every // args.n_worlds, 1),
                    str(output / "checkpoints"),
                    name_prefix="hover",
                )
            )
        remaining = args.steps - model.num_timesteps
        if remaining > 0:
            model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=False)
        model.save(output / "policy.zip")
        # Reload the serialized policy: the artifact itself must work.
        restored = PPO.load(output / "policy.zip", device=args.policy_device)
        after = evaluate(restored, eval_env, args.eval_episodes, args.eval_first_seed)
        result = {
            "actual_training_steps": model.num_timesteps,
            "untrained": before,
            "trained": after,
            "throughput": callbacks[0].summary() if remaining > 0 else None,
        }
        (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        print(f"Saved checkpoint and metrics to {output}")
    finally:
        env.close()
        eval_env.close()


def benchmark(args):
    """Environment throughput with fixed zero actions, excluding the first (compiling) step."""
    from hover.vec_env import HoverVecEnv

    require_jax_device(args.sim_device)
    start = time.perf_counter()
    if args.n_worlds == 1:
        env = HoverEnv(args.episode_seconds, device=args.sim_device)
        env.reset(seed=args.seed)
        step = lambda: env.step(np.zeros(4, np.float32))  # noqa: E731
    else:
        env = HoverVecEnv(args.n_worlds, args.episode_seconds, device=args.sim_device)
        env.seed(args.seed)
        env.reset()
        actions = np.zeros((args.n_worlds, 4), np.float32)
        step = lambda: env.step(actions)  # noqa: E731
    try:
        step()
        compile_seconds = time.perf_counter() - start
        calls = max(args.steps // args.n_worlds, 1)
        start = time.perf_counter()
        for _ in range(calls):
            if args.n_worlds == 1 and env.steps >= env.horizon:
                env.reset(seed=args.seed)
            step()
        seconds = time.perf_counter() - start
    finally:
        env.close()
    print(
        json.dumps(
            {
                "sim_device": args.sim_device,
                "n_worlds": args.n_worlds,
                "world_steps": calls * args.n_worlds,
                "setup_and_first_step_seconds": compile_seconds,
                "seconds": seconds,
                "world_steps_per_second": calls * args.n_worlds / seconds,
                "jax_device": str(require_jax_device(args.sim_device)),
            }
        )
    )


def devices(args):
    """Check both frameworks independently, then roll out the requested simulator."""
    import jax

    report = {
        "jax_default_backend": jax.default_backend(),
        "jax_devices": str(jax.devices()),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_build": torch.version.cuda,
        **run_environment(),
    }
    if torch.cuda.is_available():
        report["torch_devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        report["torch_cuda_matmul_ok"] = bool(
            torch.isfinite(
                torch.ones(256, 256, device="cuda") @ torch.ones(256, 256, device="cuda")
            ).all()
        )
    report.update(device_report(args.sim_device, args.policy_device))
    from hover.vec_env import HoverVecEnv

    env = HoverVecEnv(max(args.n_worlds, 2), args.episode_seconds, device=args.sim_device)
    try:
        env.seed(args.seed)
        obs = env.reset()
        for _ in range(100):
            obs, _, _, _ = env.step(np.zeros((env.num_envs, 4), np.float32))
        data_device = {str(d) for d in env.sim.data.states.pos.devices()}
        report["rollout"] = {
            "worlds": env.num_envs,
            "policy_steps": 100,
            "finite": bool(np.isfinite(obs).all()),
            "state_devices": sorted(data_device),
            "mean_position_error_m": float(np.linalg.norm(obs[:, :3], axis=1).mean()),
        }
    finally:
        env.close()
    print(json.dumps(report, indent=2, default=str))
    if not report["rollout"]["finite"]:
        raise SystemExit("simulator rollout produced non-finite states")


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["smoke", "train", "evaluate", "baselines", "benchmark", "devices", "render"],
    )
    parser.add_argument(
        "--steps",
        type=positive_int,
        help="total training (default 100000) or benchmark world steps",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-seconds", type=float, default=5.0)
    parser.add_argument("--eval-episodes", type=positive_int, default=5)
    parser.add_argument("--eval-first-seed", type=int, default=EVAL_FIRST_SEED)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="PPO policy to evaluate or render")
    parser.add_argument(
        "--baseline",
        choices=["pd", "zero"],
        help="render a baseline controller instead of --checkpoint",
    )
    parser.add_argument("--video", type=Path, help="render: save a GIF instead of opening a window")
    parser.add_argument("--resume", type=Path, help="continue training in this run directory")
    parser.add_argument(
        "--checkpoint-every",
        type=nonnegative_int,
        default=100000,
        help="training steps between checkpoints (0 disables)",
    )
    parser.add_argument("--sim-device", choices=SIM_DEVICES, default="cpu")
    parser.add_argument("--policy-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--n-worlds",
        type=positive_int,
        default=1,
        help="parallel simulator worlds; 1 uses the plain Gymnasium env",
    )
    parser.add_argument("--torch-threads", type=positive_int, default=1)
    parser.add_argument("--n-steps", type=positive_int, default=256, help="PPO steps per world")
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--n-epochs", type=positive_int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--log-std-init", type=float, default=-1.0)
    args = parser.parse_args()
    if args.command == "evaluate" and args.checkpoint is None:
        parser.error("evaluate requires --checkpoint")
    if args.resume is not None and args.output is not None:
        parser.error("use either --output or --resume")
    torch.set_num_threads(args.torch_threads)
    if args.command == "train":
        return train(args)
    if args.command == "benchmark":
        args.steps = args.steps or 10000
        return benchmark(args)
    if args.command == "devices":
        return devices(args)
    env = HoverEnv(episode_seconds=args.episode_seconds, device=args.sim_device)
    try:
        if args.command == "smoke":
            obs, _ = env.reset(seed=args.seed)
            for _ in range(100):
                obs, reward, terminated, truncated, info = env.step(np.zeros(4, np.float32))
                if terminated or truncated:
                    break
            print(json.dumps({"observation_shape": list(obs.shape), "reward": reward, **info}))
            return
        if args.command == "render":
            from hover.render import render_episode

            if (args.checkpoint is None) == (args.baseline is None):
                parser.error("render needs exactly one of --checkpoint or --baseline")
            if args.checkpoint:
                policy = PPO.load(args.checkpoint, device=args.policy_device)
            else:
                policy = PositionPD() if args.baseline == "pd" else ZeroPolicy()
            result = render_episode(policy, env, args.eval_first_seed, video=args.video)
            print(json.dumps(result, indent=2))
            return
        if args.command == "baselines":
            results = {
                name: evaluate(policy, env, args.eval_episodes, args.eval_first_seed)
                for name, policy in [("zero_action", ZeroPolicy()), ("position_pd", PositionPD())]
            }
            print(json.dumps(results, indent=2))
            return
        require_torch_device(args.policy_device)
        model = PPO.load(args.checkpoint, device=args.policy_device)
        print(json.dumps(evaluate(model, env, args.eval_episodes, args.eval_first_seed), indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
