"""Run with python -m hover {smoke,train,evaluate}."""

import argparse
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from hover.env import HoverEnv


def evaluate(model, env, episodes):
    """Compare deterministic policies on a repeatable set of held-out resets."""
    errors, returns, lengths, crashes = [], [], [], []
    for seed in range(10000, 10000 + episodes):
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
        "first_seed": 10000,
        "mean_position_error_m": float(np.mean(errors)),
        "mean_episode_return": float(np.mean(returns)),
        "mean_duration_seconds": float(np.mean(lengths)),
        "crash_or_boundary_exit_rate": float(np.mean(crashes)),
    }


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke", "train", "evaluate"])
    parser.add_argument("--steps", type=positive_int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-seconds", type=float, default=5.0)
    parser.add_argument("--eval-episodes", type=positive_int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.command == "evaluate" and args.checkpoint is None:
        parser.error("evaluate requires --checkpoint")
    torch.set_num_threads(1)
    env = HoverEnv(episode_seconds=args.episode_seconds)
    try:
        if args.command == "smoke":
            obs, _ = env.reset(seed=args.seed)
            for _ in range(100):
                obs, reward, terminated, truncated, info = env.step(np.zeros(4, np.float32))
                if terminated or truncated:
                    break
            print(json.dumps({"observation_shape": list(obs.shape), "reward": reward, **info}))
            return
        if args.command == "evaluate":
            model = PPO.load(args.checkpoint, device="cpu")
            print(json.dumps(evaluate(model, env, args.eval_episodes), indent=2))
            return
        output = args.output or Path("runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output.mkdir(parents=True, exist_ok=False)
        settings = dict(
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            policy_kwargs=dict(net_arch=[64, 64], log_std_init=-1.0),
        )
        model = PPO("MlpPolicy", Monitor(env), seed=args.seed, device="cpu", **settings)
        config = {
            **vars(args),
            "output": str(output),
            "checkpoint": None,
            "ppo": settings,
            "policy_device": "cpu",
            "versions": {
                name: importlib.metadata.version(name)
                for name in ["crazyflow", "stable-baselines3", "gymnasium", "jax", "torch"]
            },
        }
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        before = evaluate(model, env, args.eval_episodes)
        model.set_logger(configure(str(output), ["stdout", "csv"]))
        model.learn(total_timesteps=args.steps)
        model.save(output / "policy.zip")
        # Reload the serialized policy: the artifact itself must work.
        restored = PPO.load(output / "policy.zip", device="cpu")
        after = evaluate(restored, env, args.eval_episodes)
        result = {
            "actual_training_steps": model.num_timesteps,
            "untrained": before,
            "trained": after,
        }
        (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        print(f"Saved checkpoint and metrics to {output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
