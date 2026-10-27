"""Run with python -m hover.suite {list,train,evaluate,render,export}."""

import argparse
import json
import sys
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import CSVOutputFormat, HumanOutputFormat, Logger
from stable_baselines3.common.vec_env import VecMonitor

from hover.__main__ import (
    Throughput,
    device_report,
    latest_checkpoint,
    positive_int,
    require_torch_device,
    run_environment,
)
from hover.env import SIM_DEVICES
from hover.suite import tasks as T
from hover.suite.controllers import TrackingPID, Zero
from hover.suite.env import DRONES, DYNAMICS, drone_constants, training_env
from hover.suite.evaluate import FINAL_SEED, evaluate

TUNING_SEED = 10000


def load_policy(args, dynamics=None):
    if args.checkpoint:
        require_torch_device(args.policy_device)
        return PPO.load(args.checkpoint, device=args.policy_device)
    constants = drone_constants(args.drone, dynamics or args.dynamics)
    return TrackingPID(constants) if args.baseline == "pid" else Zero()


def env_kwargs(args):
    return dict(
        robustness=args.robustness, device=args.sim_device, drone=args.drone, dynamics=args.dynamics
    )


def train(args):
    resume = args.resume is not None
    output = args.resume if resume else args.output
    if resume:
        config = json.loads((output / "config.json").read_text())
        if (output / "evaluation.json").exists():
            print(f"{output} already finished; see evaluation.json")
            return
        saved = config["args"]
        for key, value in saved.items():
            if key not in ("resume", "output", "steps", "command"):
                setattr(args, key, value)
        args.steps = args.steps or saved["steps"]
    else:
        if output is None:
            raise SystemExit("train needs --output or --resume")
        output.mkdir(parents=True, exist_ok=False)
        config = {
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "tasks": [t.name for t in T.resolve(args.tasks)],
            "devices": device_report(args.sim_device, args.policy_device),
            "environment": run_environment(),
            "resumes": [],
        }
    tasks = T.resolve(args.tasks)
    args.steps = args.steps or 5_000_000
    settings = dict(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=0.95,
        ent_coef=args.ent_coef,
        policy_kwargs=dict(net_arch=list(args.net_arch), log_std_init=args.log_std_init),
    )
    config["ppo"] = settings
    require_torch_device(args.policy_device)
    env = VecMonitor(
        training_env(
            tasks,
            args.n_worlds,
            dynamics=args.dynamics,
            smoothness=args.smoothness,
            robustness=args.robustness,
            device=args.sim_device,
            drone=args.drone,
        )
    )
    try:
        checkpoint = latest_checkpoint(output) if resume else None
        if checkpoint:
            model = PPO.load(checkpoint, env=env, device=args.policy_device)
            model.get_env().seed(args.seed + model.num_timesteps)
            config["resumes"].append(
                {"checkpoint": str(checkpoint), "from_steps": model.num_timesteps}
            )
            log_name = f"progress.resume-{model.num_timesteps}.csv"
        else:
            model = PPO("MlpPolicy", env, seed=args.seed, device=args.policy_device, **settings)
            if args.init_from:
                # Curriculum: start from an easier task's policy (same interface and network).
                source = PPO.load(args.init_from, device=args.policy_device)
                model.policy.load_state_dict(source.policy.state_dict())
                config["init_from"] = str(args.init_from)
            log_name = "progress.csv"
        (output / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
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
        restored = PPO.load(output / "policy.zip", device=args.policy_device)
        eval_kwargs = {
            **env_kwargs(args),
            "dynamics": "so_rpy" if args.dynamics == "mix" else args.dynamics,
        }
        result = {
            "actual_training_steps": model.num_timesteps,
            "throughput": callbacks[0].summary() if remaining > 0 else None,
            "tuning_evaluation": evaluate(
                restored, tasks, args.eval_episodes, TUNING_SEED, **eval_kwargs
            ),
        }
        (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "train", "evaluate", "render", "export"])
    parser.add_argument("--tasks", nargs="+", default=["hover"], help="task names or 'all'")
    parser.add_argument("--robustness", choices=list(T.ROBUSTNESS), default="none")
    parser.add_argument("--drone", choices=DRONES, default="cf2x_L250")
    parser.add_argument(
        "--dynamics",
        choices=[*DYNAMICS, "mix"],
        default="so_rpy",
        help="'mix' (training only): half the worlds on each model",
    )
    parser.add_argument("--sim-device", choices=SIM_DEVICES, default="cpu")
    parser.add_argument("--policy-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--checkpoint", type=Path, help="PPO policy to evaluate/render/export")
    parser.add_argument("--baseline", choices=["pid", "zero"])
    parser.add_argument("--episodes", type=positive_int, default=100, help="evaluate: per task")
    parser.add_argument("--seed", type=int, help="train: 0; evaluate/render: 20000 (final)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video", type=Path, help="render: save a GIF instead of a window")
    # Training
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-from", type=Path, help="warm-start from an easier task policy")
    parser.add_argument("--steps", type=positive_int)
    parser.add_argument("--n-worlds", type=positive_int, default=64)
    parser.add_argument("--n-steps", type=positive_int, default=256)
    parser.add_argument("--batch-size", type=positive_int, default=1024)
    parser.add_argument("--n-epochs", type=positive_int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument(
        "--smoothness",
        type=float,
        default=0.05,
        help="reward penalty weight on squared action changes",
    )
    parser.add_argument("--log-std-init", type=float, default=-1.0)
    parser.add_argument("--net-arch", type=positive_int, nargs="+", default=[128, 128])
    parser.add_argument("--checkpoint-every", type=int, default=1_000_000)
    parser.add_argument("--eval-episodes", type=positive_int, default=32)
    parser.add_argument("--torch-threads", type=positive_int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    if args.command == "list":
        for task in T.TASKS.values():
            print(f"{task.level}. {task.name:12s} {task.episode_seconds:4.0f} s  {task.summary}")
        return
    if args.command == "train":
        args.seed = 0 if args.seed is None else args.seed
        return train(args)
    if args.command == "export":
        from hover.suite.deploy import export

        if not (args.checkpoint and args.output):
            parser.error("export needs --checkpoint and --output")
        export(args.checkpoint, args.output, drone=args.drone, dynamics=args.dynamics)
        print(f"wrote {args.output}")
        return
    if (args.checkpoint is None) == (args.baseline is None):
        parser.error(f"{args.command} needs exactly one of --checkpoint or --baseline")
    seed = FINAL_SEED if args.seed is None else args.seed
    if args.dynamics == "mix":
        parser.error("evaluate and render need a single --dynamics model")
    policy = load_policy(args)
    if args.command == "evaluate":
        results = evaluate(policy, T.resolve(args.tasks), args.episodes, seed, **env_kwargs(args))
        report = {
            "policy": str(args.checkpoint or args.baseline),
            "seed": seed,
            **{k: args.__dict__[k] for k in ("robustness", "drone", "dynamics", "episodes")},
            "tasks": results,
        }
        text = json.dumps(report, indent=2)
        if args.output:
            args.output.write_text(text + "\n")
        print(text)
        return
    from hover.suite.render import render

    (task,) = T.resolve(args.tasks)
    print(json.dumps(render(policy, task, seed, video=args.video, **env_kwargs(args)), indent=2))


if __name__ == "__main__":
    main()
