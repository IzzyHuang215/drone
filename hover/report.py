"""Summarize an experiment directory: python -m hover.report runs/<experiment>."""

import json
import sys
from pathlib import Path

import numpy as np


def row(name, metrics, extra=""):
    return (
        f"| {name} | {metrics['mean_position_error_m']:.4f} | "
        f"{metrics['mean_duration_seconds']:.3f} | "
        f"{100 * metrics['crash_or_boundary_exit_rate']:.0f}% | {extra} |"
    )


def main(root):
    root = Path(root)
    baselines = json.loads((root / "final-baselines.json").read_text())
    first = next(iter(baselines.values()))
    lines = [
        f"Final evaluation: {first['episodes']} episodes from seed {first['first_seed']}.",
        "",
        "| Policy | Mean position error (m) | Mean duration (s) | Crash / exit rate | Notes |",
        "| --- | ---: | ---: | ---: | --- |",
        row("Zero action", baselines["zero_action"]),
        row("Position PD", baselines["position_pd"]),
    ]
    errors, details = [], []
    for run in sorted(root.glob("seed-*/")):
        final = json.loads((run / "final-evaluation.json").read_text())
        training = json.loads((run / "evaluation.json").read_text())
        config = json.loads((run / "config.json").read_text())
        throughput = training["throughput"] or {}
        sps = throughput.get("steady_steps_per_second")
        errors.append(final["mean_position_error_m"])
        lines.append(
            row(
                f"PPO {run.name}",
                final,
                f"{training['actual_training_steps']:,} steps"
                + (f", {sps:,.0f} steps/s" if sps else ""),
            )
        )
        devices = config["devices"]
        details.append(
            f"- `{run / 'policy.zip'}`: sim {devices['jax_device']}, policy "
            f"{devices.get('torch_device_name', devices['policy_device'])}, "
            f"{config['n_worlds']} worlds, last segment {throughput.get('wall_seconds', 0):.0f} s "
            f"wall, {len(config['resumes'])} resume(s), commit "
            f"{config['environment']['git_commit']}"
        )
    if errors:
        lines.append(
            f"\nPPO mean position error across seeds: {np.mean(errors):.4f} ± "
            f"{np.std(errors):.4f} m (n={len(errors)})."
        )
    lines += ["", *details]
    if details:
        config = json.loads((sorted(root.glob("seed-*/"))[0] / "config.json").read_text())
        lines.append(f"- versions: {config['environment']['versions']}")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
