# drone
Custom autonomous quadcopter build — notes, design, and progress

- [Project goals](Notes/goals.md)
- [Learning-based flight-control project discussion](Notes/learning-based-flight-control.md) — hardware candidates, simulation roadmap, and research connections imported from the ChatGPT conversation.

## Hover training

Start training in simulation before buying hardware: [setup and commands](docs/hover-training.md).

```sh
uv sync --locked
uv run python -m hover smoke
uv run python -m hover train --steps 4096 --output runs/hover-smoke
uv run python -m hover evaluate --checkpoint runs/hover-smoke/policy.zip
```

The short validation runs exercised the pipeline but did **not** converge to stable hover.
See [measured results and limitations](docs/hover-validation.md).

For cluster setup, give the next agent [the NVIDIA HPC handoff](docs/hpc-handoff.md).
The resulting PARCC setup (GPU extra, batched worlds, checkpoint/resume, SLURM jobs) is in
[docs/hpc-setup.md](docs/hpc-setup.md); with enough data PPO now hovers stably in simulation.

## Flight task suite

Nine tasks from hover to a waypoint course, sharing one interface and built for transfer to a real
Crazyflie: [docs/suite.md](docs/suite.md) (tasks, results), [docs/deployment.md](docs/deployment.md)
(bring-up on hardware) and exported policies in [policies/](policies/).

```sh
uv run python -m hover.suite list
uv run python -m hover.suite render --baseline pid --tasks figure8
```

