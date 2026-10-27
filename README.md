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
