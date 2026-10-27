# Hover training setup

Approved scope: a CPU Crazyflow setup managed by uv, PPO hover training, saved checkpoints
and metrics, evaluation, a short verified training run, and a push to main.

Implementation choices: Python 3.12, released Crazyflow 0.4.0, Stable-Baselines3 PPO,
a single Gymnasium environment using fitted attitude-response dynamics, a fixed target at
(0, 0, 1) metres, randomized nearby starts, 100 Hz policy and 500 Hz simulation.
This is a simulation learning baseline, not hardware deployment software.

- Add a lockfile and ignored local environments / training outputs.
- Test the Gymnasium contract, repeatable reset, episode horizon, and action validation.
- Implement a bounded attitude / thrust task with finite observations and crash termination.
- Add train / evaluate / smoke commands with saved configuration, CSV metrics, and checkpoints.
- Run tests and short PPO training; reload and evaluate its checkpoint on held-out seeds.
- Document exact commands and observed limitations; commit and push main.
