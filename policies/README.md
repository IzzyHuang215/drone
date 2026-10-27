# Exported flight policies

NumPy-only policies for `hover.suite.deploy.PolicyRunner`, trained in simulation for a
brushed Crazyflie 2.x (`cf2x_L250`). Not yet flown: follow [docs/deployment.md](../docs/deployment.md).

| File | Source run | Use for | Hardest-test success |
| --- | --- | --- | --- |
| `generalist.npz` | `runs/suite/generalist-robust/all-s0` | every task except landing | 86% mean over the nine tasks |
| `landing.npz` | `runs/suite/specialist-robust/landing-s0` | landing | 92% (3-seed mean) |

"Hardest test" is the held-out Crazyflie variant in the second physics model with realistic
noise, latency and parameter errors; see [docs/suite.md](../docs/suite.md). Each file carries
its observation layout, action limits, timing and drone constants in its `meta` record.
