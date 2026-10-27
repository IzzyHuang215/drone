# Flight task suite

Nine simulated Crazyflie tasks, from holding a point to flying a waypoint course, built for an
eventual transfer to a real drone (see [deployment.md](deployment.md)). Every task uses the
same observation and action, so a policy trained on one task can start the next, and one
generalist policy can fly all of them.

```sh
uv run python -m hover.suite list
uv run python -m hover.suite train --tasks circle --steps 10000000 --output runs/suite/my-circle
uv run python -m hover.suite evaluate --checkpoint runs/suite/my-circle/policy.zip --tasks circle
uv run python -m hover.suite render --checkpoint runs/suite/my-circle/policy.zip --tasks circle
uv run python -m hover.suite render --baseline pid --tasks landing --video landing.gif
```

On PARCC, `bash slurm/suite-campaign.sh` reproduces the whole portfolio (3 seeds of every
specialist and generalist, nominal and realistic) and is safe to rerun;
`sbatch slurm/suite-baselines.sbatch` evaluates the baselines, and
`python -m hover.suite.report runs/suite` prints the tables below.

## The curriculum

| Level | Task | Episode | What it asks | Success |
| ---: | --- | ---: | --- | --- |
| 1 | hover | 5 s | Hold a point from ±0.2 m, upright | mean error < 10 cm in the last second, no crash |
| 2 | hover-wide | 5 s | ±0.5 m, ±0.5 m/s, 20° tilt, any heading | same |
| 3 | reach | 6 s | Fly to a target up to ±1 m (±0.5 m vertical) and stop | same |
| 4 | recover | 5 s | Thrown: 60° tilt, 6 rad/s spin, 2 m/s | same |
| 5 | disturbance | 8 s | Gusts (10% of weight), wind (≤10%), random 0.75 m/s pushes | mean error < 25 cm after 1 s, no crash |
| 6 | circle | 10 s | 0.3–1 m radius, 5–8 s loops, either direction | mean error < 15 cm after 1 s, no crash |
| 7 | figure8 | 12 s | 0.3–1 m half-width, 6–10 s loops | same |
| 8 | landing | 8 s | Descend 0.5–1.5 m, slow to 0.15 m/s for the last 25 cm | touchdown < 0.4 m/s, < 20° tilt, < 20 cm from the pad |
| 9 | waypoints | 12 s | Four random waypoints within ±1 m, in order | every waypoint captured (15 cm), no crash |

Trajectories speed up over their first second so they can start from hover. Leaving the
reference by more than 2–3 m, or dropping below 5 cm altitude (except when landing), ends the
episode as a crash.

## Interface

**Action** (4 values in [−1, 1]): roll, pitch and yaw targets (±0.35 rad) and collective
thrust (0 = the model's hover thrust), sent to the drone's own attitude controller. This is the
Crazyflie's attitude-setpoint interface, so the onboard stabilizer stays in the loop.

**Observation** (29 values, all available on a real drone): position and velocity relative to
the reference, the attitude as a rotation matrix (no quaternion sign ambiguity when the drone
tumbles), body rates, the previous action, where the reference will be in 0.5 s (or the next
waypoint), altitude, and an anti-windup integral of the position error. The integral is
needed: without it neither a policy nor a PD controller can cancel a constant bias such as a
tired battery or steady wind (in a development check, a PD controller without it succeeded in
0 of 16 realistic hover episodes).

**Reward**: a survival bonus, a bonus for being within ~10 cm of the reference, penalties for
distance (tanh), velocity error, action size, action changes (smoothness matters on hardware)
and tilt, −10 for a crash. Landing has no survival bonus and pays +10 for a soft touchdown, so
landing beats hovering above the pad. While waypoints remain, the reward is the distance closed
toward the current waypoint (+5 per capture); rewarding closeness instead made the first
policy loiter outside the capture radius (16% success), because capturing switches to a distant
target.

## Robustness profiles and the second model

`--robustness real` resamples per episode, scaled for a Crazyflie 2.x with external
positioning and setpoints sent over the radio:

| Effect | Range |
| --- | --- |
| Thrust per command (mass, battery, motors) | ±15% |
| Roll/pitch trim | ±2° |
| Constant wind / gust std / linear drag | ≤5% / 3% of weight / ≤0.02 N per m/s |
| Command latency | 0–30 ms |
| Attitude response: first-order lag, gain | 0–60 ms, ±15% |
| Sensor noise: position, velocity, attitude, gyro | 1 cm, 5 cm/s, ~1°, 0.05 rad/s |

The training model is Crazyflow's `so_rpy`, an attitude-response model identified from
Crazyflie flight data. The second model, `first_principles`, is a rigid body with rotor dynamics
and a firmware-style attitude controller. Its attitude loop is about 30 ms slower and its yaw
response roughly 8x slower; both models report body-frame rates.

Policies are built in stages, each warm-starting from the previous one:

| Stage | Training | Purpose |
| --- | --- | --- |
| `specialist` | one task, `so_rpy`, no randomization | can the task be learned at all |
| `specialist-real` | + realistic profile | robustness to noise, latency, parameter errors |
| `specialist-robust` | + half the worlds on each physics model (`--dynamics mix`) | robustness to model error |
| `generalist`, `generalist-robust` | all nine tasks in one policy | one network for a real flight |

The realistic profile alone did not carry policies across physics models (figure-eight and
landing fell to 0–1% on the second model), and an ablation of action smoothing and wider timing
ranges was inconclusive. Training on both models did close most of the gap.

**The held-out test** is `cf2x_P250`, a separately identified Crazyflie 2.x configuration
whose attitude loop is about 35% weaker. No policy trained on it and no randomization range was
fitted to it. Five evaluation conditions, 100 episodes each from seed 20000:

| Column | Robustness | Physics | Drone |
| --- | --- | --- | --- |
| Nominal | none | so_rpy | cf2x_L250 |
| Realistic | real | so_rpy | cf2x_L250 |
| Second model | real | first_principles | cf2x_L250 (trained on by the robust stages) |
| Held-out drone | real | so_rpy | cf2x_P250 |
| Held-out drone, 2nd model | real | first_principles | cf2x_P250 (the hardest test) |

## Results

Recorded 2026-10-27 on PARCC CPU nodes (`slurm/suite-campaign.sh`, commit range on branch
`hpc-gpu-setup`). Visual version with flight videos: the published portfolio page.

**Summary on the hardest test** (held-out drone, second physics model, realistic profile):
robust specialists average 87% success across the nine tasks and the robust generalist 86%,
against 39% for the PID baseline. Learned policies win by wide margins on seven tasks, tie on
landing (specialist 92% vs PID 88%; the generalist lands less reliably at 64%) and do not beat
PID on the waypoint course (PID 84%; best learned is the nominal specialist at 83%, then the
generalist at 79% and the robust specialist at 66%). Continuing the robust waypoint specialists
to 30M steps made them worse (41%, seeds 23–75%).

**Recommended policies for a first real flight:** `generalist-robust/all-s0` for everything
except landing, `specialist-robust/landing-s0` for landing, and the PID baseline for waypoint
courses. Export them with `python -m hover.suite export`.

How to read the tables: success rate over 100 held-out episodes per cell; with three training
seeds, the mean is followed by the min–max across seeds. Rows are training stages (see above).
The PID baseline sees the same observation, noise and latency as the policies.

#### 1. hover: Hold a point from small upright offsets (the original task).

| Policy | Nominal | Realistic | Second model | Held-out drone | Held-out drone, 2nd model |
| --- | ---: | ---: | ---: | ---: | ---: |
| PID baseline | 100% | 43% | 50% | 35% | 49% |
| generalist/all | 100% (100–100) | 99% (97–100) | 41% (38–46) | 100% (100–100) | 71% (69–75) |
| generalist-robust/all | 100% (100–100) | 100% (100–100) | 84% (68–99) | 97% (91–100) | 94% (88–99) |
| specialist/hover | 100% (100–100) | 100% (100–100) | 38% (33–41) | 100% (100–100) | 52% (50–54) |
| specialist-real/hover | 100% (100–100) | 100% (100–100) | 44% (39–48) | 100% (100–100) | 65% (57–70) |
| specialist-robust/hover | 100% (100–100) | 100% (100–100) | 76% (70–86) | 100% (100–100) | 85% (81–88) |

#### 2. hover-wide: Hold a point from larger offsets, velocities, tilts and any heading.

| Policy | Nominal | Realistic | Second model | Held-out drone | Held-out drone, 2nd model |
| --- | ---: | ---: | ---: | ---: | ---: |
| PID baseline | 100% | 33% | 21% | 27% | 28% |
| generalist/all | 100% (100–100) | 100% (100–100) | 29% (19–36) | 100% (100–100) | 60% (59–61) |
| generalist-robust/all | 100% (100–100) | 100% (100–100) | 65% (41–83) | 97% (92–100) | 89% (85–97) |
| specialist/hover-wide | 100% (100–100) | 100% (100–100) | 58% (48–64) | 100% (100–100) | 81% (77–85) |
| specialist-real/hover-wide | 100% (100–100) | 100% (100–100) | 55% (31–79) | 100% (100–100) | 83% (65–95) |
| specialist-robust/hover-wide | 100% (100–100) | 100% (100–100) | 74% (69–77) | 100% (99–100) | 96% (95–98) |

#### 3. reach: Fly to a random target up to 1 m away in each direction and stop there.

| Policy | Nominal | Realistic | Second model | Held-out drone | Held-out drone, 2nd model |
| --- | ---: | ---: | ---: | ---: | ---: |
| PID baseline | 100% | 50% | 57% | 42% | 58% |
| generalist/all | 100% (100–100) | 100% (100–100) | 51% (46–58) | 100% (100–100) | 73% (70–75) |
| generalist-robust/all | 100% (100–100) | 100% (100–100) | 91% (78–100) | 99% (97–100) | 96% (94–100) |
| specialist/reach | 100% (100–100) | 100% (100–100) | 34% (29–40) | 100% (100–100) | 50% (45–56) |
| specialist-real/reach | 100% (100–100) | 100% (100–100) | 56% (49–61) | 100% (100–100) | 72% (65–76) |
| specialist-robust/reach | 100% (100–100) | 100% (100–100) | 83% (75–89) | 100% (100–100) | 89% (86–93) |

#### 4. recover: Catch a thrown drone: up to 60 deg tilt, 6 rad/s spin and 2 m/s velocity.

| Policy | Nominal | Realistic | Second model | Held-out drone | Held-out drone, 2nd model |
| --- | ---: | ---: | ---: | ---: | ---: |
| PID baseline | 100% | 36% | 15% | 27% | 20% |
| generalist/all | 100% (100–100) | 100% (100–100) | 21% (19–26) | 100% (100–100) | 50% (46–57) |
| generalist-robust/all | 100% (100–100) | 100% (100–100) | 49% (38–62) | 98% (96–100) | 69% (67–73) |
| specialist/recover | 100% (100–100) | 100% (100–100) | 50% (47–52) | 100% (100–100) | 69% (66–70) |
| specialist-real/recover | 100% (100–100) | 100% (99–100) | 47% (40–53) | 100% (99–100) | 72% (67–77) |
| specialist-robust/recover | 100% (100–100) | 98% (93–100) | 51% (48–56) | 95% (86–99) | 70% (67–75) |

#### 5. disturbance: Hold position in gusty wind with random 0.75 m/s pushes.

| Policy | Nominal | Realistic | Second model | Held-out drone | Held-out drone, 2nd model |
| --- | ---: | ---: | ---: | ---: | ---: |
| PID baseline | 3% | 3% | 3% | 3% | 3% |
| generalist/all | 97% (97–97) | 89% (88–91) | 40% (32–48) | 84% (82–86) | 68% (61–77) |
| generalist-robust/all | 97% (97–97) | 89% (89–90) | 81% (70–90) | 81% (77–85) | 88% (84–93) |
| specialist/disturbance | 97% (97–98) | 90% (90–91) | 22% (8–37) | 85% (83–87) | 45% (25–69) |
| specialist-real/disturbance | 97% (97–98) | 91% (90–92) | 29% (17–44) | 86% (85–87) | 56% (41–66) |
| specialist-robust/disturbance | 98% (98–98) | 91% (90–92) | 74% (64–84) | 88% (87–90) | 89% (88–91) |

#### 6. circle: Track a horizontal circle (0.3-1 m radius, 5-8 s per loop, either direction).

| Policy | Nominal | Realistic | Second model | Held-out drone | Held-out drone, 2nd model |
| --- | ---: | ---: | ---: | ---: | ---: |
| PID baseline | 32% | 10% | 7% | 6% | 8% |
| generalist/all | 100% (100–100) | 100% (100–100) | 40% (30–57) | 100% (100–100) | 73% (57–95) |
| generalist-robust/all | 100% (100–100) | 100% (100–100) | 94% (84–100) | 100% (100–100) | 98% (93–100) |
| specialist/circle | 100% (100–100) | 100% (100–100) | 52% (19–69) | 96% (90–100) | 77% (43–96) |
| specialist-real/circle | 100% (100–100) | 100% (100–100) | 80% (68–95) | 100% (99–100) | 97% (93–100) |
| specialist-robust/circle | 100% (100–100) | 100% (100–100) | 96% (90–100) | 100% (99–100) | 99% (97–100) |

#### 7. figure8: Track a figure-eight (0.3-1 m half-width, 6-10 s per loop).

| Policy | Nominal | Realistic | Second model | Held-out drone | Held-out drone, 2nd model |
| --- | ---: | ---: | ---: | ---: | ---: |
| PID baseline | 31% | 7% | 9% | 3% | 9% |
| generalist/all | 100% (100–100) | 100% (100–100) | 45% (36–60) | 100% (100–100) | 71% (59–94) |
| generalist-robust/all | 100% (100–100) | 100% (100–100) | 93% (80–100) | 100% (100–100) | 98% (94–100) |
| specialist/figure8 | 100% (100–100) | 100% (100–100) | 17% (1–33) | 100% (99–100) | 48% (10–80) |
| specialist-real/figure8 | 100% (100–100) | 100% (100–100) | 35% (6–62) | 100% (100–100) | 80% (59–93) |
| specialist-robust/figure8 | 100% (100–100) | 100% (100–100) | 81% (57–98) | 100% (100–100) | 99% (97–100) |

#### 8. landing: Descend 0.5-1.5 m onto a pad, slowing near the ground, and touch down gently.

| Policy | Nominal | Realistic | Second model | Held-out drone | Held-out drone, 2nd model |
| --- | ---: | ---: | ---: | ---: | ---: |
| PID baseline | 100% | 85% | 87% | 80% | 88% |
| generalist/all | 100% (100–100) | 85% (65–95) | 22% (10–31) | 82% (57–95) | 42% (27–54) |
| generalist-robust/all | 100% (100–100) | 95% (91–100) | 58% (34–93) | 90% (86–98) | 64% (42–95) |
| specialist/landing | 98% (96–100) | 63% (44–77) | 8% (2–15) | 17% (0–33) | 13% (2–21) |
| specialist-real/landing | 99% (97–100) | 94% (90–99) | 38% (8–66) | 77% (67–92) | 51% (26–66) |
| specialist-robust/landing | 100% (100–100) | 98% (98–99) | 90% (86–94) | 85% (67–95) | 92% (89–94) |

#### 9. waypoints: Fly through four random waypoints in order within 12 s.

| Policy | Nominal | Realistic | Second model | Held-out drone | Held-out drone, 2nd model |
| --- | ---: | ---: | ---: | ---: | ---: |
| PID baseline | 100% | 85% | 85% | 82% | 84% |
| generalist/all | 85% (76–96) | 84% (73–93) | 61% (57–66) | 84% (74–94) | 77% (71–84) |
| generalist-robust/all | 82% (72–91) | 80% (70–88) | 78% (68–87) | 82% (73–88) | 79% (68–86) |
| specialist/waypoints | 95% (90–100) | 86% (83–90) | 79% (77–81) | 84% (83–85) | 83% (81–84) |
| specialist-real/waypoints | 85% (80–89) | 69% (62–78) | 58% (41–68) | 68% (62–75) | 64% (52–74) |
| specialist-robust/waypoints | 78% (73–83) | 66% (63–69) | 64% (62–68) | 61% (58–65) | 66% (61–69) |
| specialist-robust-long/waypoints | 61% (31–99) | 43% (24–72) | 42% (23–77) | 30% (16–45) | 41% (23–75) |
