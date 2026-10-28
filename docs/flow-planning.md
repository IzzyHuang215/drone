# Flying flow-planner plans on the Crazyflie

The ese6180-generative-flows planning study samples planar paths from a flow model, steers the
sampler toward a task (pass a via point, avoid a new obstacle) with terminal refinement or with
early / constant / late guidance, and checks the plans with a double-integrator tracker. This
experiment flies the same plans on the simulated Crazyflie (Crazyflow, 1 m altitude) with four
trackers and asks what the double integrator misses.

## Conclusion

1. **The plans transfer when the tracker uses the plan's acceleration.** With feedforward
   (PID + plan feedforward) or a learned policy that sees a 0.5 s lookahead, drone success matches
   the double integrator to within 1–3 points (about 80% vs 82% over all plans, 96–98% vs 97% over
   guided plans). That holds even with wind, latency, thrust and gain errors and on a held-out
   airframe and physics model. Plain feedback PID loses 15–25 points.
2. **Success lost on the drone follows the dynamics a plan demands, not the sampler's
   sensitivity.** Across (task, method, flow) cells, the drop is strongly correlated with bending
   energy and peak acceleration (Spearman ρ ≈ 0.85–0.91 for PID). For the good trackers it is
   most correlated with the share of plans beyond the drone's tilt limit (ρ ≈ 0.75 under
   disturbances, 0.40–0.77 nominal). The planning
   study's sampler gain is not a significant predictor for any tracker (|ρ| ≤ 0.35, p ≥ 0.14).
3. **Guided sampling beats terminal refinement on a real vehicle.** Refinement pushes plans
   through sharp corrections. 9% of its plans exceed the 20° tilt limit (3.6 m/s² sideways), and
   its plans pass close to obstacles. It loses about 10 points on every tracker (89.5% double
   integrator vs 77–80% drone under realistic disturbances), mostly from collisions on via_rare
   and from crashes on via_wide. Early / constant / late guidance gives smooth plans that fly at
   96–99%.
4. **Making the planner drone-aware closes the gap for guided sampling, but not for
   refinement.** Adding an acceleration-limit penalty (2.86 m/s², 80% of the tilt limit) to the
   guidance brings the guided laws to 99.4–99.8% on the drone for every controller except plain
   PID, which still gains 8–12 points.
   - Under terminal refinement the penalty fights the task: on via_wide the plans no longer reach
     the via point. Double-integrator success falls from 89.5% to 84.0% while drone success rises
     only 1–5 points.
   - Feasibility has to be part of the sampling process, not a correction applied afterwards.
5. **The learned-vs-ideal flow difference stays small on the drone.** The share of paired plans
   whose outcome differs is 1.2–1.9% with the good trackers, the same as the double integrator
   (1.5%). Only feedback PID inflates it (3–4%), because it amplifies small plan differences near
   its failure boundary.
6. **Learned trackers bring no tracking advantage over classical feedforward here, and training
   on the plan family is unnecessary.**
   - The suite's generalist, which never saw these paths, the path specialist trained on random
     prior paths, and a tuned PID + feedforward are within a point of one another.
   - The learned policies' value is robustness without hand-derived feedforward, not better
     tracking.
   - For a real flight, PID + feedforward on drone-aware guided plans is the simplest stack that
     works.

## Setup

- **Plans.** We regenerate them with the study's own code and test noise, reading its repository
  without writing to it.
  - Scale: 8 trained flow models (seeds) × 4 tasks (via_common, via_rare, via_wide, blocked).
  - Methods: unguided, refinement, early, constant, late, each on the learned and the ideal flow,
    plus the trajectory optimiser.
  - Each cell has 512 plans: 330 cells in all, about 169k plans.
  - Each plan has 4 sine modes from (−2, 0) to (2, 0), flown over T = 6 s.
  - Success means no collision with the inflated discs, finishing within 0.25 m of the goal, and,
    for via tasks, passing within 0.25 m of the via point. These are the study's rules, applied to
    the flown trajectory.
- **Conditions.**
  - Nominal: cf2x_L250, so_rpy model, no disturbances.
  - Realistic: the suite's "real" robustness profile (±15% thrust, trim, wind and gusts, drag,
    0–30 ms latency, 0–60 ms attitude lag, ±15% gains, sensor noise).
  - Held-out: the realistic profile on cf2x_P250 with first-principles physics.
- **Controllers.**
  - Suite tracking PID.
  - PID + analytic plan acceleration as feedforward, with gains tuned on random prior paths: kp 30,
    kd 11, ki 9, 3.5 cm RMS.
  - Suite robust generalist (3 seeds), zero-shot.
  - Suite level-10 `path` specialist (3 seeds), trained on random sine-series paths under the
    robust mixed-physics setup.
- **Feasibility intervention.** The task penalty plus `AccelerationLimit` is used at each law's
  calibrated level.
  - The penalty is weighted by min(1, 25 / level). The via tasks (levels ≤ 23) use it at full
    weight, so their plans match the unweighted export to within 5e-16.
  - The clearance task's levels (240–570) would make the term too stiff for the sampler's
    integrator: the rollouts diverged to NaN. There its strength is capped at the via tasks'
    level. That first export is kept in `runs/flow/_superseded/`.
  - We ran it with PID, PID + feedforward, generalist-s0 and path-s0.

## Key numbers

Over all plans ([full report](flow-planning-report.md) §1):

| Tracker | nominal | realistic | held-out |
| --- | ---: | ---: | ---: |
| Double integrator (study) | 82.2% | 78.6% (strongest force) | — |
| PID | 67.4% | 60.0% | 57.1% |
| PID + plan feedforward | 81.2% | 80.2% | 79.6% |
| Learned generalist (zero-shot) | 81.6% | 80.5% | 79.4% |
| Learned path specialist | 81.3% | 80.1% | 79.7% |

Guided laws only (early / constant / late; seed-0 policies), before → after the feasibility
penalty:

| Tracker | nominal | realistic | held-out |
| --- | ---: | ---: | ---: |
| PID | 80.8 → 92.4% | 72.4 → 81.2% | 68.7 → 75.3% |
| PID + plan feedforward | 97.0 → 99.7% | 96.7 → 99.6% | 96.4 → 99.4% |
| Learned generalist | 98.0 → 99.8% | 97.6 → 99.6% | 97.4 → 99.5% |
| Learned path specialist | 97.9 → 99.8% | 97.4 → 99.6% | 97.2 → 99.4% |

Per plan, success under realistic disturbances by the peak acceleration a plan demands (§4):

| Peak acceleration | < 1 | 1–2 | 2–3.6 | > 3.6 m/s² |
| --- | ---: | ---: | ---: | ---: |
| Double integrator | 78% | 97% | 71% | 72% |
| PID + feedforward | 78% | 97% | 62% | 21% |
| Learned generalist | 78% | 98% | 64% | 16% |

The low success in the < 1 m/s² bin is the unguided samples, which mostly ignore the task.
Beyond the tilt limit the drone fails where the double integrator still succeeds; this is the
check the study's tracker cannot make.

## Reproduce

All jobs run on PARCC genoa (CPU). Run them from the drone repository root, with the flows
checkout at `../ese6180-generative-flows`.

```bash
# 1. export plans + paired double-integrator outcomes (flows env, read-only)
SEEDS=0 sbatch experiments/flow_plans/export.sbatch            # one job per seed 0..7
SCRIPT=export_feasible.py OUT=plans-feasible SEEDS=0 sbatch experiments/flow_plans/export.sbatch
# 2. PID + feedforward gains (writes runs/flow/pid_ff_gains.json)
sbatch experiments/flow_plans/tune.sbatch
# 3. path specialists: suite task level 10 (slurm/suite-campaign.sh)
# 4. fly every plan set with every controller and condition
PLAN_SEED=0 sbatch experiments/flow_plans/execute.sbatch       # one job per seed 0..7
PLAN_SEED=0 PLANS=runs/flow/plans-feasible OUT=runs/flow/exec-feasible \
  CONTROLLERS="pid pid-ff generalist-s0 path-s0" sbatch experiments/flow_plans/execute.sbatch
# 5. tables
.venv/bin/python experiments/flow_plans/analyze.py              # -> runs/flow/analysis/
```

## Caveats

- **Simulation only.** The realistic and held-out conditions are the suite's randomization ranges,
  not measured hardware. The first real flights should use guided plans with the feasibility
  penalty and PID + feedforward, or the exported generalist.
- **The acceleration limit ignores time scaling.** The traversal time is fixed at 6 s. A slower
  traversal would make most refinement plans feasible, at the cost of the study's timing.
- **Sampler gain is limited.** It is available only for learned guided cells (n = 19 cells after
  dropping cells without a calibrated level), so its non-correlation is a weak negative rather
  than proof of independence.
