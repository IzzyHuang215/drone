## 1. Success rate over all plans

Double integrator (planning study, tight tracker): nominal 82.2%, strongest random force 78.6%; 330 cells of 512 plans.

| Controller on the Crazyflie | nominal | realistic | held-out |
| --- | ---: | ---: | ---: |
| PID | 67.4% | 60.0% | 57.1% |
| PID + plan feedforward | 81.2% | 80.2% | 79.6% |
| Learned generalist (never saw these paths) | 81.6% | 80.5% | 79.4% |
| Learned path specialist | 81.3% | 80.1% | 79.7% |


## 1b. By sampling method (mean over tasks, flows and planner seeds)


**realistic**

| Method | Double integrator | Peak accel. (m/s²) | Beyond drone limit | PID | PID + plan feedforward | Learned generalist (never saw these paths) | Learned path specialist |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unguided | 26.6% | 0.63 | 0.0% | 23.3% | 26.7% | 26.6% | 26.6% |
| refinement | 89.5% | 1.63 | 9.2% | 49.7% | 80.1% | 79.6% | 78.3% |
| early | 98.0% | 0.89 | 0.0% | 76.4% | 98.1% | 98.7% | 98.6% |
| constant | 96.8% | 0.94 | 2.1% | 72.2% | 96.4% | 97.2% | 96.9% |
| late | 96.6% | 0.97 | 3.0% | 69.7% | 96.0% | 96.5% | 96.2% |
| optimiser | 100.0% | 0.67 | 0.0% | 88.1% | 100.0% | 100.0% | 100.0% |

**held-out**

| Method | Double integrator | Peak accel. (m/s²) | Beyond drone limit | PID | PID + plan feedforward | Learned generalist (never saw these paths) | Learned path specialist |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unguided | 26.6% | 0.63 | 0.0% | 21.2% | 26.8% | 26.5% | 26.7% |
| refinement | 89.5% | 1.63 | 9.2% | 48.2% | 77.8% | 76.9% | 76.7% |
| early | 98.0% | 0.89 | 0.0% | 72.5% | 98.1% | 98.2% | 98.7% |
| constant | 96.8% | 0.94 | 2.1% | 68.3% | 96.1% | 96.4% | 96.8% |
| late | 96.6% | 0.97 | 3.0% | 66.0% | 95.5% | 95.4% | 95.8% |
| optimiser | 100.0% | 0.67 | 0.0% | 85.0% | 99.3% | 98.5% | 99.9% |


## 2. What predicts the success lost on the drone?

Spearman correlation across (task, method, flow) cells between the drop (double-integrator nominal success minus drone success) and each predictor. Sampler gain is measured for learned flows only, as in the planning study.

| Controller | Condition | Cells | Bending | Peak accel. | Beyond limit | Planned clearance | Sampler gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PID | nominal | 42 | +0.91 (p=7.2e-17) | +0.91 (p=1.4e-16) | +0.63 (p=6.3e-06) | -0.22 (p=0.16) | -0.03 (p=0.89, n=19) |
| PID | realistic | 42 | +0.86 (p=1.7e-13) | +0.85 (p=8.7e-13) | +0.74 (p=2.8e-08) | -0.29 (p=0.063) | +0.09 (p=0.71, n=19) |
| PID | held-out | 42 | +0.84 (p=4.2e-12) | +0.84 (p=3.5e-12) | +0.72 (p=6.2e-08) | -0.27 (p=0.081) | +0.06 (p=0.79, n=19) |
| PID + plan feedforward | nominal | 42 | +0.05 (p=0.76) | +0.05 (p=0.74) | +0.40 (p=0.008) | +0.18 (p=0.24) | +0.28 (p=0.25, n=19) |
| PID + plan feedforward | realistic | 42 | +0.49 (p=0.00095) | +0.48 (p=0.0012) | +0.76 (p=4.8e-09) | -0.49 (p=0.001) | +0.34 (p=0.15, n=19) |
| PID + plan feedforward | held-out | 42 | +0.49 (p=0.001) | +0.51 (p=0.00064) | +0.74 (p=2.9e-08) | -0.47 (p=0.0016) | +0.35 (p=0.14, n=19) |
| Learned generalist (never saw these paths) | nominal | 42 | +0.15 (p=0.33) | +0.08 (p=0.63) | +0.52 (p=0.00037) | +0.05 (p=0.74) | +0.30 (p=0.22, n=19) |
| Learned generalist (never saw these paths) | realistic | 42 | +0.41 (p=0.0069) | +0.34 (p=0.03) | +0.76 (p=5.8e-09) | -0.18 (p=0.25) | +0.30 (p=0.21, n=19) |
| Learned generalist (never saw these paths) | held-out | 42 | +0.48 (p=0.0013) | +0.51 (p=0.00049) | +0.74 (p=2.5e-08) | -0.28 (p=0.07) | +0.17 (p=0.5, n=19) |
| Learned path specialist | nominal | 42 | +0.40 (p=0.0079) | +0.31 (p=0.044) | +0.77 (p=3.1e-09) | -0.17 (p=0.3) | +0.34 (p=0.15, n=19) |
| Learned path specialist | realistic | 42 | +0.42 (p=0.0055) | +0.35 (p=0.023) | +0.75 (p=1.4e-08) | -0.20 (p=0.2) | +0.30 (p=0.21, n=19) |
| Learned path specialist | held-out | 42 | +0.49 (p=0.0011) | +0.49 (p=0.00088) | +0.74 (p=1.7e-08) | -0.31 (p=0.046) | +0.32 (p=0.18, n=19) |


## 3. Learned vs ideal flow: share of paired plans whose outcome differs

Double integrator, nominal: 1.5%

| Controller | nominal | realistic | held-out |
| --- | ---: | ---: | ---: |
| PID | 4.0% | 3.2% | 3.0% |
| PID + plan feedforward | 1.4% | 1.4% | 1.5% |
| Learned generalist (never saw these paths) | 1.2% | 1.2% | 1.9% |
| Learned path specialist | 1.2% | 1.2% | 1.3% |


## 4. Success by the acceleration a plan demands (per plan, all cells)


**realistic** (plans per bin: 108470, 43296, 12611, 4583)

| Peak acceleration (m/s²) | Double integrator | PID | PID + plan feedforward | Learned generalist (never saw these paths) | Learned path specialist |
| --- | ---: | ---: | ---: | ---: | ---: |
| < 1 | 78.1% | 64.6% | 78.0% | 78.1% | 78.0% |
| 1–2 | 97.3% | 69.6% | 97.3% | 98.0% | 97.9% |
| 2–3.6 | 70.5% | 8.7% | 62.4% | 64.3% | 60.7% |
| > 3.6 (beyond tilt limit) | 71.6% | 1.5% | 20.7% | 15.4% | 12.9% |

**held-out** (plans per bin: 108470, 43296, 12611, 4583)

| Peak acceleration (m/s²) | Double integrator | PID | PID + plan feedforward | Learned generalist (never saw these paths) | Learned path specialist |
| --- | ---: | ---: | ---: | ---: | ---: |
| < 1 | 78.1% | 60.8% | 77.7% | 77.3% | 78.0% |
| 1–2 | 97.3% | 67.1% | 97.0% | 97.2% | 97.8% |
| 2–3.6 | 70.5% | 10.3% | 59.0% | 59.7% | 57.0% |
| > 3.6 (beyond tilt limit) | 71.6% | 1.5% | 16.3% | 12.8% | 10.6% |


## 5. Adding a feasibility penalty to the planner's guidance

Each guided law (same schedule, calibrated level and test noise) with the task penalty plus an acceleration limit of 2.86 m/s² over the 6 s traversal. Means over tasks, flows and planner seeds of paired cells.

| Controller | Condition | Method | Cells | Beyond tilt limit | Double integrator success | Drone success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Learned generalist (never saw these paths) | held-out | constant | 58 | 2.1% → 0.0% | 96.8% → 99.8% | 97.3% → 99.4% |
| Learned generalist (never saw these paths) | held-out | early | 48 | 0.0% → 0.0% | 98.0% → 100.0% | 99.2% → 100.0% |
| Learned generalist (never saw these paths) | held-out | late | 64 | 3.0% → 0.0% | 96.6% → 99.1% | 96.2% → 99.2% |
| Learned generalist (never saw these paths) | held-out | refinement | 64 | 9.2% → 6.0% | 89.5% → 84.0% | 78.8% → 80.7% |
| Learned generalist (never saw these paths) | nominal | constant | 58 | 2.1% → 0.0% | 96.8% → 99.8% | 97.9% → 99.8% |
| Learned generalist (never saw these paths) | nominal | early | 48 | 0.0% → 0.0% | 98.0% → 100.0% | 99.4% → 100.0% |
| Learned generalist (never saw these paths) | nominal | late | 64 | 3.0% → 0.0% | 96.6% → 99.1% | 97.0% → 99.6% |
| Learned generalist (never saw these paths) | nominal | refinement | 64 | 9.2% → 6.0% | 89.5% → 84.0% | 85.8% → 88.5% |
| Learned generalist (never saw these paths) | realistic | constant | 58 | 2.1% → 0.0% | 96.8% → 99.8% | 97.5% → 99.6% |
| Learned generalist (never saw these paths) | realistic | early | 48 | 0.0% → 0.0% | 98.0% → 100.0% | 99.2% → 100.0% |
| Learned generalist (never saw these paths) | realistic | late | 64 | 3.0% → 0.0% | 96.6% → 99.1% | 96.4% → 99.4% |
| Learned generalist (never saw these paths) | realistic | refinement | 64 | 9.2% → 6.0% | 89.5% → 84.0% | 79.8% → 81.9% |
| Learned path specialist | held-out | constant | 58 | 2.1% → 0.0% | 96.8% → 99.8% | 97.0% → 99.4% |
| Learned path specialist | held-out | early | 48 | 0.0% → 0.0% | 98.0% → 100.0% | 98.8% → 100.0% |
| Learned path specialist | held-out | late | 64 | 3.0% → 0.0% | 96.6% → 99.1% | 96.0% → 99.0% |
| Learned path specialist | held-out | refinement | 64 | 9.2% → 6.0% | 89.5% → 84.0% | 78.1% → 80.2% |
| Learned path specialist | nominal | constant | 58 | 2.1% → 0.0% | 96.8% → 99.8% | 97.9% → 99.8% |
| Learned path specialist | nominal | early | 48 | 0.0% → 0.0% | 98.0% → 100.0% | 99.0% → 100.0% |
| Learned path specialist | nominal | late | 64 | 3.0% → 0.0% | 96.6% → 99.1% | 97.1% → 99.6% |
| Learned path specialist | nominal | refinement | 64 | 9.2% → 6.0% | 89.5% → 84.0% | 84.3% → 89.5% |
| Learned path specialist | realistic | constant | 58 | 2.1% → 0.0% | 96.8% → 99.8% | 97.3% → 99.6% |
| Learned path specialist | realistic | early | 48 | 0.0% → 0.0% | 98.0% → 100.0% | 98.8% → 100.0% |
| Learned path specialist | realistic | late | 64 | 3.0% → 0.0% | 96.6% → 99.1% | 96.4% → 99.3% |
| Learned path specialist | realistic | refinement | 64 | 9.2% → 6.0% | 89.5% → 84.0% | 78.8% → 80.9% |
| PID | held-out | constant | 58 | 2.1% → 0.0% | 96.8% → 99.8% | 68.3% → 75.8% |
| PID | held-out | early | 48 | 0.0% → 0.0% | 98.0% → 100.0% | 72.5% → 77.9% |
| PID | held-out | late | 64 | 3.0% → 0.0% | 96.6% → 99.1% | 66.0% → 73.0% |
| PID | held-out | refinement | 64 | 9.2% → 6.0% | 89.5% → 84.0% | 48.2% → 48.2% |
| PID | nominal | constant | 58 | 2.1% → 0.0% | 96.8% → 99.8% | 80.6% → 93.1% |
| PID | nominal | early | 48 | 0.0% → 0.0% | 98.0% → 100.0% | 84.7% → 94.5% |
| PID | nominal | late | 64 | 3.0% → 0.0% | 96.6% → 99.1% | 78.1% → 90.1% |
| PID | nominal | refinement | 64 | 9.2% → 6.0% | 89.5% → 84.0% | 57.1% → 57.1% |
| PID | realistic | constant | 58 | 2.1% → 0.0% | 96.8% → 99.8% | 72.2% → 81.7% |
| PID | realistic | early | 48 | 0.0% → 0.0% | 98.0% → 100.0% | 76.4% → 84.6% |
| PID | realistic | late | 64 | 3.0% → 0.0% | 96.6% → 99.1% | 69.7% → 78.2% |
| PID | realistic | refinement | 64 | 9.2% → 6.0% | 89.5% → 84.0% | 49.7% → 49.7% |
| PID + plan feedforward | held-out | constant | 58 | 2.1% → 0.0% | 96.8% → 99.8% | 96.1% → 99.5% |
| PID + plan feedforward | held-out | early | 48 | 0.0% → 0.0% | 98.0% → 100.0% | 98.1% → 100.0% |
| PID + plan feedforward | held-out | late | 64 | 3.0% → 0.0% | 96.6% → 99.1% | 95.5% → 98.9% |
| PID + plan feedforward | held-out | refinement | 64 | 9.2% → 6.0% | 89.5% → 84.0% | 77.8% → 78.9% |
| PID + plan feedforward | nominal | constant | 58 | 2.1% → 0.0% | 96.8% → 99.8% | 96.8% → 99.8% |
| PID + plan feedforward | nominal | early | 48 | 0.0% → 0.0% | 98.0% → 100.0% | 98.1% → 100.0% |
| PID + plan feedforward | nominal | late | 64 | 3.0% → 0.0% | 96.6% → 99.1% | 96.5% → 99.3% |
| PID + plan feedforward | nominal | refinement | 64 | 9.2% → 6.0% | 89.5% → 84.0% | 84.5% → 84.4% |
| PID + plan feedforward | realistic | constant | 58 | 2.1% → 0.0% | 96.8% → 99.8% | 96.4% → 99.8% |
| PID + plan feedforward | realistic | early | 48 | 0.0% → 0.0% | 98.0% → 100.0% | 98.1% → 100.0% |
| PID + plan feedforward | realistic | late | 64 | 3.0% → 0.0% | 96.6% → 99.1% | 96.0% → 99.2% |
| PID + plan feedforward | realistic | refinement | 64 | 9.2% → 6.0% | 89.5% → 84.0% | 80.1% → 81.2% |
