# Flying a suite policy on a real Crazyflie

Status: **simulation-validated only.** Nothing here has flown. The repository provides a
NumPy-only policy and a step function that reproduces the training observation exactly
(tested in `tests/test_suite.py`). The radio link, unit conversions marked *verify* and
thrust calibration below must be checked on your hardware, in the order given, before any
untethered flight.

## Architecture

```
positioning system ──► Crazyflie state estimator ──(cflib log, 100 Hz)──► laptop
                                                                           │
                       PolicyRunner: reference + state ──► policy.npz ──► roll, pitch, yaw, thrust
                                                                           │
Crazyflie attitude controller ◄──(cflib commander.send_setpoint, 100 Hz)───┘
```

The policy replaces the Crazyflie's *position* controller only. The firmware keeps its
attitude and rate loops (the simulator's `so_rpy` model was identified from exactly that
loop), which is the safer first transfer path recommended in
[the project notes](../Notes/learning-based-flight-control.md).

**Positioning is required.** Every task observes position and velocity relative to a
reference. Use a system that gives the onboard estimator absolute position (Lighthouse
positioning deck + base stations, Loco, or motion capture). The Flow deck v2 alone drifts and
is not enough for the trajectory, waypoint or landing tasks.

## 1. Export the policy

```sh
uv run python -m hover.suite export --checkpoint runs/suite/<run>/policy.zip \
  --output policy.npz --drone cf2x_L250
```

`policy.npz` holds the weights and a `meta` record with the observation layout, action limits,
control period (0.01 s), lookahead (0.5 s), integral settings and the drone constants. Use the
`--drone` the policy was trained for: `cf2x_L250` (Crazyflie 2.x brushed) or `cf21B_500`
(Crazyflie 2.1 Brushless).

## 2. The control step

```python
from hover.suite.deploy import PolicyRunner

runner = PolicyRunner("policy.npz")
runner.reset()                       # before every flight
# every 10 ms:
action, sp = runner(pos, vel, quat_xyzw, gyro_rad_s, ref_pos, ref_vel, ref_ahead)
# sp = {"roll": rad, "pitch": rad, "yaw": rad (target heading), "thrust": N}
```

Inputs come from the Crazyflie log variables `stateEstimate.{x,y,z,vx,vy,vz,qx,qy,qz,qw}`
and `gyro.{x,y,z}`. **Verify:** `gyro` is logged in deg/s and must be converted to rad/s;
check the quaternion order and that velocity is in the world frame. `ref_ahead` is the
reference 0.5 s ahead (or the next waypoint). The trajectories in `hover/suite/env.py`
(`reference`) show how each task builds it.

## 3. Converting to `send_setpoint`

`cflib`'s `commander.send_setpoint(roll, pitch, yawrate, thrust)` takes degrees, a yaw
**rate** in deg/s and an integer thrust (10001–60000). From the firmware source
(`crtp_commander_rpyt.c`): roll and pitch pass straight through as angle setpoints in the
default mode, the legacy yaw rate is negated, and thrust stays locked until one setpoint with
thrust 0 has been sent.

| Policy output | `send_setpoint` argument | Notes |
| --- | --- | --- |
| roll (rad) | `degrees(roll) * ROLL_SIGN` | **verify sign** |
| pitch (rad) | `degrees(pitch) * PITCH_SIGN` | **verify sign**: in the simulator +pitch accelerates toward +x |
| yaw target (rad) | `yawrate = YAW_SIGN * K_YAW * degrees(wrap(yaw_target - yaw))` | the policy outputs a heading; turn it into a rate with K_YAW ≈ 2 /s |
| thrust (N) | `int(HOVER_PWM * thrust / hover_thrust)` | linear around hover; **calibrate HOVER_PWM**, clamp to 10001–60000 |

`hover_thrust` is `meta["constants"]["hover_thrust"]` (0.314 N for `cf2x_L250`). Measure
`HOVER_PWM` by logging `stabilizer.thrust` during a stable hover under the onboard position
controller. Battery sag lowers the real thrust per PWM; the training randomized thrust by
±15% and the integral observation absorbs the rest.

## 4. Bring-up sequence

Do not skip steps. Keep an independent kill switch (a key that calls
`commander.send_stop_setpoint()` and stops the loop) in reach at all times.

1. **Bench, propellers off.** Run the loop with the drone held level and the real estimator.
   Confirm 100 Hz timing (training tolerated up to 30 ms latency), that `runner` outputs
   ≈ zero tilt and hover thrust at the reference, and that moving the drone 10 cm in +x makes
   the policy command a tilt back toward −x. Set `ROLL_SIGN`, `PITCH_SIGN` and `YAW_SIGN`
   from what the motors and estimator report.
2. **Onboard hover first.** Take off and hover with the Crazyflie's own high-level commander.
   Log `HOVER_PWM`. Land.
3. **Handover in a hover.** From the onboard hover at 1 m, switch to the policy with the
   `hover` task reference (hold the current position). Arm a watchdog that hands control back
   (stop sending setpoints and call the high-level `land`) if the position error exceeds 0.3 m,
   tilt exceeds 30 deg, altitude leaves 0.2–1.8 m, or a setpoint is late by more than 50 ms.
4. **Climb the curriculum.** Fly the tasks in level order: hover, hover-wide (small offsets
   first), reach (short distances first), landing, circle, figure8, waypoints. Recover and
   disturbance are mainly simulation tests; only throw or push the drone with guards fitted
   and a net.
5. **Compare against the baseline.** Fly the same reference with the PID baseline
   (`hover.suite.controllers.TrackingPID`, which uses the same observation) and with the
   Crazyflie's own controller, and log both. A policy is only an improvement if it beats them on
   the same logged references.

## What the simulator does not cover

Ground effect, propeller damage, motor heating, battery dynamics beyond a fixed thrust scale,
estimator glitches and radio dropouts are not modelled. `docs/suite.md` lists what *is*
randomized; the "realistic + unseen model" column there is the best available proxy for how a
policy will degrade, and it is only a proxy.
