# Odin1 Tier 0 characterization

Autonomous data collection on the ARUW sentry plus the analysis that fits and validates the
Tier 0 error model of the *Odin1 Tier 0 Error Characterization Protocol* (static field ε_s,
FOGM η with Student-t innovations, latency τ_d, INIT/TRACK/LOST events).

Ground truth is the mocap `mocap → base_link` transform. **Wheel encoders are never used**:
the robot is driven on mocap feedback, and its command gains are identified from mocap.

## What runs where

```
mocap ──/tf mocap→base_link──► campaign (robot computer, ROS 2, 50 Hz)
                                  │ UDP (vx, vy, wz body frame)
                                  ▼
               project-otto  tools/chassis_velocity_bridge.py  (Jetson, replaces otto main)
                                  │ UART msg 0x14, 100 Hz
                                  ▼
               aruw-mcb  ExternalVelocityCommand + TurretMajorChassisHoldCommand
                         (remote: left UP / right MID)
```

* **aruw-mcb**, branch `odin-characterization`: new UART message `CV_MESSAGE_TYPE_CHASSIS_VELOCITY`
  (20). On the sentry, *left UP / right MID* now runs `ExternalVelocityCommand`, which drives
  the chassis from that message and stops if no command arrives for 150 ms. The same switch
  state runs `TurretMajorChassisHoldCommand`, which holds turret major (and Odin on it) at a
  fixed chassis yaw. Flip out of that switch state to stop the robot. Do not take this branch
  to competition: it replaces auto-nav + auto-aim on that switch state.
* **project-otto**, branch `odin-characterization`: `ChassisVelocityMessage` (0x14) and
  `tools/chassis_velocity_bridge.py`.
* **this repo**: the campaign (`odin_char/robot`) and the analysis (`odin_char/*.py`).

Odin facts used here come from `manifoldsdk/odin_ros_driver` v0.14.4: frames `odom→imu` and
`odom→map` (parent `odom`, child `map`, published only in relocalization mode), topics
`/odin1/{imu,odometry,odometry_highfreq,cloud_raw}`, the `/odin1/save_map` service, and
the `relocalization success!` log line.

## The scripts you run

| | where | what |
|---|---|---|
| `scripts/setup.sh` | robot computer (ROS 2 sourced) and/or laptop | venv + self-tests |
| `scripts/start_bridge.sh` | Jetson, own terminal | UDP → UART bridge (stop otto main first) |
| `scripts/run_campaign.sh day1` | robot computer | Phases 0, 1, 2 (session 1), 3, 4, 5 |
| `scripts/run_campaign.sh day2` / `day3` | robot computer, later days | Phase 2 sessions 2 and 3 |
| `scripts/analyze.sh` | anywhere | Phase 1 fits, Phases 2–6 analysis → `results/report.md` |

## Before the first run

1. **MCB.** Flash `aruw-mcb@odin-characterization` (`scons run robot=SENTRY_ACHLYS`).
2. **otto.** Check out `project-otto@odin-characterization` on the Jetson and run `poetry install`.
3. **This repo.** Run `scripts/setup.sh` on the robot computer, after sourcing ROS 2 and the
   workspace containing `odin_ros_driver`.
4. **Config.** Edit `config/campaign.yaml`. Everything marked `VERIFY`:
   * `frames.mocap_world` / `mocap_body`: your mocap tf names. Mocap stamps are taken to be on
     the **mocap server clock** (`frames.mocap_clock: server`); each bag is mapped onto the host
     clock from message arrival times.
   * `room.fence_min/max`, `room.home`: the safe box in the mocap world frame, and where the robot
     parks. Start every run with the robot at least 1 m inside the fence.
   * `phases.phase2`: the figure-8 (centre, size, rotation, speed) through the regions that matter.
   * `phases.phase3.locations`: 3–5 hold spots with different geometry, inside the figure-8's
     area if possible.
   * `driver.base_config`: your deployed `control_command.yaml`. It is copied per run with the
     map mode, `use_host_ros_time: 2` and `devstatuslog: 1`; the installed file is not modified.
   * `hand_eye_prior`: base_link → Odin IMU from CAD (`[tx ty tz rx ry rz]`, rotation vector).
     Planar driving cannot observe the vertical lever arm, so this pins it. The report says which
     parameters came from the prior.
   * `metadata`: optional `firmware`, `driver_version`, `people_present` for the metadata sheet.
5. **Mocap.** Ideally put the mocap rigid body on Odin's bracket and set `mocap_body` to it. If it
   stays on the chassis, the turret-major hold keeps Odin rigid relative to it.

## Day 1

Terminal A (Jetson): `scripts/start_bridge.sh`

Terminal B (robot computer, ROS 2 sourced): `scripts/run_campaign.sh day1`

The script prompts (with a beep) only when it needs a person:

1. Set the remote to **left UP / right MID**. The script then nudges the robot back and forth to
   identify the command gains from mocap.
2. Phase 0: turn the mocap strobes **off**, then back on. Then power Odin **off**, then back on.
3. Phase 2: stand at `operator_spot` and stay there (a moving person is a dynamic object).
4. Phase 5 outages: cover Odin's window on **COVER NOW** and uncover on **UNCOVER NOW**. Set
   `phases.phase5.shutter_cmd_cover/uncover` to automate this with a shutter.

Everything else is autonomous: 20 min warm-up, calibration motion, SLAM mapping with automatic
map save, session 1 (fresh relocalization start, 10 nominal laps, 2 at 0.5×, 2 at 1.5×), 10 min
holds, latency sinusoids, 24 INIT cold starts and outages. `--no-prompt` skips the steps that need
a person so the rest can run unattended. Steps are resumable: after an abort, rerun the same
command and finished steps are skipped (`--redo` repeats them).

## Days 2 and 3

`scripts/run_campaign.sh day2`, and later `day3`: a gain check plus one Phase 2 session each.
Keep the scene and firmware/driver versions the same.

## Analysis

`scripts/analyze.sh campaigns/odin1` writes `campaigns/odin1/results/`:

* `report.md`: every pass/fail check from the protocol, Tier 1 screens and the Tier 0 verdict
* `model.yaml`: τ_d; per-axis P, τ_c, ν for moving and stationary; events (λ_fix, logistic
  p(success), p_wrong, λ_lost or its rule-of-three bound, λ_rec)
* `static_field_grid.npz`: ε_s on a 0.1 m grid in the map frame
* `calibration.json`: time offset, hand-eye X, frozen T_WM
* `results.json`, `figures/`

## Safety

The robot moves only while the remote is in left UP / right MID. The campaign stops it when:

* mocap is stale (by arrival time, > 50 ms),
* the robot or its stopping point leaves the fence,
* the tracking error exceeds 0.5 m, or
* you press Ctrl-C.

Every trajectory is checked against the fence before it starts. The bridge zeroes the command if
the campaign goes quiet for 100 ms. The MCB zeroes it if no command arrives for 150 ms, if otto is
offline or if a motor is offline, and caps speed at 2.5 m/s and 6 rad/s.

## Where this deviates from the protocol, and why

* **Hand-eye.** Ground robots mostly yaw, so the vertical lever arm is unobservable from motion.
  A CAD prior pins it, and the report lists the prior-dominated parameters. Odometry lags mocap
  by τ_d, so that lag is estimated and removed before solving AX = XB.
* **Latency regression.** Done within position bins. On a figure-8, yaw rate correlates with
  position, so a static field would otherwise leak into τ_d. The speed sweeps provide the
  variation.
* **Static field.** Kept per axis. An axis with negative held-out R² only adds noise. Set
  `analysis.static_field_per_axis: false` for the literal reading.
* **Stationary holds.** The static error at each hold spot is the hold's mean, because the GP
  field is only trustworthy where the figure-8 went. Both values are reported.
* **Error translation.** Uses p̂ − p in body-aligned axes rather than the translation part of
  Log(T̂ T⁻¹), which carries a lever arm about the map origin.
* **Phase 6 verdict.** Uses exact Monte Carlo rank tests (a global envelope for the ACF, QQ and RPE
  curves), family-wise over the six axes, with parameter uncertainty in the simulations. The
  protocol's per-axis 5–95% band results are reported alongside. 200 realizations, since 50
  cannot reach the per-axis level.
* **INIT offsets.** Distance to the *nearest* point of the mapping path, since a lawnmower map has
  no single trajectory. The achieved offset is logged and used.

## Tests

`.venv/bin/python -m pytest -q`:

* `test_pipeline.py`: a synthetic campaign written as real rosbag2 files with known values
  (mocap server clock offset/drift and jitter, hand-eye, T_WM, τ_d, static field, FOGM/t, events).
  The full analysis must recover them.
* `test_validate.py`: Phase 6 passes the true model and rejects wrong σ, τ_c or τ_d.
* `test_motion.py`: trajectories, controller tracking, fence, command-gain identification.
* `test_campaign_dryrun.py`: the whole `day1` + `day2` orchestration against a simulated sentry,
  mocap and Odin driver (~5 min, real time).
