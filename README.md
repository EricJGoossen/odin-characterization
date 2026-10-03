# Odin1 Tier 0 characterization

Fits and validates the Tier 0 error model of the *Odin1 Tier 0 Error Characterization Protocol*
on the ARUW sentry: static field ε_s, FOGM η with Student-t innovations, latency τ_d, and
INIT/TRACK/LOST events. Ground truth is the already-running mocap. **Wheel encoders are never
used.**

## Two ways to run it (`robot.control` in `config/campaign.yaml`)

* **`human` (default):** a person drives the sentry with the remote. The script starts and stops
  Odin, records every run, tells you what to drive, and shows live mocap feedback (speed, laps,
  map coverage, distance and heading to the next spot). It waits for Enter, and starts a hold by
  itself once you're parked on target. **Nothing is sent to the robot**, so you need only this
  repo, ROS 2, Odin and the mocap. Otto stays as it is, and the MCB can run `develop`.
* **`autonomous`:** the script drives the robot on mocap feedback through otto's
  odin-characterization mode and the MCB branch (parts 1 and 2 below). It needs a working
  Jetson ↔ MCB UART link.

The data and the analysis are identical either way. Hand-driven Phase 2 laps repeat less exactly,
but the static field is binned by position, so that's fine.

### Running in human mode

```
scripts/setup.sh                      # once (ROS 2 + ~/odin_ws are sourced automatically)
scripts/run_campaign.sh find-mocap    # once: paste the printed mocap: block into the config
scripts/run_campaign.sh mocap         # check the live mocap stream (nothing moves)
scripts/run_campaign.sh odin          # check Odin starts and publishes (nothing moves)
scripts/run_campaign.sh day1          # follow the on-screen instructions
scripts/run_campaign.sh day2          # later: one more Phase 2 session (and day3)
scripts/analyze.sh                    # -> campaigns/odin1/results/report.md
```

What day 1 asks of the driver, in order:

| step | what you do | live feedback |
|---|---|---|
| preflight | mocap strobes off/on, Odin power off/on, robot still | countdown |
| calib | ~2.5 min of driving with lots of spinning both ways and swerving | elapsed, yaw rate |
| map | park centrally; then lanes ~1 m apart over the whole area plus loops around the edge | coverage %, distance |
| session | park at the figure-8 start; Odin relocalizes; 10 steady laps, 2 slow, 2 fast | laps, speed vs target |
| holds | drive to each spot (auto-detects when you're parked), then hands off for 10 min | distance and turn to target, countdown |
| latency | ~80 s quick back-and-forth, then ~80 s rotating back and forth in place | elapsed, speed / yaw rate |
| init | drive to each of 24 start poses (auto-detected); Odin restarts; for "moving" trials drive 1 m then turn 90° | distance and turn to target, time to fix |
| outage | drive slowly back and forth; a second person covers Odin on **COVER NOW / UNCOVER NOW** | elapsed |

## Autonomous mode: the three parts

The work is split into three parts, each on its own branch or repo:

```
                 (3) odin-characterization  ── this repo, robot computer with ROS 2
                     │  subscribes to mocap (configurable), starts/stops the Odin driver,
                     │  records bags, closes the loop on mocap, runs the analysis
                     │ UDP :47020  (seq, vx, vy, wz; chassis frame)
                     ▼
                 (1) project-otto @ odin-characterization  ── Jetson
                     │  `poetry run odin-characterization` (replaces otto's main loop)
                     │ UART message 0x14 @ 100 Hz
                     ▼
                 (2) aruw-mcb @ odin-characterization  ── sentry MCB
                        remote left UP / right MID: ExternalVelocityCommand
                        + TurretMajorChassisHoldCommand (holds Odin's turret still)
```

## 1. project-otto: Odin characterization mode

Branch `odin-characterization`, documented in `docs/odin_characterization.md` there.

* `project_otto/messages/_chassis_velocity_message.py`: `ChassisVelocityMessage`
  (type `0x14`: `uint32 seq, float vx, float vy, float wz`).
* `tools/chassis_velocity_bridge.py`, run as `poetry run odin-characterization`: a UDP → UART
  bridge at 100 Hz. It sends zeros if the script goes quiet for 100 ms and prints a status line
  every second. `--dummy` runs it without an MCB.
* `tests/messages/test_chassis_velocity_message.py`.

## 2. aruw-mcb: chassis velocity from the coprocessor

Branch `odin-characterization`, documented in `docs/odin_characterization.md` there. **Don't take
it to competition.**

* `vision_coprocessor.{hpp,cpp}`: RX message `CV_MESSAGE_TYPE_CHASSIS_VELOCITY = 20`.
* `control/chassis/external_velocity_command.{hpp,cpp}`: chassis velocity in SI units, a 150 ms
  watchdog, and hard caps of 2.5 m/s and 6 rad/s.
* `robot/sentry/turret/turret_major_chassis_hold_command.{hpp,cpp}`: holds turret major at a
  fixed, encoder-referenced chassis yaw.
* `robot/sentry/sentry_control.cpp`: **left UP / right MID** runs both commands.

## 3. odin-characterization (this repo): stitching, mocap, Odin, analysis

| path | what |
|---|---|
| `config/campaign.yaml` | **the one file to edit**: mocap source, room, figure-8, Odin driver |
| `odin_char/robot/campaign.py` | the campaign: bring-up steps, all protocol phases, safety, resume |
| `odin_char/robot/ros_io.py` | ROS 2: mocap subscriber (any source type), Odin tf/IMU watch, UDP sender |
| `odin_char/robot/processes.py` | Odin driver (per-run config copy, log watching, map save) and `ros2 bag record` |
| `odin_char/robot/motion.py` | trajectories, mocap-feedback controller, geofence, command-gain identification |
| `odin_char/robot/discover.py` | `find-mocap`: lists the live mocap candidates |
| `odin_char/mocap_config.py` | the `mocap:` config block (tf, PoseStamped, TransformStamped, Odometry) |
| `odin_char/*.py` | analysis: calibration, errors, latency, static field, FOGM, events, model, Phase 6 |
| `scripts/` | `setup.sh`, `start_bridge.sh`, `run_campaign.sh`, `analyze.sh` |

### Mocap configuration

The mocap topic and frame names aren't known in advance, so they're configured in one place:

```yaml
mocap:
  source: tf            # tf | pose (PoseStamped) | transform (TransformStamped) | odometry
  topic: /tf            # the topic carrying it
  parent_frame: mocap   # tf only
  child_frame: base_link  # tf only: the robot's rigid body
  clock: server         # stamps from the mocap server's clock (mapped to host time per bag)
  qos: best_effort      # also receives reliable publishers
```

`scripts/run_campaign.sh find-mocap` listens to every topic that could carry a pose. It prints
each one's rate, frames and arrival-minus-stamp offset, then a ready-to-paste `mocap:` block.

Odin's names come from `manifoldsdk/odin_ros_driver` v0.14.4 and are already set:

* tf `odom → imu`, and `odom → map` (parent `odom`, published only in relocalization mode)
* topics `/odin1/imu`, `/odin1/odometry`, `/odin1/cloud_raw`
* service `/odin1/save_map`

The campaign runs the driver itself (`ros2 run odin_ros_driver host_sdk_sample` with its own
copy of your `control_command.yaml`), so don't start Odin by hand.

## Bring-up order for autonomous mode (first time)

```
1. MCB:    flash aruw-mcb@odin-characterization   (cd aruw-mcb-project && scons run robot=SENTRY_ACHLYS)
2. otto:   on the Jetson, stop otto's main loop, then  scripts/start_bridge.sh
           (or: cd project-otto && poetry run odin-characterization)
3. this:   scripts/setup.sh                     (once; ROS 2 + odin driver workspace sourced)
           scripts/run_campaign.sh find-mocap   -> paste the mocap: block into config/campaign.yaml
           scripts/run_campaign.sh mocap        live pose/rate/clock, fence check (no motion)
           scripts/run_campaign.sh odin         starts Odin 20 s, lists its topics (no motion)
           scripts/run_campaign.sh jog          remote L-UP/R-MID; tiny pulses, prints motion per command
           scripts/run_campaign.sh check        identifies command -> velocity gains from mocap
4. collect: scripts/run_campaign.sh day1        (later days: day2, day3)
5. analyse: scripts/analyze.sh                  -> campaigns/odin1/results/report.md
```

Each bring-up step isolates one link of the chain:

| failing step | what's broken |
|---|---|
| `find-mocap` | ROS networking or `ROS_DOMAIN_ID` |
| `mocap` | the `mocap:` config |
| `odin` | the Odin driver, USB, or `driver.base_config` |
| `jog` | otto bridge (watch its status line), UART, the MCB branch, or the remote switch |
| `check` | gains, if they're implausible |

## Before day 1, edit `config/campaign.yaml`

* `mocap:` (see above).
* `room.fence_min/max`, `room.home`: a safe box in the mocap world frame. Start with the robot at
  least 1 m inside it.
* `phases.phase2`: the figure-8 through the regions that matter. `phases.phase3.locations`: 3–5
  hold spots.
* `driver.base_config`: your deployed `control_command.yaml`. It is copied per run with the map
  mode, `use_host_ros_time: 2` and `devstatuslog: 1` applied; the installed file isn't changed.
* `hand_eye_prior`: rigid body → Odin IMU from CAD (`[tx ty tz rx ry rz]`, rotation vector).
  Planar driving can't observe the vertical lever arm.

## Collection

`scripts/run_campaign.sh day1` beeps and waits only when it needs a person:

1. the remote switch,
2. mocap strobes off/on and Odin power off/on (Phase 0),
3. standing at the operator spot (Phase 2),
4. covering Odin on **COVER NOW** / **UNCOVER NOW** (Phase 5 outages, unless a shutter command is set).

Everything else is autonomous: calibration, SLAM map plus automatic save, session 1,
holds, latency, 24 cold starts and outages. The protocol's 20 min thermal warm-up is disabled
for speed (`driver.warmup_min: 0`, thermal effects ignored); set it to 20 to restore it. `--no-prompt` skips the steps that need a person.
After an abort, rerun the same command: finished steps are skipped (`--redo` repeats them).
`day2` and `day3` each add one Phase 2 session.

Outputs per campaign: `bags/`, `logs/` (driver logs and per-run config copies), `manifest.json`,
`metadata.csv`, `state.json`.

## Analysis

`scripts/analyze.sh [campaigns/odin1]` runs anywhere, no ROS needed. It writes `results/`:

* `report.md`: every protocol pass/fail check, the Tier 1 screens and the verdict
* `model.yaml`
* `static_field_grid.npz`
* `calibration.json`
* `results.json`
* `figures/`

## Safety

The robot moves only while the remote is in left UP / right MID. The campaign stops it when:

* mocap is stale (by arrival time, > 50 ms),
* the robot or its stopping point leaves the fence,
* the tracking error exceeds 0.5 m, or
* you press Ctrl-C.

Every trajectory is checked against the fence before it starts. The bridge zeroes the command
after 100 ms of silence. The MCB zeroes it after 150 ms without a command, when CV is offline, or
when a motor is offline.

## Deviations from the protocol, and why

* **Hand-eye:** a CAD prior pins the unobservable vertical lever arm, and the odometry lag is
  removed before solving AX = XB.
* **Latency:** the regression runs within position bins, because on a figure-8 yaw rate
  correlates with position.
* **Static field:** kept per axis (`analysis.static_field_per_axis`). At hold spots the static
  error is the hold mean.
* **Error translation:** uses p̂ − p in body-aligned axes, which avoids a lever arm about the map
  origin.
* **Phase 6:** exact Monte Carlo rank tests (a global envelope for the curves), family-wise over
  the six axes, with parameter uncertainty and 200 realizations. The doc's 5–95% bands are
  reported too.
* **INIT offsets:** distance to the nearest point of the mapping path; the achieved values are
  logged.

## Tests

`.venv/bin/python -m pytest -q` (~8 min):

* synthetic rosbag campaign → full analysis recovers the known parameters
* mocap source types
* Phase 6 power: accepts the true model, rejects wrong ones
* motion and controller
* full `day1` + `day2` dry run against a simulated sentry, mocap and driver
