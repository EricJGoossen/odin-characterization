# Sensors

Everything specific to the sensor under test lives in one folder here. The campaign, the analysis and
the scripts only talk to the `Sensor` interface in [base.py](base.py). `config/campaign.yaml` picks the
sensor:

```yaml
sensor:
  type: odin      # the folder name
  driver:         # anything here overrides the sensor's defaults.yaml
    base_config: ~/odin_ws/src/odin_ros_driver/config/control_command.yaml
```

## Adding a sensor

Create `odin_char/sensors/<type>/` with:

| file | needed | what |
|---|---|---|
| `__init__.py` | yes | defines `SENSOR = YourSensor` (a `Sensor` subclass) |
| `defaults.yaml` | usually | name, frames, topics, driver command; merged under the campaign's `sensor:` block |
| `env.sh` | if the driver is in its own workspace | sourced after ROS 2 by `setup.sh` / `run_campaign.sh`; may define `sensor_env_check` |
| anything else | no | driver helpers, a hardware self-test (see `odin/selftest.py`) |

Then set `sensor.type: <type>` in `config/campaign.yaml` and run the bring-up steps:
`run_campaign.sh find-mocap`, then `mocap`, then `sensor`.

### The easy case: the sensor publishes tf

If the driver publishes `odom -> body` (smooth odometry) and `odom -> map` (registration to the prior
map, once it has a fix) on `/tf`, `TfSensor` already does all the analysis. A launch command is
enough for the driver:

```python
# odin_char/sensors/mysensor/__init__.py
from ..base import TfSensor

class MySensor(TfSensor):
    name = "MySensor"

SENSOR = MySensor
```

```yaml
# odin_char/sensors/mysensor/defaults.yaml
name: MySensor
frames: {odom: odom, body: imu_link, map: map, odom_map_is_ros_convention: false}
namespace: /mysensor              # find-mocap ignores these topics
topics:
  imu: /mysensor/imu              # gyro for the time-offset fit; arrival = "driver is up"
  odometry: /mysensor/odometry    # optional
  cloud: /mysensor/points         # optional: Phase 0 preflight, outage timing
  extra: []                       # anything else to record
warmup_min: 0
connect_timeout_s: 60
driver:
  # {config} = what Driver.prepare() returned (nothing by default), {mode} = 0 odometry / 1 mapping / 2 relocalize
  run_cmd: "ros2 launch mysensor_driver run.launch.py mode:={mode}"
  process_name: mysensor_node     # pgrep pattern: refuse to start next to a driver started by hand
  watch:                          # driver log text -> event the campaign reacts to
    relocalized: "localized in map"
```

### What else you may need to override

| situation | override |
|---|---|
| the driver needs a per-run config file (mode, map path) | `Driver.prepare()` returns its path; `restore_config()` undoes changes outside the campaign directory |
| the sensor builds maps (the `map` step) | `Driver.save_map()` returns the saved file |
| no `odom -> map` tf (e.g. a nav_msgs/Odometry pose in the map frame) | `Sensor.map_pose()` and `odometry()` read it from `bag`: `topics.odometry` is already parsed into `bag.odom.series(use_arrival)`, and any tf pair is in `bag.tf_series(parent, child)` |
| no IMU topic | `Sensor.heartbeat` returns another `(topic, msgtype)` that streams while the driver is up |
| driver log lines that mean trouble | `Sensor.log_issue()` (reported by `scripts/check_runs.py`) |
| device serial for `metadata.csv` | `Driver.device_serial()` |

The campaign uses three driver modes (`Mode` in base.py): odometry for preflight and calibration,
mapping for the `map` step, and relocalization for everything that is scored. A sensor without a
prior-map mode cannot run the map, session and init steps as they stand.

Scan sensors that give no pose don't fit this interface: [ld19/](ld19/) is the 2D lidar, with its
own procedure and analysis (`python -m odin_char.sensors.ld19 --help`), no `SENSOR` and no mocap.

[odin/](odin/) is the worked example: a `TfSensor` plus a driver that rewrites `control_command.yaml`
per run, saves maps over a ROS service and reads the device serial.
