#!/usr/bin/env bash
# LD19 2D lidar characterization (no mocap). See `python -m odin_char.sensors.ld19 --help`.
#   scripts/ld19.sh check | static | map | analyze  [--dir campaigns/ld19] [flags]
set -euo pipefail
cd "$(dirname "$0")/.."
STEP=${1:?usage: ld19.sh <check|static|map|analyze> [--dir dir] [flags]}
if [ "$STEP" != analyze ]; then
    set +u
    command -v ros2 >/dev/null || source /opt/ros/${ROS_DISTRO:-humble}/setup.bash
    source odin_char/sensors/ld19/env.sh
    set -u
    if [ "$STEP" = check ] || [ "$STEP" = static ]; then
        ros2 pkg prefix ldlidar_stl_ros2 >/dev/null 2>&1 || { echo "ldlidar_stl_ros2 not found (set LD19_WS)" >&2; exit 1; }
    else
        sensor_env_check || exit 1
    fi
fi
exec .venv/bin/python -m odin_char.sensors.ld19 "$@"
