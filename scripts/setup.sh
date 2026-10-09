#!/usr/bin/env bash
# One-time setup on the robot computer (the one the sensor is plugged into) and/or the analysis machine.
# Creates .venv next to this repo. On the robot computer, ROS 2 and the sensor driver's workspace
# (odin_char/sensors/<type>/env.sh) are sourced automatically so rclpy is visible inside the venv.
set -eo pipefail
cd "$(dirname "$0")/.."

command -v ros2 >/dev/null 2>&1 || { [ -f /opt/ros/${ROS_DISTRO:-humble}/setup.bash ] && source /opt/ros/${ROS_DISTRO:-humble}/setup.bash; }
if command -v ros2 >/dev/null 2>&1; then
    source scripts/sensor_env.sh
    echo "ROS 2 found ($ROS_DISTRO)"
    python3 -m venv --system-site-packages .venv   # keeps rclpy / tf2_msgs from the ROS install
    sensor_env_check || echo "WARNING: the $SENSOR_TYPE driver is needed for the campaign."

else
    echo "ROS 2 not found: setting up for analysis only."
    python3 -m venv .venv
fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
if [ "${SKIP_TESTS:-0}" != 1 ]; then
    echo "running the self-tests (analysis + motion, a few minutes) ..."
    .venv/bin/python -m pytest -q tests/test_motion.py tests/test_mocap_config.py tests/test_validate.py tests/test_pipeline.py
fi
echo
echo "setup OK. Next: edit config/campaign.yaml (every 'VERIFY'), then see README.md."
