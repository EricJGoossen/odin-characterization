#!/usr/bin/env bash
# One-time setup on the robot computer (the one Odin is plugged into) and/or the analysis machine.
# Creates .venv next to this repo. On the robot computer, ROS 2 and the odin_ros_driver workspace
# (ODIN_WS, default ~/odin_ws) are sourced automatically so rclpy is visible inside the venv.
set -eo pipefail
cd "$(dirname "$0")/.."

command -v ros2 >/dev/null 2>&1 || { [ -f /opt/ros/${ROS_DISTRO:-humble}/setup.bash ] && source /opt/ros/${ROS_DISTRO:-humble}/setup.bash; }
if command -v ros2 >/dev/null 2>&1; then
    ros2 pkg prefix odin_ros_driver >/dev/null 2>&1 || { [ -f "${ODIN_WS:-$HOME/odin_ws}/install/setup.bash" ] && source "${ODIN_WS:-$HOME/odin_ws}/install/setup.bash"; }
    echo "ROS 2 found ($ROS_DISTRO)"
    python3 -m venv --system-site-packages .venv   # keeps rclpy / tf2_msgs from the ROS install
    if ! ros2 pkg prefix odin_ros_driver >/dev/null 2>&1; then
        echo "WARNING: odin_ros_driver not found (set ODIN_WS); needed for the campaign."
    fi
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
