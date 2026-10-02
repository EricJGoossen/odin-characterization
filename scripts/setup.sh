#!/usr/bin/env bash
# One-time setup on the robot computer (the one Odin is plugged into) and/or the analysis machine.
# Creates .venv next to this repo. On the robot computer, source ROS 2 first so rclpy is visible.
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v ros2 >/dev/null 2>&1; then
    echo "ROS 2 found: $(ros2 --version 2>/dev/null || echo ok)"
    python3 -m venv --system-site-packages .venv   # keeps rclpy / tf2_msgs from the ROS install
    if ! ros2 pkg prefix odin_ros_driver >/dev/null 2>&1; then
        echo "WARNING: odin_ros_driver not found in the sourced ROS workspace (needed for the campaign)."
    fi
else
    echo "ROS 2 not sourced: setting up for analysis only."
    python3 -m venv .venv
fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
echo "running the self-tests (analysis + motion, ~2 min) ..."
.venv/bin/python -m pytest -q tests/test_motion.py tests/test_validate.py tests/test_pipeline.py
echo
echo "setup OK. Next: edit config/campaign.yaml (every 'VERIFY'), then see README.md."
