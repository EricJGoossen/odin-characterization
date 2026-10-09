# Sourced (after ROS 2) by scripts/setup.sh and scripts/run_campaign.sh: puts odin_ros_driver on the
# ROS path from its workspace (ODIN_WS, default ~/odin_ws).
if ! ros2 pkg prefix odin_ros_driver >/dev/null 2>&1 && [ -f "${ODIN_WS:-$HOME/odin_ws}/install/setup.bash" ]; then
    source "${ODIN_WS:-$HOME/odin_ws}/install/setup.bash"
fi

sensor_env_check() {
    ros2 pkg prefix odin_ros_driver >/dev/null 2>&1 || { echo "odin_ros_driver not found: set ODIN_WS to its workspace" >&2; return 1; }
}
