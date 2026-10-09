# Sourced (after ROS 2) by scripts/ld19.sh: puts the LD19 driver, rf2o and slam_toolbox on the ROS path
# from their workspace (LD19_WS, default ~/ld19_ws) if they aren't installed system-wide.
if ! ros2 pkg prefix ldlidar_stl_ros2 >/dev/null 2>&1 && [ -f "${LD19_WS:-$HOME/ld19_ws}/install/setup.bash" ]; then
    source "${LD19_WS:-$HOME/ld19_ws}/install/setup.bash"
fi

sensor_env_check() {
    local missing=""
    for pkg in ldlidar_stl_ros2 rf2o_laser_odometry slam_toolbox; do
        ros2 pkg prefix "$pkg" >/dev/null 2>&1 || missing="$missing $pkg"
    done
    [ -z "$missing" ] || { echo "missing ROS packages:$missing (set LD19_WS to their workspace)" >&2; return 1; }
}
