# Sourced (after ROS 2, from the repo root) by setup.sh and run_campaign.sh: sources the configured
# sensor's odin_char/sensors/<type>/env.sh, which puts its driver on the ROS path and may define
# sensor_env_check (non-zero with a message when the driver is missing).
SENSOR_TYPE=$(awk '/^sensor:/ {s = 1; next} s && /^[^[:space:]#]/ {s = 0} s && /^[[:space:]]+type:/ {print $2; exit}' \
    "${CAMPAIGN_CONFIG:-config/campaign.yaml}")
SENSOR_TYPE=${SENSOR_TYPE:-odin}
sensor_env_check() { :; }
if [ -f "odin_char/sensors/$SENSOR_TYPE/env.sh" ]; then
    source "odin_char/sensors/$SENSOR_TYPE/env.sh"
fi
