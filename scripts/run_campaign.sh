#!/usr/bin/env bash
# Data collection. Run on the robot computer with ROS 2 and the Odin driver workspace sourced.
#
# Bring-up (run in this order the first time; nothing is recorded):
#   scripts/run_campaign.sh find-mocap          list live mocap candidates -> paste into config
#   scripts/run_campaign.sh mocap  [dir]        watch the configured mocap stream
#   scripts/run_campaign.sh odin   [dir]        start/stop Odin, report its topics
#   scripts/run_campaign.sh jog    [dir]        tiny pulses: does the robot move the right way?
#   scripts/run_campaign.sh check  [dir]        identify command gains from mocap
#
# Collection:
#   scripts/run_campaign.sh day1 [dir]          everything except the 2nd/3rd sessions
#   scripts/run_campaign.sh day2 [dir]          Phase 2 session on another day
#   scripts/run_campaign.sh day3 [dir]
#   scripts/run_campaign.sh <step> [dir]        one step: preflight calib map session holds
#                                               latency init outage
# dir defaults to campaigns/odin1. Extra flags pass through (--no-prompt, --redo, --skip-warmup).
set -euo pipefail
cd "$(dirname "$0")/.."
STEP=${1:?usage: run_campaign.sh <find-mocap|mocap|odin|jog|check|day1|day2|day3|step> [campaign_dir] [flags]}
shift
command -v ros2 >/dev/null || { echo "source ROS 2 and the odin driver workspace first" >&2; exit 1; }
if [ "$STEP" = "find-mocap" ]; then
    exec .venv/bin/python -m odin_char.robot.discover "$@"
fi
DIR=campaigns/odin1
if [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; then
    DIR=$1
    shift
fi
exec .venv/bin/python -m odin_char.robot.campaign "$STEP" --config config/campaign.yaml --dir "$DIR" "$@"
