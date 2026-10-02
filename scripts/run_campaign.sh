#!/usr/bin/env bash
# Data collection. Run on the robot computer with ROS 2 and the Odin driver workspace sourced.
#
#   scripts/run_campaign.sh day1 [campaign_dir]     everything except the 2nd/3rd sessions
#   scripts/run_campaign.sh day2 [campaign_dir]     Phase 2 session on another day
#   scripts/run_campaign.sh day3 [campaign_dir]
#   scripts/run_campaign.sh <step> [campaign_dir]   one step: check preflight calib map session
#                                                    holds latency init outage
# Extra flags are passed through (e.g. --no-prompt, --redo).
set -euo pipefail
cd "$(dirname "$0")/.."
STEP=${1:?usage: run_campaign.sh <day1|day2|day3|step> [campaign_dir] [flags]}
shift
DIR=campaigns/odin1
if [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; then
    DIR=$1
    shift
fi
command -v ros2 >/dev/null || { echo "source ROS 2 and the odin driver workspace first" >&2; exit 1; }
exec .venv/bin/python -m odin_char.robot.campaign "$STEP" --config config/campaign.yaml --dir "$DIR" "$@"
