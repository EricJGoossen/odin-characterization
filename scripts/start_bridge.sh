#!/usr/bin/env bash
# Starts project otto in Odin characterization mode on the sentry's Jetson (instead of otto's main
# loop: both need the UART). Forwards this repo's UDP velocity commands to the MCB (message 0x14).
#
#   scripts/start_bridge.sh [extra args]     e.g. --dummy, --port /dev/ttyUSB0, --baud 1000000
#
# OTTO_DIR: project-otto checkout on the odin-characterization branch (default ~/project-otto).
# Equivalent to: cd $OTTO_DIR && poetry run odin-characterization [args]
set -euo pipefail
OTTO_DIR=${OTTO_DIR:-$HOME/project-otto}

if systemctl is-active --quiet project_otto.service 2>/dev/null; then
    echo "project_otto.service is running and owns the UART: sudo systemctl stop project_otto.service" >&2
    exit 1
fi
if pgrep -f "project_otto/(new_)?main.py" >/dev/null; then
    echo "project otto's main loop is running and owns the UART: stop it first." >&2
    exit 1
fi
cd "$OTTO_DIR"
branch=$(git rev-parse --abbrev-ref HEAD)
[ "$branch" = "odin-characterization" ] || echo "WARNING: project-otto is on '$branch', not odin-characterization"
exec poetry run odin-characterization "$@"
