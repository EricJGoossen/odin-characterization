#!/usr/bin/env bash
# Run on the sentry's Jetson INSTEAD of project otto's main loop (both use the UART).
# Forwards the campaign's UDP velocity commands to the MCB (message 0x14) at 100 Hz and sends
# zeros if the campaign goes quiet for 100 ms.
#
#   scripts/start_bridge.sh [serial_port] [baud] [udp_port]
#
# OTTO_DIR: project-otto checkout on the odin-characterization branch (default ~/project-otto)
set -euo pipefail
OTTO_DIR=${OTTO_DIR:-$HOME/project-otto}
PORT=${1:-/dev/ttyTHS0}
BAUD=${2:-1000000}      # sentry (TARGET_SENTRY_ACHLYS) uses 1 Mbaud
UDP=${3:-47020}

if pgrep -f "project_otto/(new_)?main.py" >/dev/null; then
    echo "project otto's main loop is running and owns the UART: stop it first." >&2
    exit 1
fi
cd "$OTTO_DIR"
branch=$(git rev-parse --abbrev-ref HEAD)
[ "$branch" = "odin-characterization" ] || echo "WARNING: project-otto is on '$branch', not odin-characterization"
exec poetry run chassis-velocity-bridge "$PORT" "$BAUD" "$UDP"
