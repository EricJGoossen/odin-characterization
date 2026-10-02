#!/usr/bin/env bash
# Fit and validate the Tier 0 model from a campaign directory (any machine, no ROS needed).
#   scripts/analyze.sh [campaign_dir]
# Writes <campaign_dir>/results/{report.md, model.yaml, static_field_grid.npz, calibration.json,
# results.json, figures/}.
set -euo pipefail
cd "$(dirname "$0")/.."
DIR=${1:-campaigns/odin1}
.venv/bin/python -m odin_char.pipeline "$DIR"
echo "report: $DIR/results/report.md"
