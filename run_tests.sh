#!/usr/bin/env bash
# Run the test suite (uses ~/odin/venv during development; any env with requirements.txt works).
cd "$(dirname "$0")"
PY=${PYTHON:-$HOME/odin/venv/bin/python}
exec "$PY" -m pytest -x -q "$@"
