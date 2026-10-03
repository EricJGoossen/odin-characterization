"""Mark campaign runs as excluded from the analysis (they stay in the manifest for the record).

    .venv/bin/python scripts/exclude_runs.py campaigns/odin1 --session s2 [--reason "..."]
    .venv/bin/python scripts/exclude_runs.py campaigns/odin1 --run 013_s2_1x
Refuses while the campaign is running (it rewrites manifest.json).
"""
import argparse
import json
import subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("dir", type=Path)
ap.add_argument("--session", action="append", default=[])
ap.add_argument("--run", action="append", default=[])
ap.add_argument("--reason", default="abandoned")
a = ap.parse_args()
if subprocess.run(["pgrep", "-f", "odin_char.robot.campaign"], capture_output=True).returncode == 0:
    raise SystemExit("the campaign is running: stop it first")
mp = a.dir / "manifest.json"
m = json.loads(mp.read_text())
n = 0
for r in m["runs"]:
    if r.get("session") in a.session or r["name"] in a.run:
        r["excluded"] = a.reason
        n += 1
        print("excluded", r["name"])
mp.write_text(json.dumps(m, indent=1))
print(f"{n} run(s) excluded")
