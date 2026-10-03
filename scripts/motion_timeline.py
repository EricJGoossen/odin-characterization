"""Coarse timeline of a bag's mocap motion: per 5 s window, translation and rotation activity.

    .venv/bin/python scripts/motion_timeline.py campaigns/odin1/bags/014_latency
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from odin_char.bagio import read_bag  # noqa: E402
from odin_char.robot.campaign import load_config  # noqa: E402

cfg = load_config(ROOT / "config" / "campaign.yaml")
cfg["topics"]["cloud"] = None
b = read_bag(Path(sys.argv[1]), cfg)
s = b.mocap_series()
t = s.t - s.t[0]
yaw = np.unwrap(np.arctan2(s.pose.R[:, 1, 0], s.pose.R[:, 0, 0]))
print(f"bag {b.t_end - b.t_start:.1f} s, mocap {t[-1]:.1f} s")
for w0 in np.arange(0, t[-1], 5.0):
    m = (t >= w0) & (t < w0 + 5)
    if m.sum() < 10:
        continue
    p = s.pose.p[m, :2]
    path = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    turn = float(np.sum(np.abs(np.diff(yaw[m]))))
    kind = "TRANSLATE" if path > 0.3 and path > turn * 0.3 else ("ROTATE" if turn > 0.5 else "still")
    print(f"  {w0:5.0f}-{w0 + 5:3.0f} s  path {path:5.2f} m  turned {np.degrees(turn):6.0f} deg  {kind}")
