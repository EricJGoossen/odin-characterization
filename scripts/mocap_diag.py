"""Mocap stream diagnostics for one bag: duplicates, publishers, gaps and where they happen.

    .venv/bin/python scripts/mocap_diag.py campaigns/odin1/bags/007_calib
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from odin_char.bagio import read_bag  # noqa: E402
from odin_char.robot.campaign import load_config  # noqa: E402

bag = Path(sys.argv[1])
cfg = load_config(ROOT / "config" / "campaign.yaml")
cfg["sensor"]["topics"]["cloud"] = None
topic = cfg["mocap"]["topic"]
from rosbags.highlevel import AnyReader  # noqa: E402
from rosbags.typesys import Stores, get_typestore  # noqa: E402

with AnyReader([bag], default_typestore=get_typestore(Stores.LATEST)) as r:
    conns = [c for c in r.connections if c.topic == topic]
    print(f"{topic}: {len(conns)} connection(s), counts {[c.msgcount for c in conns]}")
b = read_bag(bag, cfg)
st = np.asarray(b.mocap.t_stamp)
rt = np.asarray(b.mocap.t_recv)
p = np.asarray(b.mocap.p)
print(f"samples {len(st)}, unique stamps {len(np.unique(st))}, exact duplicate (stamp,pos) "
      f"{len(st) - len(np.unique(np.column_stack([st, p]), axis=0))}")
o = np.argsort(st)
ds = np.diff(st[o])
print(f"stamp spacing: median {np.median(ds) * 1000:.2f} ms, zeros {int((ds == 0).sum())}, "
      f"backwards in arrival order {int((np.diff(st) < 0).sum())}")
s = b.mocap_series()
gaps = np.diff(s.t)
big = np.where(gaps > 0.1)[0]
print(f"gaps > 100 ms: {len(big)}")
for i in big[:15]:
    print(f"  {gaps[i]:5.2f} s at x {s.pose.p[i, 0]:+.2f}, y {s.pose.p[i, 1]:+.2f} -> resumes at x {s.pose.p[i + 1, 0]:+.2f}, y {s.pose.p[i + 1, 1]:+.2f}")
v = np.linalg.norm(np.diff(s.pose.p[:, :2], axis=0), axis=1) / np.maximum(gaps, 1e-4)
print(f"speed (deduplicated series): median {np.median(v[v > 0.05]):.2f} m/s, p95 {np.percentile(v, 95):.2f}")
