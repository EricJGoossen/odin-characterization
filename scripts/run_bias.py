"""Per-run / per-relocalization mean error after a pipeline run (uses results/calibration.json).

Splits every run at driver restarts (gaps in the sensor's odometry) so each block is one relocalization,
then prints the map-frame position error and attitude error mean/std per block.

    .venv/bin/python scripts/run_bias.py campaigns/odin1
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from odin_char.errors import Calibration  # noqa: E402
from odin_char.pipeline import Campaign  # noqa: E402
from odin_char.se3 import Pose  # noqa: E402

camp = Path(sys.argv[1])
cfg = yaml.safe_load((camp / "campaign.yaml").read_text())
c = Campaign(camp, cfg)
cal = json.loads((camp / "results" / "calibration.json").read_text())
model = yaml.safe_load((camp / "results" / "model.yaml").read_text())
tau = float(model.get("tau_d", 0.0))
calib = Calibration(cal["time_offset_s"], Pose.from_vec(np.array(cal["X_T_base_to_imu"]["vec_txyz_rotvec"])),
                    Pose.from_vec(np.array(cal["T_WM_mocap_to_map"]["vec_txyz_rotvec"])))
print(f"tau_d {tau * 1e3:.1f} ms")
print(f"{'run':24s} {'blk':>3s} {'t0':>6s} {'dur':>5s} {'n':>5s} | {'ex_mm':>6s} {'ey_mm':>6s} {'ez_mm':>6s} "
      f"{'sd_xy':>6s} | {'roll':>6s} {'pitch':>6s} {'yaw':>6s} mrad  {'sd_yaw':>6s}")
rows = []
for r in c.manifest["runs"]:
    if r["phase"] in ("preflight",) or r.get("excluded"):
        continue
    try:
        tab = c.errors([r], calib, tau)
    except (KeyError, ValueError) as e:
        print(f"{r['name']:24s} error: {e}")
        continue
    tab = tab.select(tab.map_age < 1.0)
    if len(tab) < 20:
        print(f"{r['name']:24s} too few tracked samples ({len(tab)})")
        continue
    b = c.bag(r)
    oi = c.sensor.odometry(b).t + calib.time_offset
    restarts = oi[1:][np.diff(oi) > 2.0]
    blk = np.searchsorted(restarts, tab.t)
    ew = tab.world_position_error()
    t_run0 = b.t_start
    for k in np.unique(blk):
        m = blk == k
        if m.sum() < 20:
            continue
        e, a = ew[m] * 1e3, tab.err[m, 3:] * 1e3
        rows.append((r["name"], k, e.mean(0), a.mean(0)))
        print(f"{r['name']:24s} {k:3d} {tab.t[m][0] - t_run0:6.1f} {np.ptp(tab.t[m]):5.1f} {m.sum():5d} | "
              f"{e[:, 0].mean():+6.1f} {e[:, 1].mean():+6.1f} {e[:, 2].mean():+6.1f} {np.sqrt(e[:, :2].var(0).sum()):6.1f} | "
              f"{a[:, 0].mean():+6.2f} {a[:, 1].mean():+6.2f} {a[:, 2].mean():+6.2f}       {a[:, 2].std():6.2f}")
