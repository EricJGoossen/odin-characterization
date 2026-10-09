"""Quick data-quality check of finished campaign runs (safe to run during a session: no point clouds,
run it with `nice`).

    .venv/bin/python scripts/check_runs.py [campaigns/odin1] [--all]

Checks every finished run not checked before (results cached in <dir>/checks.json; --all redoes
them): duration, mocap rate and gaps > 20 ms, IMU rate, odometry rate/gaps, registration (relocalization
runs), problems the sensor's driver logged during the run (e.g. Odin queue drops), and per-phase motion
(speed for laps, stillness for holds). Prints OK / WARN / BAD per run.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from odin_char.bagio import read_bag  # noqa: E402
from odin_char.robot.campaign import load_config  # noqa: E402
from odin_char.sensors import load_sensor  # noqa: E402

MOCAP_GAP = 0.02


def driver_issues(root: Path, sensor, t0: float, t1: float) -> tuple[dict, bool]:
    """Problems the driver logged between t0 and t1: ({what: count}, any serious)."""
    out: dict = {}
    serious = False
    for log in (root / "logs").glob("driver_*.log"):
        for line in log.open(errors="ignore"):
            issue = sensor.log_issue(line)
            if issue is None:
                continue
            try:
                t = float(line.split(" ", 1)[0])
            except ValueError:
                continue
            if t0 <= t <= t1:
                out[issue[0]] = out.get(issue[0], 0) + 1
                serious |= issue[1]
    return out, serious


def check_run(root: Path, cfg: dict, run: dict) -> tuple[str, list[str]]:
    notes, level = [], 0
    bag = root / run["bag"]
    if not (bag / "metadata.yaml").exists():
        return "BAD", ["no complete bag (metadata.yaml missing)"]
    b = read_bag(bag, cfg)
    dur = b.t_end - b.t_start
    notes.append(f"{dur:5.0f} s")
    sensor = load_sensor(cfg)

    def warn(msg, bad=False):
        nonlocal level
        level = max(level, 2 if bad else 1)
        notes.append(("BAD: " if bad else "warn: ") + msg)

    # mocap
    mt = np.sort(np.asarray(b.mocap.t_recv))
    expect_mocap = run.get("condition") not in ("mocap_off",)
    if len(mt) < 2:
        if expect_mocap:
            warn("no mocap", bad=True)
    else:
        rate = (len(mt) - 1) / (mt[-1] - mt[0])
        gaps = np.diff(mt)
        n_gap = int((gaps > MOCAP_GAP).sum())
        notes.append(f"mocap {rate:4.0f} Hz, {n_gap} gaps>20ms (max {gaps.max() * 1000:.0f} ms)")
        if rate < 100:
            warn(f"mocap rate {rate:.0f} Hz < 100")
        if gaps.max() > 1.0:
            warn(f"mocap gap of {gaps.max():.1f} s")
        lost = float(gaps[gaps > MOCAP_GAP].sum()) / max(mt[-1] - mt[0], 1e-6)
        if lost > 0.05:
            warn(f"{100 * lost:.0f}% of the run inside mocap gaps")
    # sensor
    if run.get("condition") not in ("sensor_off", "odin_off"):
        imu = np.asarray(b.imu_t)
        irate = len(imu) / max(dur, 1e-6)
        notes.append(f"imu {irate:3.0f} Hz")
        if irate < 300:
            warn(f"IMU rate {irate:.0f} Hz", bad=irate < 100)
        try:
            ot = np.sort(sensor.odometry(b, use_arrival=True).t)
            og = np.diff(ot)
            notes.append(f"odometry {(len(ot) - 1) / (ot[-1] - ot[0]):4.1f} Hz (max gap {og.max():.2f} s)")
            if og.max() > 0.5:
                warn(f"odometry gap {og.max():.1f} s")
        except KeyError:
            warn("no odometry", bad=True)
        try:
            nmap = len(sensor.registration(b).t)
        except KeyError:
            nmap = 0
        if run["phase"] in ("phase2", "phase3", "phase4", "outage"):
            notes.append(f"odom->map {nmap}")
            if nmap == 0:
                warn(f"no odom->map: {sensor.name} not relocalized during this run", bad=True)
        issues, serious = driver_issues(root, sensor, b.t_start, b.t_end)
        if issues:
            warn(f"driver problems {issues}", bad=serious)
    # motion
    if len(mt) > 10:
        s = b.mocap_series()
        p = s.pose.p[:, :2]
        if len(s.t) > 20:
            v = np.linalg.norm(np.diff(p, axis=0), axis=1) / np.maximum(np.diff(s.t), 1e-3)
            v = v[np.isfinite(v)]
            moving = v[v > 0.05]
            path = float(np.sum(np.linalg.norm(np.diff(p[:: max(len(p) // 2000, 1)], axis=0), axis=1)))
            notes.append(f"path {path:5.1f} m, speed median {np.median(moving) if len(moving) else 0:.2f} m/s")
            if run["phase"] == "phase3" and np.ptp(p, axis=0).max() > 0.05:
                warn(f"robot moved {np.ptp(p, axis=0).max() * 100:.0f} cm during a hold")
            if run["phase"] == "phase2" and len(moving):
                target = cfg["phases"]["phase2"]["nominal_speed"] * float(run.get("speed_class", "1x").rstrip("x"))
                med = float(np.median(moving))
                if abs(med - target) > 0.35 * target:
                    warn(f"median speed {med:.2f} vs target {target:.2f} m/s")
    return ("OK", "WARN", "BAD")[level], notes


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = Path(args[0]) if args else ROOT / "campaigns" / "odin1"
    cfg = load_config(ROOT / "config" / "campaign.yaml")
    cfg["sensor"]["topics"]["cloud"] = None  # never parse clouds here
    man = json.loads((root / "manifest.json").read_text())
    cache_p = root / "checks.json"
    cache = {} if "--all" in sys.argv or not cache_p.exists() else json.loads(cache_p.read_text())
    new = 0
    for run in man["runs"]:
        if not run.get("ended") or run["name"] in cache:
            continue
        try:
            status, notes = check_run(root, cfg, run)
        except Exception as e:  # noqa: BLE001
            status, notes = "BAD", [f"check failed: {type(e).__name__}: {e}"]
        cache[run["name"]] = {"status": status, "notes": notes}
        print(f"[{status:4s}] {run['name']:30s} " + " | ".join(notes), flush=True)
        new += 1
    cache_p.write_text(json.dumps(cache, indent=1))
    free = shutil.disk_usage(root).free / 1e9
    if new or "--all" in sys.argv:
        print(f"disk free {free:.1f} GB" + ("  (LOW)" if free < 3 else ""))
    abort = root / "aborts.log"
    if abort.exists():
        print("last abort:", abort.read_text().strip().splitlines()[-1][:200])


if __name__ == "__main__":
    main()
