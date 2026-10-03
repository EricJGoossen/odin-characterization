"""Estimate how long each campaign step takes with the current config.

    .venv/bin/python scripts/estimate_time.py [config/campaign.yaml]

Motion times come from the exact trajectories the campaign builds. Things that depend on the
hardware are assumptions, listed at the top (driver start, relocalization time, map save), so
treat those lines as rough.
"""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))
from odin_char.robot.motion import (  # noqa: E402
    calib_motion,
    figure8,
    follow_path,
    goto,
    lawnmower_path,
    line_back_and_forth,
    sinusoid_axis,
    yaw_sinusoid,
)

# ---- assumptions (hardware-dependent, not measurable here)
DRIVER_START_S = 10.0  # launch -> first IMU message
RELOC_S = 10.0  # driver start -> relocalization fix (timeout is phases.phase5.timeout_s)
MAP_SAVE_S = 60.0  # /odin1/save_map transfer
BAG_OVERHEAD_S = 4.0  # ros2 bag start/stop per run
DRIVER_STOP_S = 3.0
PROMPT_S = 30.0  # a person doing something when prompted

cfg = yaml.safe_load(Path(sys.argv[1] if len(sys.argv) > 1 else "config/campaign.yaml").read_text())
rb, room, ph = cfg["robot"], cfg["room"], cfg["phases"]
home = np.array(room["home"], float)
a = rb["max_accel"]


def go(p, q):
    return goto(p, q, min(0.6, rb["max_speed"]), a, 1.0, rb["max_yaw_accel"]).duration


rows = []
warm = cfg["driver"]["warmup_min"] * 60
rows.append(("check (gain identification)", 6 * 1.0 + 6 * 0.7 + PROMPT_S, "1 prompt"))
rows.append(("preflight", warm + 3 * ph["preflight"]["duration_s"] + 4 * PROMPT_S + 3 * BAG_OVERHEAD_S
             + DRIVER_START_S + DRIVER_STOP_S, "4 prompts"))
c = calib_motion(home, ph["calib"]["duration_s"], 0.8)
rows.append(("calib", c.duration + go(home, c.start) + BAG_OVERHEAD_S, ""))
lo = np.array(room["fence_min"]) + room["margin"] + 0.3
hi = np.array(room["fence_max"]) - room["margin"] - 0.3
m = follow_path(lawnmower_path(lo, hi, ph["mapping"]["lane_spacing"], ph["mapping"]["loops"]), ph["mapping"]["speed"], a)
rows.append(("map", m.duration + go(home, m.start) + DRIVER_START_S + MAP_SAVE_S + BAG_OVERHEAD_S + DRIVER_STOP_S,
             f"path {np.sum(np.linalg.norm(np.diff(m.xy, axis=0), axis=1)):.0f} m"))
p2 = ph["phase2"]
rot = np.radians(p2.get("rotation_deg", 0))
f1 = figure8(p2["center"], p2["half_length"], p2["half_width"], rot, p2["nominal_laps"], p2["nominal_speed"], a)
sess = f1.duration + go(home, f1.start) + DRIVER_START_S + RELOC_S + 3 + BAG_OVERHEAD_S
for f in p2["sweep_factors"]:
    t = figure8(p2["center"], p2["half_length"], p2["half_width"], rot, p2["sweep_laps"], p2["nominal_speed"] * f, a)
    sess += t.duration + go(f1.end, t.start) + BAG_OVERHEAD_S
lap = figure8(p2["center"], p2["half_length"], p2["half_width"], rot, 1, p2["nominal_speed"], a).duration
rows.append(("session (one)", sess + PROMPT_S, f"1 lap at nominal = {lap:.0f} s"))
holds = sum(ph["phase3"]["hold_s"] + go(home, loc) + BAG_OVERHEAD_S for loc in ph["phase3"]["locations"])
rows.append(("holds", holds + DRIVER_START_S + RELOC_S, ""))
p4 = ph["phase4"]
lat = sinusoid_axis(home, home[2], p4["frequencies_hz"], p4["peak_speed"], p4["seconds_per_frequency"]).duration + \
    yaw_sinusoid(home, p4["frequencies_hz"], p4["peak_yaw_rate"], p4["seconds_per_frequency"]).duration
rows.append(("latency", lat + BAG_OVERHEAD_S, ""))
p5 = ph["phase5"]
n_init = len(p5["offsets_m"]) * len(p5["heading_offsets_deg"]) * len(p5["start_behaviours"])
per_init = DRIVER_STOP_S + 8.0 + BAG_OVERHEAD_S + DRIVER_START_S + RELOC_S + 3.0  # ~8 s drive to the start pose
moving_extra = go(home, home + [1, 0, 0]) + go(home, home + [0, 0, np.pi / 2])
rows.append(("init", n_init * per_init + n_init / 2 * moving_extra,
             f"{n_init} trials; worst case (no fixes) {n_init * (per_init - RELOC_S + p5['timeout_s']) / 60:.0f} min"))
out = 0.0
for d in p5["cover_durations_s"]:
    tr = line_back_and_forth(home, 1.5, p5["cover_speed"], 0.5, d + 15)
    out += p5["cover_repeats"] * (tr.duration + 5 + BAG_OVERHEAD_S + 3)
rows.append(("outage", out + PROMPT_S, f"{len(p5['cover_durations_s']) * p5['cover_repeats']} trials, person covers Odin"))

print(f"{'step':30s} {'minutes':>8s}   notes")
total = 0.0
for name, s, note in rows:
    total += s
    print(f"{name:30s} {s / 60:8.1f}   {note}")
print(f"{'day1 total':30s} {total / 60:8.1f}")
day2 = rows[0][1] + next(s for n, s, _ in rows if n.startswith("session")) + warm
print(f"{'day2 / day3 (check + session)':30s} {day2 / 60:8.1f}")
print(f"\nassumed: driver start {DRIVER_START_S:.0f} s, relocalization {RELOC_S:.0f} s, map save {MAP_SAVE_S:.0f} s, "
      f"bag start/stop {BAG_OVERHEAD_S:.0f} s, each prompt {PROMPT_S:.0f} s, warm-up {warm / 60:.0f} min")
