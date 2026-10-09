"""LD19 analysis (no ROS needed): what the raw scans get wrong, and how well they line up with the
map built from them. Writes <dir>/results/{report.md, results.json, map.pgm, map.yaml, figures/}.

static runs (lidar still): scan rate and timing, points per scan, per-direction dropouts, range
noise vs range, outliers and isolated ("ghost") points.

map runs (pushed around, SLAM pose from rf2o + slam_toolbox): distance of every scan point to the
nearest obstacle in the final map, by range, turn rate and speed (spin smear shows up here), which
per-point timing convention lines the scans up best, and how much the scan-only odometry drifted.
There is no external truth: a bias common to all scans, or one that SLAM absorbs, is invisible.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...bagio import BagData, _stamp, _xyz, _xyzw
from ...se3 import Pose, PoseSeries, wrap, yaw_of

OCCUPIED = 50  # OccupancyGrid value above which a cell counts as an obstacle


@dataclass
class Scans:
    t: np.ndarray  # header stamps
    t_recv: np.ndarray  # bag (arrival) times
    angles: list = field(default_factory=list)  # per scan: angle of every beam (rad, sensor frame)
    ranges: list = field(default_factory=list)
    intensities: list = field(default_factory=list)
    time_increment: np.ndarray | None = None  # s between beams, as the driver reports it
    range_min: float = 0.0
    range_max: float = np.inf

    def valid(self, k: int) -> np.ndarray:
        r = self.ranges[k]
        return np.isfinite(r) & (r > max(self.range_min, 1e-3)) & (r < self.range_max)


@dataclass
class Grid:
    data: np.ndarray  # (height, width) int8, -1 unknown, 0..100 occupancy
    resolution: float
    origin: np.ndarray  # x, y, yaw of cell (0, 0)'s corner in the map frame

    def to_cells(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c, s = np.cos(self.origin[2]), np.sin(self.origin[2])
        d = xy - self.origin[:2]
        lx, ly = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
        ix, iy = np.floor(lx / self.resolution).astype(int), np.floor(ly / self.resolution).astype(int)
        ok = (ix >= 0) & (iy >= 0) & (ix < self.data.shape[1]) & (iy < self.data.shape[0])
        return ix, iy, ok

    def cell_centres(self, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        c, s = np.cos(self.origin[2]), np.sin(self.origin[2])
        lx, ly = (ix + 0.5) * self.resolution, (iy + 0.5) * self.resolution
        return self.origin[:2] + np.column_stack([c * lx - s * ly, s * lx + c * ly])


def read_bag(path: Path, cfg: dict) -> tuple[Scans, BagData, Grid | None]:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    t, tr, ang, rng, inten, tinc = [], [], [], [], [], []
    rmin, rmax = 0.0, np.inf
    tf = BagData(path=str(path))
    grid = None
    with AnyReader([Path(path)], default_typestore=get_typestore(Stores.LATEST)) as reader:
        names = {cfg["scan_topic"], cfg["map_topic"], "/tf", "/tf_static"}
        for conn, t_ns, raw in reader.messages(connections=[c for c in reader.connections if c.topic in names]):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == cfg["scan_topic"]:
                r = np.asarray(msg.ranges, float)
                t.append(_stamp(msg.header))
                tr.append(t_ns * 1e-9)
                ang.append(msg.angle_min + msg.angle_increment * np.arange(len(r)))
                rng.append(r)
                inten.append(np.asarray(msg.intensities, float) if len(msg.intensities) == len(r) else np.full(len(r), np.nan))
                tinc.append(float(msg.time_increment))
                rmin, rmax = float(msg.range_min), float(msg.range_max)
            elif conn.topic == cfg["map_topic"]:
                o = msg.info.origin
                yaw = float(yaw_of(Pose.from_quat(np.zeros(3), np.array(_xyzw(o.orientation))).R))
                grid = Grid(np.asarray(msg.data, np.int8).reshape(msg.info.height, msg.info.width),
                            float(msg.info.resolution), np.array([o.position.x, o.position.y, yaw]))
            else:
                for x in msg.transforms:
                    s = tf.tf[(x.header.frame_id.lstrip("/"), x.child_frame_id.lstrip("/"))]
                    s.t_stamp.append(_stamp(x.header))
                    s.t_recv.append(t_ns * 1e-9)
                    s.p.append(_xyz(x.transform.translation))
                    s.q.append(_xyzw(x.transform.rotation))
    if not t:
        raise ValueError(f"no scans on {cfg['scan_topic']} in {path}")
    o = np.argsort(t)
    pick = lambda xs: [xs[i] for i in o]
    return (Scans(np.asarray(t)[o], np.asarray(tr)[o], pick(ang), pick(rng), pick(inten), np.asarray(tinc)[o], rmin, rmax),
            tf, grid)


# ---------------------------------------------------------------------------- timing
def timing(s: Scans) -> dict:
    dt = np.diff(s.t)
    period = float(np.median(dt)) if len(dt) else np.nan
    npts = np.array([len(r) for r in s.ranges])
    valid = np.array([s.valid(k).sum() for k in range(len(s.ranges))])
    inc = s.time_increment[s.time_increment > 0]
    return {
        "scans": len(s.t), "duration_s": float(s.t[-1] - s.t[0]),
        "rate_hz": 1.0 / period, "period_jitter_ms": float(np.std(dt) * 1e3),
        "gaps": int((dt > 1.5 * period).sum()), "longest_gap_s": float(dt.max()) if len(dt) else np.nan,
        "points_per_scan": float(np.median(npts)), "points_per_scan_range": [int(npts.min()), int(npts.max())],
        "valid_fraction": float(valid.sum() / max(npts.sum(), 1)),
        "arrival_minus_stamp_ms": float(np.median(s.t_recv - s.t) * 1e3),
        "time_increment_us": float(np.median(inc) * 1e6) if len(inc) else None,
        "beam_spacing_deg": float(np.degrees(np.median([np.median(np.diff(a)) for a in s.angles if len(a) > 1]))),
    }


# ---------------------------------------------------------------------------- static
RANGE_BINS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.5]


def static_analysis(s: Scans, bin_deg: float = 1.0, min_samples: int = 20) -> dict:
    """Per direction (1 deg bins): dropout fraction, median range, robust temporal std, outliers.
    The scene is assumed still, so the spread of one direction's ranges over time is sensor noise."""
    a = np.concatenate([np.mod(np.degrees(x), 360.0) for x in s.angles])
    r = np.concatenate(s.ranges)
    v = np.concatenate([s.valid(k) for k in range(len(s.ranges))])
    inten = np.concatenate(s.intensities)
    nb = int(round(360 / bin_deg))
    b = np.minimum((a / bin_deg).astype(int), nb - 1)
    rows = []
    for i in range(nb):
        m = b == i
        rv = r[m & v]
        row = {"angle_deg": (i + 0.5) * bin_deg, "n": int(m.sum()), "dropout": float(1 - (m & v).sum() / max(m.sum(), 1))}
        if len(rv) >= min_samples:
            med = float(np.median(rv))
            sig = float(1.4826 * np.median(np.abs(rv - med)))
            row.update(range=med, sigma=sig, outlier=float(np.mean(np.abs(rv - med) > max(5 * sig, 0.05))),
                       intensity=float(np.nanmedian(inten[m & v])) if np.isfinite(inten[m & v]).any() else None)
        rows.append(row)
    good = [x for x in rows if "sigma" in x]
    rr = np.array([x["range"] for x in good])
    sg = np.array([x["sigma"] for x in good])
    by_range = []
    for lo, hi in zip(RANGE_BINS[:-1], RANGE_BINS[1:]):
        m = (rr >= lo) & (rr < hi)
        if m.sum():
            by_range.append({"range_m": [lo, hi], "directions": int(m.sum()), "sigma_median_mm": float(np.median(sg[m]) * 1e3),
                             "sigma_p90_mm": float(np.percentile(sg[m], 90) * 1e3)})
    fit = np.polyfit(rr, sg, 1) if len(rr) >= 5 else [np.nan, np.nan]
    # isolated points: a valid return that differs from both neighbours by > 0.2 m (mixed pixels at
    # edges, spurious returns): each one can become a false obstacle cell
    iso = []
    for k in range(len(s.ranges)):
        x, ok = s.ranges[k], s.valid(k)
        if len(x) < 3:
            continue
        d1, d2 = np.abs(x[1:-1] - x[:-2]), np.abs(x[1:-1] - x[2:])
        both = ok[1:-1] & ok[:-2] & ok[2:]
        iso.append(((d1 > 0.2) & (d2 > 0.2) & both).sum() / max(ok.sum(), 1))
    vr = r[v]
    return {
        "directions_with_data": len(good),
        "dropout_overall": float(1 - v.mean()),
        "dropout_directions_over_50pct": int(sum(x["dropout"] > 0.5 for x in rows)),
        "sigma_median_mm": float(np.median(sg) * 1e3) if len(sg) else None,
        "sigma_vs_range": by_range,
        "sigma_fit_mm": {"at_0m": float(fit[1] * 1e3), "per_m": float(fit[0] * 1e3)},
        "outlier_fraction": float(np.mean([x["outlier"] for x in good])) if good else None,
        "isolated_point_fraction": float(np.mean(iso)) if iso else None,
        "range_seen_m": [float(vr.min()), float(np.percentile(vr, 99.5))] if len(vr) else None,
        "_directions": rows,
    }


# ---------------------------------------------------------------------------- map
def laser_poses(tf: BagData, frame: str) -> tuple[PoseSeries, PoseSeries | None]:
    """odom -> laser (scan-matching odometry) and map -> odom (SLAM correction), if recorded."""
    odom = tf.tf_series("odom", frame)
    try:
        corr = tf.tf_series("map", "odom")
    except KeyError:
        corr = None
    return odom, corr


def pose_at(odom: PoseSeries, corr: PoseSeries | None, t: np.ndarray) -> tuple[Pose, np.ndarray]:
    T_OL, ok = odom.interp(t, max_gap=0.5)
    if corr is None:
        return T_OL, ok
    T_MO, has, _ = corr.zoh(t)
    return T_MO @ T_OL, ok & has


def scan_points(s: Scans, ks: np.ndarray, timing_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Valid points of scans ks: (time of each point, x, y in the laser frame, range, scan index).
    timing_mode: 'stamp' (whole scan at the header stamp), 'first' (stamp = first beam, later beams
    time_increment apart, the ROS convention), 'last' (stamp = last beam)."""
    ts, xs, ys, rs, kk = [], [], [], [], []
    period = float(np.median(np.diff(s.t))) if len(s.t) > 1 else 0.1
    for k in ks:
        ok = s.valid(k)
        n = len(s.ranges[k])
        inc = s.time_increment[k] if s.time_increment[k] > 0 else period / max(n, 1)
        i = np.arange(n)
        off = {"stamp": 0.0 * i, "first": i * inc, "last": (i - (n - 1)) * inc}[timing_mode]
        ts.append(s.t[k] + off[ok])
        xs.append(s.ranges[k][ok] * np.cos(s.angles[k][ok]))
        ys.append(s.ranges[k][ok] * np.sin(s.angles[k][ok]))
        rs.append(s.ranges[k][ok])
        kk.append(np.full(ok.sum(), k))
    cat = np.concatenate
    return cat(ts), np.column_stack([cat(xs), cat(ys)]), cat(rs), cat(kk)


def residuals(s: Scans, odom, corr, grid: Grid, ks: np.ndarray, timing_mode: str) -> dict:
    from scipy.ndimage import distance_transform_edt

    occ = grid.data > OCCUPIED
    # nearest obstacle cell of every cell; the residual is the exact distance from the point to that
    # cell's centre (not cell-to-cell, which only comes in whole cells)
    _, (ny, nx) = distance_transform_edt(~occ, return_indices=True)
    t, xy, r, k = scan_points(s, ks, timing_mode)
    T, ok = pose_at(odom, corr, t)
    w = T.apply(np.column_stack([xy, np.zeros(len(xy))]))[:, :2]
    ix, iy, inside = grid.to_cells(w)
    ok &= inside
    near = grid.cell_centres(nx[iy[ok], ix[ok]], ny[iy[ok], ix[ok]])
    return {"res": np.linalg.norm(w[ok] - near, axis=1), "range": r[ok], "scan": k[ok]}


def map_analysis(s: Scans, tf: BagData, grid: Grid, frame: str, max_scans: int = 800) -> dict:
    odom, corr = laser_poses(tf, frame)
    T, ok = pose_at(odom, corr, s.t)
    xy, yaw = T.p[:, :2], yaw_of(T.R)
    dt = np.gradient(s.t)
    speed = np.linalg.norm(np.gradient(xy, axis=0), axis=1) / dt
    yaw_rate = np.abs(np.gradient(np.unwrap(yaw))) / dt
    path = float(np.sum(np.linalg.norm(np.diff(xy[ok], axis=0), axis=1)))
    ks = np.where(ok)[0]
    ks = ks[np.linspace(0, len(ks) - 1, min(max_scans, len(ks))).astype(int)] if len(ks) else ks

    out: dict = {"path_m": path, "turned_deg": float(np.degrees(np.sum(np.abs(np.diff(np.unwrap(yaw[ok])))))),
                 "max_speed_mps": float(np.percentile(speed[ok], 99)) if ok.any() else None,
                 "max_yaw_rate_dps": float(np.degrees(np.percentile(yaw_rate[ok], 99))) if ok.any() else None,
                 "map": {"size_m": [grid.data.shape[1] * grid.resolution, grid.data.shape[0] * grid.resolution],
                         "resolution_m": grid.resolution,
                         "known_m2": float((grid.data >= 0).sum() * grid.resolution ** 2),
                         "obstacle_m2": float((grid.data > OCCUPIED).sum() * grid.resolution ** 2)}}
    # scan-only odometry drift: how far SLAM had to move odom to keep it on the map
    if corr is not None and len(corr) > 1:
        cy = yaw_of(corr.pose.R)
        drift = float(np.linalg.norm(corr.pose.p[-1, :2] - corr.pose.p[0, :2]))
        out["odometry_drift"] = {"total_m": drift, "per_m_travelled": drift / path if path > 0 else None,
                                 "yaw_deg": float(np.degrees(abs(wrap(cy[-1] - cy[0]))))}
    if not len(ks):
        out["status"] = "no scans with a pose"
        return out

    # which per-point timing lines the scans up best (it matters while turning)
    turning = ks[yaw_rate[ks] > 0.5]
    timing_cmp = {}
    for mode in ("stamp", "first", "last"):
        sub = turning if len(turning) >= 20 else ks
        timing_cmp[mode] = float(np.median(residuals(s, odom, corr, grid, sub, mode)["res"]) * 1e3)
    best = min(timing_cmp, key=timing_cmp.get)
    out["timing_convention"] = {"median_residual_mm_while_turning": timing_cmp, "best": best,
                                "turning_scans": int(len(turning))}

    R = residuals(s, odom, corr, grid, ks, best)
    R0 = residuals(s, odom, corr, grid, ks, "stamp")
    res, rng, k = R["res"], R["range"], R["scan"]
    cell = grid.resolution

    def stats(m, rr=res):
        return {"points": int(m.sum()), "median_mm": float(np.median(rr[m]) * 1e3) if m.any() else None,
                "p90_mm": float(np.percentile(rr[m], 90) * 1e3) if m.any() else None,
                "off_obstacles_pct": float(100 * np.mean(rr[m] > 2 * cell)) if m.any() else None}

    out["residual"] = stats(np.ones(len(res), bool)) | {"note": f"distance to the centre of the nearest obstacle cell (a point on a wall averages ~{cell / 4 * 1e3:.0f} mm from it); off_obstacles = more than 2 cells ({2 * cell * 100:.0f} cm) away, i.e. a point that would mark a wrong cell"}
    out["residual_by_range"] = [{"range_m": [lo, hi]} | stats((rng >= lo) & (rng < hi))
                                for lo, hi in zip(RANGE_BINS[:-1], RANGE_BINS[1:]) if ((rng >= lo) & (rng < hi)).any()]
    yr = np.degrees(yaw_rate[k])
    yr0 = np.degrees(yaw_rate[R0["scan"]])
    rows = []
    for lo, hi in ((0, 20), (20, 60), (60, 120), (120, 240), (240, 1e9)):
        m, m0 = (yr >= lo) & (yr < hi), (yr0 >= lo) & (yr0 < hi)
        if m.any():
            rows.append({"yaw_rate_dps": [lo, hi if hi < 1e9 else None]} | stats(m)
                        | {"median_mm_without_deskew": float(np.median(R0["res"][m0]) * 1e3) if m0.any() else None})
    out["residual_by_yaw_rate"] = rows
    sp = speed[k]
    out["residual_by_speed"] = [{"speed_mps": [lo, hi if hi < 1e9 else None]} | stats((sp >= lo) & (sp < hi))
                                for lo, hi in ((0, 0.05), (0.05, 0.3), (0.3, 0.7), (0.7, 1e9)) if ((sp >= lo) & (sp < hi)).any()]
    out["_track"] = xy[ok]
    return out


# ---------------------------------------------------------------------------- outputs
def write_map(grid: Grid, path: Path):
    """nav2 map_server format: <path>.pgm + <path>.yaml."""
    img = np.full(grid.data.shape, 205, np.uint8)  # unknown
    img[(grid.data >= 0) & (grid.data <= 25)] = 254
    img[grid.data > OCCUPIED] = 0
    img = img[::-1]  # pgm rows run top-down, the grid's bottom-up
    with open(path.with_suffix(".pgm"), "wb") as f:
        f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode())
        f.write(img.tobytes())
    path.with_suffix(".yaml").write_text(
        f"image: {path.with_suffix('.pgm').name}\nmode: trinary\nresolution: {grid.resolution}\n"
        f"origin: [{grid.origin[0]}, {grid.origin[1]}, {grid.origin[2]}]\nnegate: 0\noccupied_thresh: 0.65\n"
        "free_thresh: 0.25\n")


def figures(fig_dir: Path, static: dict, maps: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for name, st in static.items():
        rows = [x for x in st["_directions"] if "range" in x]
        if not rows:
            continue
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(131, projection="polar")
        ax.scatter(np.radians([x["angle_deg"] for x in rows]), [x["range"] for x in rows], s=3)
        ax.set_title("scene (median range)")
        ax = fig.add_subplot(132)
        ax.scatter([x["range"] for x in rows], [x["sigma"] * 1e3 for x in rows], s=4)
        ax.set(xlabel="range (m)", ylabel="temporal std (mm)", title="range noise")
        ax = fig.add_subplot(133)
        ax.plot([x["angle_deg"] for x in st["_directions"]], [100 * x["dropout"] for x in st["_directions"]])
        ax.set(xlabel="angle (deg)", ylabel="dropout (%)", title="missing returns")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{name}.png", dpi=100)
        plt.close(fig)
        made.append(f"{name}.png")
    for name, (grid, m) in maps.items():
        fig, ax = plt.subplots(figsize=(8, 8))
        img = np.where(grid.data < 0, 0.5, 1.0 - grid.data / 100.0)
        h, w = grid.data.shape
        o, res = grid.origin, grid.resolution
        ax.imshow(img, cmap="gray", origin="lower", vmin=0, vmax=1,
                  extent=[o[0], o[0] + w * res, o[1], o[1] + h * res])  # assumes an axis-aligned map (slam_toolbox's is)
        if "_track" in m:
            ax.plot(m["_track"][:, 0], m["_track"][:, 1], "r-", lw=1)
        ax.set(title=f"{name}: map and path", xlabel="x (m)", ylabel="y (m)", aspect="equal")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{name}.png", dpi=110)
        plt.close(fig)
        made.append(f"{name}.png")
    return made


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def _f(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


def report(out: dict, made: list[str], name: str) -> str:
    L = [f"# {name} characterization report", "", "No external truth (no mocap): noise and dropouts come from still scans, "
         "alignment from scans against the map built from them.", ""]
    for run, st in out["static"].items():
        t = out["timing"][run]
        L += [f"## {run} (still)",
              f"- {t['rate_hz']:.2f} Hz (jitter {t['period_jitter_ms']:.1f} ms, {t['gaps']} gaps, longest {t['longest_gap_s']:.2f} s), "
              f"{t['points_per_scan']:.0f} points/scan ({t['points_per_scan_range'][0]}-{t['points_per_scan_range'][1]}), "
              f"beam spacing {t['beam_spacing_deg']:.2f} deg",
              f"- valid returns {100 * t['valid_fraction']:.1f}%; directions missing > 50% of the time: {st['dropout_directions_over_50pct']}/360",
              f"- range noise (temporal std): median {_f(st['sigma_median_mm'])} mm; fit {_f(st['sigma_fit_mm']['at_0m'])} mm + "
              f"{_f(st['sigma_fit_mm']['per_m'])} mm per m",
              f"- outliers (> 5 sigma from a direction's median): {_f(100 * (st['outlier_fraction'] or 0), 2)}%; "
              f"isolated points (likely false obstacles): {_f(100 * (st['isolated_point_fraction'] or 0), 2)}% of returns",
              f"- ranges seen {st['range_seen_m']} m; scan arrival - stamp {t['arrival_minus_stamp_ms']:.1f} ms; "
              f"driver time_increment {_f(t['time_increment_us'])} us",
              "", "| range (m) | directions | std median (mm) | std p90 (mm) |", "|---|---|---|---|"]
        L += [f"| {b['range_m'][0]}-{b['range_m'][1]} | {b['directions']} | {b['sigma_median_mm']:.1f} | {b['sigma_p90_mm']:.1f} |"
              for b in st["sigma_vs_range"]]
        L.append("")
    for run, m in out["maps"].items():
        t = out["timing"][run]
        if "path_m" not in m:
            L += [f"## {run} (pushed around, live SLAM)", f"- {m.get('status')}", ""]
            continue
        L += [f"## {run} (pushed around, live SLAM)",
              f"- {t['duration_s']:.0f} s, {t['rate_hz']:.2f} Hz, path {m['path_m']:.1f} m, turned {m['turned_deg']:.0f} deg; "
              f"p99 speed {_f(m['max_speed_mps'], 2)} m/s, p99 turn rate {_f(m['max_yaw_rate_dps'], 0)} deg/s",
              f"- map {m['map']['size_m'][0]:.1f} x {m['map']['size_m'][1]:.1f} m at {m['map']['resolution_m'] * 100:.0f} cm, "
              f"{m['map']['known_m2']:.1f} m2 seen, {m['map']['obstacle_m2']:.1f} m2 obstacles"]
        if "odometry_drift" in m:
            d = m["odometry_drift"]
            L.append(f"- scan-only odometry (rf2o) drift corrected by SLAM: {d['total_m']:.2f} m, {d['yaw_deg']:.1f} deg "
                     f"({_f(None if d['per_m_travelled'] is None else 100 * d['per_m_travelled'])}% of distance)")
        if "residual" in m:
            tc = m["timing_convention"]
            r = m["residual"]
            L += [f"- point timing: best `{tc['best']}` (median residual while turning, mm: "
                  + ", ".join(f"{k} {v:.1f}" for k, v in tc["median_residual_mm_while_turning"].items()) + ")",
                  f"- scan point to nearest map obstacle: median {_f(r['median_mm'])} mm, p90 {_f(r['p90_mm'])} mm, "
                  f"{_f(r['off_obstacles_pct'])}% of points off the obstacles",
                  "", "| turn rate (deg/s) | points | median (mm) | p90 (mm) | off obstacles (%) | median without per-point timing (mm) |",
                  "|---|---|---|---|---|---|"]
            L += [f"| {x['yaw_rate_dps'][0]}-{x['yaw_rate_dps'][1] or ''} | {x['points']} | {_f(x['median_mm'])} | {_f(x['p90_mm'])} | "
                  f"{_f(x['off_obstacles_pct'])} | {_f(x['median_mm_without_deskew'])} |" for x in m["residual_by_yaw_rate"]]
            L += ["", "| range (m) | points | median (mm) | p90 (mm) | off obstacles (%) |", "|---|---|---|---|---|"]
            L += [f"| {x['range_m'][0]}-{x['range_m'][1]} | {x['points']} | {_f(x['median_mm'])} | {_f(x['p90_mm'])} | "
                  f"{_f(x['off_obstacles_pct'])} |" for x in m["residual_by_range"]]
            L += ["", "| speed (m/s) | points | median (mm) | off obstacles (%) |", "|---|---|---|---|"]
            L += [f"| {x['speed_mps'][0]}-{x['speed_mps'][1] or ''} | {x['points']} | {_f(x['median_mm'])} | {_f(x['off_obstacles_pct'])} |"
                  for x in m["residual_by_speed"]]
        else:
            L.append(f"- {m.get('status', 'no residuals')}")
        L += ["", f"Map for nav2 / reuse: `results/{run}.pgm` + `.yaml`", ""]
    if made:
        L += ["## Figures", ""] + [f"![{f}](figures/{f})" for f in made]
    return "\n".join(L) + "\n"


def analyze(root: Path, cfg: dict) -> dict:
    runs = json.loads((root / "manifest.json").read_text())["runs"]
    res = root / "results"
    res.mkdir(exist_ok=True)
    out: dict = {"timing": {}, "static": {}, "maps": {}}
    grids = {}
    for run in runs:
        bag = root / run["bag"]
        if not (bag / "metadata.yaml").exists():
            print(f"skipping {run['name']}: incomplete bag")
            continue
        scans, tf, grid = read_bag(bag, cfg)
        out["timing"][run["name"]] = timing(scans)
        if run["kind"] == "static":
            out["static"][run["name"]] = static_analysis(scans)
        elif run["kind"] == "map":
            if grid is None:
                out["maps"][run["name"]] = {"status": f"no {cfg['map_topic']} in the bag (SLAM not running?)"}
                continue
            try:
                out["maps"][run["name"]] = map_analysis(scans, tf, grid, cfg["frame"])
            except KeyError as e:
                out["maps"][run["name"]] = {"status": f"no pose: {e}"}
                continue
            grids[run["name"]] = (grid, out["maps"][run["name"]])
            write_map(grid, res / run["name"])
    made = figures(res / "figures", out["static"], grids)
    (res / "results.json").write_text(json.dumps(_jsonable(out), indent=2))
    (res / "report.md").write_text(report(out, made, cfg.get("name", "LD19")))
    return out
