"""LD19 analysis on synthetic rosbag2 recordings of a known room (no ROS needed).

Static: ray-traced scans with known range noise and dropouts. Map: the lidar drives and spins
through the room with beams timed the ROS way (stamp = first beam), perfect odometry on tf and
the true occupancy grid on /map; the analysis must line the points up with it and find the timing.
"""
import json

import numpy as np
import pytest
import yaml
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from odin_char.sensors import load_sensor
from odin_char.sensors.ld19.__main__ import load_config
from odin_char.sensors.ld19.analyze import analyze

TS = get_typestore(Stores.ROS2_HUMBLE)
T = TS.types
N_BEAMS, RATE = 450, 10.0
SIGMA0, SIGMA_PER_M, DROPOUT = 0.006, 0.003, 0.05
# room walls and a box obstacle, as segments (x0, y0, x1, y1)
WALLS = [(-3, -2, 4, -2), (4, -2, 4, 2.5), (4, 2.5, -3, 2.5), (-3, 2.5, -3, -2),
         (1.0, 0.5, 1.6, 0.5), (1.6, 0.5, 1.6, 1.1), (1.6, 1.1, 1.0, 1.1), (1.0, 1.1, 1.0, 0.5)]


def raycast(x, y, ang):
    """Range from (x, y) along world angles `ang` to the nearest wall."""
    d = np.stack([np.cos(ang), np.sin(ang)], 1)
    best = np.full(len(ang), np.inf)
    for x0, y0, x1, y1 in WALLS:
        e = np.array([x1 - x0, y1 - y0])
        den = d[:, 0] * e[1] - d[:, 1] * e[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((x0 - x) * e[1] - (y0 - y) * e[0]) / den
            u = ((x0 - x) * d[:, 1] - (y0 - y) * d[:, 0]) / den
        hit = (np.abs(den) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
        best = np.where(hit & (t < best), t, best)
    return best


def _time(t):
    sec, nsec = divmod(int(round(t * 1e9)), 1_000_000_000)
    return T["builtin_interfaces/msg/Time"](sec=sec, nanosec=nsec)


def _hdr(t, frame):
    return T["std_msgs/msg/Header"](stamp=_time(t), frame_id=frame)


def _scan(t, ranges):
    return T["sensor_msgs/msg/LaserScan"](
        header=_hdr(t, "base_laser"), angle_min=0.0, angle_max=float(2 * np.pi * (N_BEAMS - 1) / N_BEAMS),
        angle_increment=float(2 * np.pi / N_BEAMS), time_increment=float(1 / RATE / N_BEAMS), scan_time=float(1 / RATE),
        range_min=0.02, range_max=12.0, ranges=ranges.astype(np.float32),
        intensities=np.full(N_BEAMS, 200.0, np.float32))


def _tf(t, parent, child, x, y, yaw):
    return T["geometry_msgs/msg/TransformStamped"](
        header=_hdr(t, parent), child_frame_id=child,
        transform=T["geometry_msgs/msg/Transform"](
            translation=T["geometry_msgs/msg/Vector3"](x=float(x), y=float(y), z=0.0),
            rotation=T["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=float(np.sin(yaw / 2)), w=float(np.cos(yaw / 2)))))


def noisy(rng, r):
    out = r + rng.normal(0, SIGMA0 + SIGMA_PER_M * r)
    return np.where(rng.random(len(r)) < DROPOUT, 0.0, out)  # the driver reports a missed return as 0


def write(path, msgs):
    msgs.sort(key=lambda m: m[0])
    with Writer(path, version=9) as w:
        conns = {}
        for t, topic, msg, mt in msgs:
            if topic not in conns:
                conns[topic] = w.add_connection(topic, mt, typestore=TS)
            w.write(conns[topic], int((t + 0.004) * 1e9), TS.serialize_cdr(msg, mt))


def true_grid(res=0.05, lo=(-3.5, -2.5), hi=(4.5, 3.0)):
    w, h = int((hi[0] - lo[0]) / res), int((hi[1] - lo[1]) / res)
    data = np.zeros((h, w), np.int8)
    for x0, y0, x1, y1 in WALLS:
        s = np.linspace(0, 1, 400)
        ix = ((x0 + s * (x1 - x0) - lo[0]) / res).astype(int)
        iy = ((y0 + s * (y1 - y0) - lo[1]) / res).astype(int)
        data[iy, ix] = 100
    info = T["nav_msgs/msg/MapMetaData"](
        map_load_time=_time(0), resolution=res, width=w, height=h,
        origin=T["geometry_msgs/msg/Pose"](position=T["geometry_msgs/msg/Point"](x=lo[0], y=lo[1], z=0.0),
                                          orientation=T["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0)))
    return data, info


def trajectory(t):
    """Drive a loop, with two fast spins in place (~4 rad/s) on the way."""
    x = 0.5 + 1.6 * np.cos(0.25 * t)
    y = -0.6 + 1.0 * np.sin(0.25 * t)
    yaw = 0.25 * t + np.pi / 2
    spin = np.clip(t - 8, 0, 3) * 4.0 + np.clip(t - 18, 0, 2) * -4.0
    return x, y, yaw + spin


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp("ld19")
    rng = np.random.default_rng(4)
    cfg = load_config(None)
    (root / "ld19.yaml").write_text(yaml.safe_dump(cfg))
    beams = np.arange(N_BEAMS) * 2 * np.pi / N_BEAMS
    t0 = 1000.0
    # static, 30 s
    msgs = [(t0 + k / RATE, "/scan", _scan(t0 + k / RATE, noisy(rng, raycast(-0.5, 0.3, beams + 0.4))),
             "sensor_msgs/msg/LaserScan") for k in range(int(30 * RATE))]
    write(root / "bags" / "static_01", msgs)
    # map run, 30 s: beam i at t_k + i * time_increment (stamp = first beam)
    t0 = 2000.0
    msgs = []
    inc = 1 / RATE / N_BEAMS
    for k in range(int(30 * RATE)):
        tk = k / RATE
        tb = tk + np.arange(N_BEAMS) * inc
        x, y, yaw = trajectory(tb)
        r = noisy(rng, raycast(x, y, yaw + beams))
        msgs.append((t0 + tk, "/scan", _scan(t0 + tk, r), "sensor_msgs/msg/LaserScan"))
    for tt in np.arange(0, 30, 0.01):  # odometry tf at 100 Hz
        x, y, yaw = trajectory(np.array([tt]))
        msgs.append((t0 + tt, "/tf", T["tf2_msgs/msg/TFMessage"](transforms=[_tf(t0 + tt, "odom", "base_laser", x[0], y[0], yaw[0])]),
                     "tf2_msgs/msg/TFMessage"))
        if round(tt * 100) % 5 == 0:
            msgs.append((t0 + tt, "/tf", T["tf2_msgs/msg/TFMessage"](transforms=[_tf(t0 + tt, "map", "odom", 0, 0, 0)]),
                         "tf2_msgs/msg/TFMessage"))
    data, info = true_grid()
    msgs.append((t0 + 29.5, "/map", T["nav_msgs/msg/OccupancyGrid"](header=_hdr(t0 + 29.5, "map"), info=info,
                                                                   data=data.ravel()), "nav_msgs/msg/OccupancyGrid"))
    write(root / "bags" / "map_01", msgs)
    (root / "manifest.json").write_text(json.dumps({"runs": [
        {"name": "static_01", "kind": "static", "bag": "bags/static_01"},
        {"name": "map_01", "kind": "map", "bag": "bags/map_01"}]}))
    return root, analyze(root, cfg)


def test_static(campaign):
    _, out = campaign
    t, st = out["timing"]["static_01"], out["static"]["static_01"]
    assert t["rate_hz"] == pytest.approx(RATE, rel=0.01) and t["points_per_scan"] == N_BEAMS
    assert t["valid_fraction"] == pytest.approx(1 - DROPOUT, abs=0.01)
    assert st["dropout_overall"] == pytest.approx(DROPOUT, abs=0.01)
    # noise model recovered (std of a direction also includes the range change across its 1 deg bin)
    assert st["sigma_fit_mm"]["per_m"] == pytest.approx(SIGMA_PER_M * 1e3, abs=1.5)
    assert st["sigma_fit_mm"]["at_0m"] == pytest.approx(SIGMA0 * 1e3, abs=3.0)
    assert st["range_seen_m"][1] < 5.0


def test_map(campaign):
    root, out = campaign
    m = out["maps"]["map_01"]
    assert m["timing_convention"]["best"] == "first"
    tc = m["timing_convention"]["median_residual_mm_while_turning"]
    assert tc["stamp"] > 2 * tc["first"]  # ignoring per-beam timing smears the spins
    # the synthetic walls lie on cell edges, so a perfect point sits half a cell (25 mm) from the cell centre
    assert m["residual"]["median_mm"] < 35 and m["residual"]["off_obstacles_pct"] < 1
    fast = [x for x in m["residual_by_yaw_rate"] if x["yaw_rate_dps"][0] >= 120]
    assert fast and fast[0]["median_mm_without_deskew"] > fast[0]["median_mm"]
    assert m["odometry_drift"]["total_m"] == pytest.approx(0.0, abs=1e-6)
    assert (root / "results" / "map_01.pgm").exists() and (root / "results" / "map_01.yaml").exists()
    assert "LD19 characterization report" in (root / "results" / "report.md").read_text()


def test_ld19_is_not_a_campaign_sensor():
    with pytest.raises(ValueError, match="not a pose sensor"):
        load_sensor({"sensor": {"type": "ld19"}})
