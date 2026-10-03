"""Synthetic Odin + mocap campaign with known Tier 0 parameters, written as real rosbag2 files.

Used to check the analysis end to end (bag parsing, the odom->map inversion, calibration,
latency, static field, FOGM, events, validation).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

from odin_char.se3 import Pose, PoseSeries, rot_z, so3_exp, yaw_of

TS = get_typestore(Stores.ROS2_HUMBLE)
T = TS.types


@dataclass
class Truth:
    time_offset: float = 0.0137  # mocap_time = odin_stamp + offset
    X: np.ndarray = field(default_factory=lambda: np.array([0.12, -0.03, 0.35, 0.015, -0.03, 0.52]))
    T_WM: np.ndarray = field(default_factory=lambda: np.array([1.3, -0.7, 0.02, 0.0, 0.0, 0.8]))
    tau_d: float = 0.045
    arrival_delay: float = 0.012
    sigma: np.ndarray = field(default_factory=lambda: np.array([0.02, 0.015, 0.01, 0.003, 0.003, 0.006]))
    tau_c: np.ndarray = field(default_factory=lambda: np.array([1.5, 1.0, 2.0, 0.8, 0.8, 1.2]))
    nu: np.ndarray = field(default_factory=lambda: np.array([4.0, np.inf, np.inf, np.inf, np.inf, 5.0]))
    stationary_scale: float = 0.4
    mocap_clock_offset: float = -3.71  # mocap server clock minus host clock
    mocap_clock_drift: float = 30e-6
    rate: float = 10.0


def static_field(pM: np.ndarray) -> np.ndarray:
    x, y = pM[:, 0], pM[:, 1]
    out = np.zeros((len(pM), 6))
    out[:, 0] = 0.03 * np.sin(0.9 * x) * np.cos(0.8 * y)
    out[:, 1] = 0.025 * np.cos(0.7 * x + 0.3)
    out[:, 5] = 0.01 * np.sin(0.6 * y)
    return out


def _t_unit(rng, nu, n):
    if not np.isfinite(nu):
        return rng.standard_normal(n)
    return rng.standard_t(nu, n) / np.sqrt(nu / (nu - 2))


def fogm(rng, tr: Truth, n: int, scale: float = 1.0) -> np.ndarray:
    dt = 1 / tr.rate
    out = np.zeros((n, 6))
    for a in range(6):
        phi = np.exp(-dt / tr.tau_c[a])
        s = tr.sigma[a] * scale
        e = _t_unit(rng, tr.nu[a], n) * s * np.sqrt(1 - phi**2)
        x = s * rng.standard_normal()
        for k in range(n):
            x = phi * x + e[k]
            out[k, a] = x
    return out


# ---------------------------------------------------------------------------- trajectories
def figure8(speed: float, laps: int, A=3.0, B=1.8, c=(0.0, 0.0), rate=200.0, lead=2.0):
    s = np.linspace(0, 2 * np.pi, 4001)
    xy = np.stack([A * np.sin(s), B * np.sin(s) * np.cos(s)], 1)
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    lap_T = L[-1] / speed
    t = np.arange(0, laps * lap_T + 2 * lead, 1 / rate)
    tau = np.clip(t - lead, 0, laps * lap_T)
    dist = (tau * speed) % L[-1]
    x = np.interp(dist, L, xy[:, 0]) + c[0]
    y = np.interp(dist, L, xy[:, 1]) + c[1]
    dx, dy = np.gradient(x), np.gradient(y)
    yaw = np.unwrap(np.arctan2(dy, dx))
    yaw[t < lead] = yaw[np.argmax(t >= lead)]
    yaw[t > lead + laps * lap_T] = yaw[np.argmax(t > lead + laps * lap_T) - 1]
    return t, np.stack([x, y, np.full_like(x, 0.1)], 1), yaw


def calib_motion(duration=120.0, rate=200.0):
    t = np.arange(0, duration, 1 / rate)
    yaw_rate = 2.5 * np.sin(2 * np.pi * (0.05 + 0.004 * t) * t)
    yaw = np.cumsum(yaw_rate) / rate
    x = 0.6 * np.sin(2 * np.pi * 0.11 * t)
    y = 0.5 * np.sin(2 * np.pi * 0.07 * t + 1.0)
    return t, np.stack([x, y, 0.1 + 0.005 * np.sin(t)], 1), yaw


def stationary(duration, p, yaw, rate=200.0):
    t = np.arange(0, duration, 1 / rate)
    return t, np.tile(np.array([p[0], p[1], 0.1]), (len(t), 1)), np.full(len(t), yaw)


def sinusoid(rate=200.0):
    t = np.arange(0, 40, 1 / rate)
    x = np.where(t < 20, 0.15 * np.sin(2 * np.pi * 1.0 * t), 0.0)
    yaw = np.where(t >= 20, 0.6 * np.sin(2 * np.pi * 0.7 * (t - 20)), 0.0)
    return t, np.stack([x, np.zeros_like(x), np.full_like(x, 0.1)], 1), yaw


# ---------------------------------------------------------------------------- bag writing
def _time(t):
    sec, nsec = divmod(int(round(t * 1e9)), 1_000_000_000)
    return T["builtin_interfaces/msg/Time"](sec=sec, nanosec=nsec)


def _tf(t, parent, child, pose: Pose):
    q = Rotation.from_matrix(pose.R).as_quat()
    return T["geometry_msgs/msg/TransformStamped"](
        header=T["std_msgs/msg/Header"](stamp=_time(t), frame_id=parent),
        child_frame_id=child,
        transform=T["geometry_msgs/msg/Transform"](
            translation=T["geometry_msgs/msg/Vector3"](x=pose.p[0], y=pose.p[1], z=pose.p[2]),
            rotation=T["geometry_msgs/msg/Quaternion"](x=q[0], y=q[1], z=q[2], w=q[3]),
        ),
    )


def write_bag(path: Path, streams: list[tuple[float, float, str, object, str]]):
    """streams: (recv_time, sort_key, topic, msg, msgtype)"""
    streams.sort(key=lambda s: s[0])
    with Writer(path, version=9) as w:
        conns = {}
        for recv, _, topic, msg, mt in streams:
            if topic not in conns:
                conns[topic] = w.add_connection(topic, mt, typestore=TS)
            w.write(conns[topic], int(recv * 1e9), TS.serialize_cdr(msg, mt))


@dataclass
class RunSpec:
    name: str
    phase: str
    t: np.ndarray
    p: np.ndarray
    yaw: np.ndarray
    session: str | None = None
    speed_class: str | None = None
    location: str | None = None
    noise_scale: float = 1.0
    map_start: float = 0.0  # seconds after bag start when odom->map first appears
    cover: tuple | None = None  # (start, end) in run time: no map updates, error drifts
    start: float = 1_000.0


def simulate_run(spec: RunSpec, tr: Truth, rng, path: Path, frames: dict):
    X = Pose.from_vec(tr.X)
    T_WM = Pose.from_vec(tr.T_WM)
    t_true = spec.t + spec.start
    R = rot_z(spec.yaw)
    tilt = so3_exp(np.stack([0.002 * np.sin(0.3 * spec.t), 0.002 * np.cos(0.2 * spec.t), 0 * spec.t], 1))
    K = PoseSeries(t_true, Pose(R @ tilt, spec.p))
    streams = []
    # mocap 120 Hz on the host clock, with a few dropouts
    tm = np.arange(t_true[0], t_true[-1], 1 / 120)
    drop = np.zeros(len(tm), bool)
    for d0 in rng.uniform(tm[0], tm[-1], 3):
        drop |= (tm > d0) & (tm < d0 + 0.05)
    Km, _ = K.interp(tm[~drop])
    # mocap stamps come from the mocap server's clock (offset + drift); arrival has network jitter
    for ti, i in zip(tm[~drop], range(len(Km))):
        noisy = Pose(Km.R[i], Km.p[i] + rng.normal(0, 0.0003, 3))
        server_t = ti + tr.mocap_clock_offset + tr.mocap_clock_drift * (ti - 1000.0)
        recv = ti + 0.0008 + rng.exponential(0.002)
        streams.append((recv, 0, "/tf", T["tf2_msgs/msg/TFMessage"](transforms=[_tf(server_t, frames["mocap_world"], frames["mocap_body"], noisy)]), "tf2_msgs/msg/TFMessage"))

    # IMU (200 Hz, Odin clock)
    ti = np.arange(t_true[0] + 0.05, t_true[-1] - 0.05, 1 / 200)
    I = PoseSeries(t_true, K.pose @ X)
    w = I.body_rates(ti, 0.005) + rng.normal(0, 0.003, (len(ti), 3))
    for k in range(len(ti)):
        msg = T["sensor_msgs/msg/Imu"](
            header=T["std_msgs/msg/Header"](stamp=_time(ti[k] - tr.time_offset), frame_id="imu"),
            orientation=T["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0),
            orientation_covariance=np.zeros(9),
            angular_velocity=T["geometry_msgs/msg/Vector3"](x=w[k, 0], y=w[k, 1], z=w[k, 2]),
            angular_velocity_covariance=np.zeros(9),
            linear_acceleration=T["geometry_msgs/msg/Vector3"](x=0.0, y=0.0, z=9.81),
            linear_acceleration_covariance=np.zeros(9),
        )
        streams.append((ti[k] + 0.002, 0, "/odin1/imu", msg, "sensor_msgs/msg/Imu"))

    # Odin estimate at 10 Hz: T_MI_hat(t) = perturbed truth at t - tau_d
    te = np.arange(t_true[0] + 0.5, t_true[-1] - 0.5, 1 / tr.rate)
    Ttrue, _ = I.interp(te - tr.tau_d)
    true_M = T_WM.inv() @ Ttrue
    v = np.gradient(true_M.p, axis=0) * tr.rate
    speed = np.linalg.norm(v[:, :2], axis=1)
    heading = np.where(speed > 0.05, np.arctan2(v[:, 1], v[:, 0]), yaw_of(true_M.R))
    eps = fogm(rng, tr, len(te), spec.noise_scale) + static_field(true_M.p)
    if spec.cover:
        c0, c1 = spec.cover[0] + t_true[0], spec.cover[1] + t_true[0]
        inside = (te > c0) & (te < c1)
        eps[inside, 0] += 0.15 * (te[inside] - c0)
    Rh = rot_z(heading)
    dp = (Rh @ eps[:, :3, None])[..., 0]
    dth = (Rh @ eps[:, 3:, None])[..., 0]
    est_M = Pose(so3_exp(dth) @ true_M.R, true_M.p + dp)
    # odometry drifts slowly; odom->map absorbs it (Odin convention: parent odom, child map = T_OM)
    drift = Pose.from_vec(np.stack([0.001 * (te - te[0]), 0.0005 * (te - te[0]), 0 * te, 0 * te, 0 * te, 0.0001 * (te - te[0])], 1))
    Wodo = Pose.from_vec(np.array([0.3, 0.2, 0.0, 0, 0, 0.4]))
    # odom->imu is smooth LIO output (lagged like everything Odin publishes); odom->map carries
    # the relocalization error so that T_OM^-1 T_OI == est_M exactly
    smooth = Pose(Ttrue.R, Ttrue.p + rng.normal(0, 0.0005, Ttrue.p.shape))
    T_OI = drift @ (Wodo @ smooth)
    T_OM = T_OI @ est_M.inv()
    stamp = te - tr.time_offset
    for k in range(len(te)):
        recv = te[k] + tr.arrival_delay
        tfs = [_tf(stamp[k], frames["odin_odom"], frames["odin_imu"], T_OI[k])]
        in_cover = spec.cover is not None and (spec.cover[0] + t_true[0]) < te[k] < (spec.cover[1] + t_true[0])
        if te[k] >= t_true[0] + spec.map_start and not in_cover:
            tfs.append(_tf(stamp[k], frames["odin_odom"], frames["odin_map"], T_OM[k]))
        streams.append((recv, 0, "/tf", T["tf2_msgs/msg/TFMessage"](transforms=tfs), "tf2_msgs/msg/TFMessage"))
    write_bag(path, streams)


def make_campaign(root: Path, seed: int = 1, laps: int = 12, short: bool = False) -> Truth:
    rng = np.random.default_rng(seed)
    tr = Truth()
    cfg = yaml.safe_load((Path(__file__).parents[1] / "config" / "campaign.yaml").read_text())
    cfg["analysis"]["mc_realizations"] = 20 if short else 50
    cfg["hand_eye_prior"] = {"X": [0.1, 0.0, 0.33, 0.0, 0.0, 0.5], "prior_sigma_t": 0.05, "prior_sigma_rot_deg": 10}
    frames = {**cfg["frames"], "mocap_world": cfg["mocap"]["parent_frame"], "mocap_body": cfg["mocap"]["child_frame"]}
    root.mkdir(parents=True, exist_ok=True)
    (root / "campaign.yaml").write_text(yaml.safe_dump(cfg))
    specs: list[RunSpec] = []
    start = 1000.0
    specs.append(RunSpec("calib", "calib", *calib_motion(), start=start))
    for s in ("s1", "s2", "s3"):
        for sc, sp, n in (("1x", 0.8, laps), ("0.5x", 0.4, 2), ("1.5x", 1.2, 2)):
            specs.append(RunSpec(f"{s}_{sc}", "phase2", *figure8(sp, n), session=s, speed_class=sc))
    for i, (p, yaw) in enumerate((((2.0, 0.5), 0.3), ((-2.0, -0.4), 2.0), ((0.0, 1.2), -1.0))):
        specs.append(RunSpec(f"hold{i}", "phase3", *stationary(60 if short else 150, p, yaw), location=f"L{i}", noise_scale=tr.stationary_scale))
    specs.append(RunSpec("sine", "phase4", *sinusoid()))
    init_trials = []
    for i, (off, hd, mv) in enumerate([(0, 0, False), (0.5, 10, True), (1, 30, False), (2, 0, True), (0.5, 0, False), (1, 10, True)]):
        delay = rng.exponential(4.0) + 1.0
        specs.append(RunSpec(f"init{i}", "init", *stationary(15, (off, 0.0), np.radians(hd)), map_start=delay))
        init_trials.append({"run": None, "offset_m": off, "heading_deg": hd, "moving": mv, "t_launch": None, "timeout": 30.0, "t_log_success": None, "_delay": delay})
    specs.append(RunSpec("cover3", "outage", *figure8(0.3, 1), cover=(8.0, 11.0)))

    runs, trial_i = [], 0
    for i, spec in enumerate(specs):
        spec.start = start
        bag = root / "bags" / spec.name
        simulate_run(spec, tr, rng, bag, frames)
        runs.append({"id": i, "name": spec.name, "phase": spec.phase, "bag": f"bags/{spec.name}", "session": spec.session,
                     "speed_class": spec.speed_class, "location": spec.location})
        if spec.phase == "init":
            it = init_trials[trial_i]
            it.update(run=i, t_launch=start + 0.0)
            trial_i += 1
        start += spec.t[-1] + 30
    outages = [{"run": runs[-1]["id"], "cover_s": 3.0, "t_cover": runs and specs[-1].start + 8.0, "t_uncover": specs[-1].start + 11.0}]
    for it in init_trials:
        it.pop("_delay")
    (root / "manifest.json").write_text(json.dumps({"runs": runs, "init_trials": init_trials, "outages": outages}, indent=1))
    return tr
