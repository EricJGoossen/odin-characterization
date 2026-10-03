"""Extract the streams the analysis needs from ROS1/ROS2 bags (via `rosbags`, no ROS install).

Every stream keeps both the header stamp and the bag (arrival) time so latency can be evaluated
either stamp-to-truth or arrival-to-truth.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .se3 import Pose, PoseSeries


def _stamp(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


@dataclass
class TfStream:
    t_stamp: list = field(default_factory=list)
    t_recv: list = field(default_factory=list)
    p: list = field(default_factory=list)
    q: list = field(default_factory=list)

    def series(self, use_arrival: bool = False) -> PoseSeries:
        t = np.array(self.t_recv if use_arrival else self.t_stamp)
        return PoseSeries(t, Pose.from_quat(np.array(self.p), np.array(self.q)))


@dataclass
class BagData:
    path: str
    tf: dict = field(default_factory=lambda: defaultdict(TfStream))  # (parent, child) -> stream
    odom: TfStream = field(default_factory=TfStream)
    mocap: TfStream = field(default_factory=TfStream)
    imu_t: list = field(default_factory=list)
    imu_gyro: list = field(default_factory=list)
    cloud_t: list = field(default_factory=list)
    cloud_n: list = field(default_factory=list)
    cloud_plane_std: list = field(default_factory=list)
    cloud_field_mean: dict = field(default_factory=lambda: defaultdict(list))
    marker_t: list = field(default_factory=list)
    marker_n: list = field(default_factory=list)
    log_t: list = field(default_factory=list)
    log_msg: list = field(default_factory=list)
    topics: dict = field(default_factory=dict)
    t_start: float = 0.0
    t_end: float = 0.0

    def tf_pairs(self) -> list[tuple[str, str]]:
        return sorted(self.tf.keys())

    def tf_series(self, parent: str, child: str, use_arrival: bool = False) -> PoseSeries:
        if (parent, child) in self.tf and len(self.tf[(parent, child)].t_stamp) > 1:
            return self.tf[(parent, child)].series(use_arrival)
        if (child, parent) in self.tf and len(self.tf[(child, parent)].t_stamp) > 1:
            s = self.tf[(child, parent)].series(use_arrival)
            return PoseSeries(s.t, s.pose.inv())
        raise KeyError(
            f"tf {parent}->{child} not found in {self.path}. Available: {self.tf_pairs()}"
        )

    mocap_clock: str = "host"  # "server": mocap stamps come from the mocap server's clock
    mocap_clock_model: dict = field(default_factory=dict)

    def mocap_series(self) -> PoseSeries:
        """Mocap poses on the host clock.

        With mocap_clock == "server" the stamps are mapped to the host clock with a per-bag
        offset + drift model fitted to the lower envelope of (arrival - stamp): the minimum
        network delay is (nearly) constant, so the envelope tracks the clock offset. The leftover
        constant (minimum delay) is absorbed by the Phase 1 IMU/mocap time offset."""
        if len(self.mocap.t_stamp) < 2:
            raise KeyError(f"no mocap data in {self.path}")
        s = self.mocap.series(False)
        if self.mocap_clock != "server":
            return s
        a, b, t0 = self.fit_mocap_clock()
        return PoseSeries(s.t + a + b * (s.t - t0), s.pose)

    def fit_mocap_clock(self, window_s: float = 5.0) -> tuple[float, float, float]:
        ts = np.asarray(self.mocap.t_stamp)
        d = np.asarray(self.mocap.t_recv) - ts
        t0 = float(ts[0])
        w = np.floor((ts - t0) / window_s).astype(int)
        idx = [np.where(w == k)[0] for k in np.unique(w)]
        mins = np.array([[ts[i[np.argmin(d[i])]] - t0, d[i].min()] for i in idx if len(i)])
        if len(mins) >= 3:
            b, a = np.polyfit(mins[:, 0], mins[:, 1], 1)
            # one robust pass: drop windows whose minimum sits well above the line (congestion)
            r = mins[:, 1] - (a + b * mins[:, 0])
            keep = r < np.median(r) + 3 * (np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-4)
            b, a = np.polyfit(mins[keep, 0], mins[keep, 1], 1)
        else:
            a, b = float(d.min()), 0.0
        resid = d - (a + b * (ts - t0))
        self.mocap_clock_model = {"offset_s": float(a), "drift_ppm": float(b * 1e6),
                                  "delay_median_ms": float(np.median(resid) * 1e3),
                                  "delay_p99_ms": float(np.percentile(resid, 99) * 1e3)}
        return float(a), float(b), t0


def _xyz(v):
    return [v.x, v.y, v.z]


def _xyzw(q):
    return [q.x, q.y, q.z, q.w]


def _cloud_stats(msg, roi, data: BagData):
    names = [f.name for f in msg.fields]
    offsets = {f.name: f.offset for f in msg.fields}
    n = msg.width * msg.height
    data.cloud_n.append(n)
    if n == 0 or not {"x", "y", "z"} <= set(names):
        data.cloud_plane_std.append(np.nan)
        return
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n, msg.point_step)
    xyz = np.stack(
        [buf[:, offsets[k] : offsets[k] + 4].copy().view("<f4")[:, 0] for k in "xyz"], axis=1
    )
    for name in names:
        if name in ("x", "y", "z"):
            continue
        f = next(f for f in msg.fields if f.name == name)
        if f.datatype == 7:  # FLOAT32
            vals = buf[:, f.offset : f.offset + 4].copy().view("<f4")[:, 0]
            data.cloud_field_mean[name].append(float(np.nanmean(vals)))
    if roi is None:
        data.cloud_plane_std.append(np.nan)
        return
    lo, hi = np.asarray(roi[0]), np.asarray(roi[1])
    pts = xyz[np.all((xyz >= lo) & (xyz <= hi), axis=1)]
    if len(pts) < 20:
        data.cloud_plane_std.append(np.nan)
        return
    c = pts - pts.mean(0)
    data.cloud_plane_std.append(float(np.linalg.svd(c, full_matrices=False)[1][-1] / np.sqrt(len(pts))))


def read_bag(path: str | Path, cfg: dict) -> BagData:
    """Read one bag. `cfg` is campaign.yaml (uses its `topics`, `mocap` and `preflight` parts)."""
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    from .mocap_config import MocapConfig

    topics = cfg["topics"]
    mocap = MocapConfig.from_cfg(cfg)
    wanted_tf = {topics.get("tf", "/tf"), topics.get("tf_static", "/tf_static")}
    roi = cfg.get("preflight", {}).get("wall_roi")

    data = BagData(path=str(path))
    data.mocap_clock = mocap.clock
    typestore = get_typestore(Stores.LATEST)
    with AnyReader([Path(path)], default_typestore=typestore) as reader:
        data.topics = {c.topic: c.msgtype for c in reader.connections}
        data.t_start = reader.start_time * 1e-9
        data.t_end = reader.end_time * 1e-9
        names = wanted_tf | {mocap.topic, topics["odometry"], topics["imu"], topics.get("cloud") or "",
                             topics.get("markers") or "", topics.get("rosout", "/rosout")}
        want = [c for c in reader.connections if c.topic in names]
        for conn, t_ns, raw in reader.messages(connections=want):
            t_recv = t_ns * 1e-9
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == mocap.topic:
                for st, p, q in mocap.extract(msg):
                    data.mocap.t_stamp.append(st)
                    data.mocap.t_recv.append(t_recv)
                    data.mocap.p.append(p)
                    data.mocap.q.append(q)
            if conn.topic in wanted_tf:
                for tr in msg.transforms:
                    key = (tr.header.frame_id.lstrip("/"), tr.child_frame_id.lstrip("/"))
                    s = data.tf[key]
                    s.t_stamp.append(_stamp(tr.header))
                    s.t_recv.append(t_recv)
                    s.p.append(_xyz(tr.transform.translation))
                    s.q.append(_xyzw(tr.transform.rotation))
            elif conn.topic == mocap.topic:
                continue
            elif conn.topic == topics["odometry"]:
                data.odom.t_stamp.append(_stamp(msg.header))
                data.odom.t_recv.append(t_recv)
                data.odom.p.append(_xyz(msg.pose.pose.position))
                data.odom.q.append(_xyzw(msg.pose.pose.orientation))
            elif conn.topic == topics["imu"]:
                data.imu_t.append(_stamp(msg.header))
                data.imu_gyro.append(_xyz(msg.angular_velocity))
            elif conn.topic == topics.get("cloud"):
                data.cloud_t.append(_stamp(msg.header))
                _cloud_stats(msg, roi, data)
            elif conn.topic == topics.get("markers"):
                data.marker_t.append(t_recv)
                data.marker_n.append(len(getattr(msg, "markers", getattr(msg, "poses", []))))
            elif conn.topic == topics.get("rosout", "/rosout"):
                data.log_t.append(t_recv)
                data.log_msg.append(getattr(msg, "msg", ""))
    if len(data.mocap.t_stamp) == 0:
        print(f"WARNING: no mocap samples in {path} for {mocap}. Topics in bag: {sorted(data.topics)}")
    return data


def list_bag(path: str | Path) -> str:
    """Human-readable summary of topics and tf pairs (for configuring frames)."""
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    lines = [f"bag: {path}"]
    pairs: dict[tuple[str, str], int] = defaultdict(int)
    with AnyReader([Path(path)], default_typestore=get_typestore(Stores.LATEST)) as reader:
        for c in reader.connections:
            lines.append(f"  {c.topic:40s} {c.msgtype:35s} {c.msgcount}")
        tfc = [c for c in reader.connections if c.topic in ("/tf", "/tf_static")]
        for conn, _, raw in reader.messages(connections=tfc):
            for tr in reader.deserialize(raw, conn.msgtype).transforms:
                pairs[(tr.header.frame_id, tr.child_frame_id)] += 1
    lines.append("  tf pairs (parent -> child : count):")
    lines += [f"    {p} -> {c} : {n}" for (p, c), n in sorted(pairs.items())]
    return "\n".join(lines)
