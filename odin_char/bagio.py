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

    def mocap_series(self) -> PoseSeries:
        if len(self.mocap.t_stamp) < 2:
            raise KeyError(f"no mocap data in {self.path}")
        return self.mocap.series(False)


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
    """Read one bag. `cfg` is the `topics`/`frames` part of campaign.yaml."""
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    topics = cfg["topics"]
    frames = cfg["frames"]
    wanted_tf = {topics.get("tf", "/tf"), topics.get("tf_static", "/tf_static")}
    mocap_topic = topics.get("mocap_pose")  # optional PoseStamped alternative to tf
    roi = cfg.get("preflight", {}).get("wall_roi")

    data = BagData(path=str(path))
    typestore = get_typestore(Stores.LATEST)
    with AnyReader([Path(path)], default_typestore=typestore) as reader:
        data.topics = {c.topic: c.msgtype for c in reader.connections}
        data.t_start = reader.start_time * 1e-9
        data.t_end = reader.end_time * 1e-9
        want = [
            c
            for c in reader.connections
            if c.topic
            in wanted_tf
            | {
                topics["odometry"],
                topics["imu"],
                topics.get("cloud", ""),
                topics.get("markers", ""),
                topics.get("rosout", "/rosout"),
                mocap_topic or "",
            }
        ]
        for conn, t_ns, raw in reader.messages(connections=want):
            t_recv = t_ns * 1e-9
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic in wanted_tf:
                for tr in msg.transforms:
                    key = (tr.header.frame_id.lstrip("/"), tr.child_frame_id.lstrip("/"))
                    target = (
                        data.mocap
                        if key == (frames["mocap_world"], frames["mocap_body"])
                        else data.tf[key]
                    )
                    target.t_stamp.append(_stamp(tr.header))
                    target.t_recv.append(t_recv)
                    target.p.append(_xyz(tr.transform.translation))
                    target.q.append(_xyzw(tr.transform.rotation))
            elif conn.topic == mocap_topic:
                data.mocap.t_stamp.append(_stamp(msg.header))
                data.mocap.t_recv.append(t_recv)
                data.mocap.p.append(_xyz(msg.pose.position))
                data.mocap.q.append(_xyzw(msg.pose.orientation))
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
                n = len(getattr(msg, "markers", getattr(msg, "poses", [])))
                data.marker_n.append(n)
            elif conn.topic == topics.get("rosout", "/rosout"):
                data.log_t.append(t_recv)
                data.log_msg.append(getattr(msg, "msg", ""))
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
