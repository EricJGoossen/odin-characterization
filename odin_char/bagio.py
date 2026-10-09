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
    cloud_valid: list = field(default_factory=list)  # valid (finite, non-zero) points per cloud
    cloud_range_mean: list = field(default_factory=list)
    px_sum: np.ndarray | None = None  # per-pixel range sums over the bag (organized clouds)
    px_sumsq: np.ndarray | None = None
    px_cnt: np.ndarray | None = None

    def pixel_range_std(self, min_count: int = 20) -> np.ndarray:
        """Temporal range std of every pixel seen in >= min_count clouds (static-scene noise)."""
        if self.px_cnt is None:
            return np.zeros(0)
        ok = self.px_cnt >= min_count
        m = self.px_sum[ok] / self.px_cnt[ok]
        return np.sqrt(np.maximum(self.px_sumsq[ok] / self.px_cnt[ok] - m * m, 0.0))
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
    mocap_clean: bool = True  # drop re-published frames and back-date burst deliveries
    mocap_clean_stats: dict = field(default_factory=dict)
    mocap_lowpass_hz: float | None = None  # zero-phase truth low-pass (set from analysis.truth_lowpass_hz)

    def mocap_series(self, lowpass: bool = True) -> PoseSeries:
        """Mocap poses on the host clock.

        With mocap_clock == "server" the stamps are mapped to the host clock with a per-bag
        offset + drift model fitted to the lower envelope of (arrival - stamp): the minimum
        network delay is (nearly) constant, so the envelope tracks the clock offset. The leftover
        constant (minimum delay) is absorbed by the Phase 1 IMU/mocap time offset.

        lowpass: apply the zero-phase truth filter (mocap_lowpass_hz). The marker body vibrates
        at ~8 Hz with the motors energised (odin1: 5-7 mrad rms) while a 10 Hz sensor output can
        represent nothing above 5 Hz; unfiltered, that vibration is scored as sensor error."""
        key = (lowpass and bool(self.mocap_lowpass_hz), self.mocap_lowpass_hz, self.mocap_clean, self.mocap_clock)
        memo = self.__dict__.setdefault("_mocap_memo", {})
        if key in memo:
            return memo[key]
        if len(self.mocap.t_stamp) < 2:
            raise KeyError(f"no mocap data in {self.path}")
        s = self.mocap.series(False)
        if self.mocap_clean:
            s = self._clean_mocap(s)
        if self.mocap_clock == "server":
            a, b, t0 = self.fit_mocap_clock()
            s = PoseSeries(s.t + a + b * (s.t - t0), s.pose)
        if key[0]:
            s = lowpass_series(s, self.mocap_lowpass_hz)
        memo[key] = s
        return s

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_mocap_memo", None)
        return state

    def _clean_mocap(self, s: PoseSeries, window: int = 64, burst_s: float = 0.025,
                     gap_s: float = 0.02) -> PoseSeries:
        """Repair two delivery artefacts seen with vrpn_mocap (campaign odin1, 2026-10-03):

        * duplicates: two client nodes publishing the same server stream put every frame on the
          topic twice. A frame whose pose is bit-identical to one of the previous `window` samples
          is a re-publication (real measurements always differ in the last float bits) and is
          dropped; the first arrival is kept.
        * bursts: the server delivers 3-4 frames per packet every ~10 ms and the client stamps
          them on arrival, so stamps within a burst are ~0.1 ms apart instead of one frame period.
          Frames are consecutive captures at a fixed rate, so frame k was captured no later than
          t_j - (j - k) * dt for every later frame j in the same delivery stream; the minimum of
          these bounds over the next `burst_s` (stopping at gaps > `gap_s`) back-dates each frame
          to its capture time up to the constant minimum delay.
        """
        t, P = s.t, s.pose
        key = np.column_stack([P.p, P.R.reshape(len(t), 9)])
        keep = np.ones(len(t), bool)
        recent: dict[bytes, int] = {}
        for i in range(len(t)):
            k = key[i].tobytes()
            j = recent.get(k)
            if j is not None and i - j <= window:
                keep[i] = False
                continue
            recent[k] = i
            if len(recent) > 4 * window:
                recent = {kk: jj for kk, jj in recent.items() if i - jj <= window}
        t = t[keep]
        P = P[keep]
        o = np.argsort(t, kind="stable")
        t, P = t[o], P[o]
        dt_all = np.diff(t)
        # frame period: count / span over 1 s windows without gaps
        rates = []
        for w0 in np.arange(t[0], t[-1] - 1.0, 1.0):
            i0, i1 = np.searchsorted(t, [w0, w0 + 1.0])
            if i1 - i0 > 50 and dt_all[i0:i1 - 1].max() < gap_s:
                rates.append((i1 - i0 - 1) / (t[i1 - 1] - t[i0]))
        dt = 1.0 / float(np.median(rates)) if rates else float(np.median(dt_all))
        tc = t.copy()
        brk = np.concatenate([dt_all > gap_s, [True]])  # brk[i]: gap right after sample i
        n = len(t)
        alive = np.ones(n, bool)
        for m in range(1, int(burst_s / dt) + 3):
            alive[: n - m] &= ~brk[m - 1 : n - 1]  # no gap between k and k + m
            alive[n - m :] = False
            idx = np.nonzero(alive[: n - m])[0]
            idx = idx[t[idx + m] - t[idx] <= burst_s]
            tc[idx] = np.minimum(tc[idx], t[idx + m] - m * dt)
        self.mocap_clean_stats = {
            "raw": int(len(keep)), "duplicates": int((~keep).sum()), "rate_hz": 1.0 / dt,
            "backdate_median_ms": float(np.median(t - tc) * 1e3),
            "backdate_p99_ms": float(np.percentile(t - tc, 99) * 1e3),
            "nonmonotonic": int((np.diff(tc) <= 0).sum()),
        }
        o = np.argsort(tc, kind="stable")
        return PoseSeries(tc[o], P[o])

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


def lowpass_series(s: PoseSeries, fc: float, gap_s: float = 0.02, min_len: int = 64) -> PoseSeries:
    """Zero-phase Butterworth low-pass (4th order effective) of a pose series, per gap-free
    stretch, resampled on a uniform grid at the series' own rate. Rotations are filtered as
    sign-continuous quaternions and renormalised (fine for the small in-window changes here)."""
    from scipy.signal import butter, sosfiltfilt
    from scipy.spatial.transform import Rotation

    t = s.t
    dt = np.diff(t)
    fs = 1.0 / float(np.median(dt))
    sos = butter(2, fc, fs=fs, output="sos")
    brk = np.where(dt > gap_s)[0] + 1
    ts, ps, qs = [], [], []
    for seg in np.split(np.arange(len(t)), brk):
        sub = PoseSeries(t[seg], s.pose[seg]) if len(seg) > 1 else None
        if sub is None or len(seg) < min_len:
            ts.append(t[seg])
            ps.append(s.pose.p[seg])
            qs.append(Rotation.from_matrix(s.pose.R[seg]).as_quat())
            continue
        tu = np.arange(t[seg[0]], t[seg[-1]] + 0.5 / fs, 1.0 / fs)
        tu = tu[tu <= t[seg[-1]]]
        P, _ = sub.interp(tu)
        q = Rotation.from_matrix(P.R).as_quat()
        flips = np.cumprod(np.concatenate([[1.0], np.sign(np.sum(q[1:] * q[:-1], axis=1) + 1e-300)]))
        q = q * flips[:, None]
        q = sosfiltfilt(sos, q, axis=0)
        ts.append(tu)
        ps.append(sosfiltfilt(sos, P.p, axis=0))
        qs.append(q / np.linalg.norm(q, axis=1, keepdims=True))
    return PoseSeries(np.concatenate(ts), Pose.from_quat(np.concatenate(ps), np.concatenate(qs)))


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
    rng = np.linalg.norm(xyz, axis=1)
    valid = np.isfinite(rng) & (rng > 0.05)
    data.cloud_valid.append(int(valid.sum()))
    data.cloud_range_mean.append(float(rng[valid].mean()) if valid.any() else np.nan)
    # organized cloud of fixed size: per-pixel range statistics over time (static-scene noise)
    if data.px_sum is None or len(data.px_sum) != n:
        data.px_sum, data.px_sumsq, data.px_cnt = np.zeros(n), np.zeros(n), np.zeros(n)
    r = np.where(valid, rng, 0.0)
    data.px_sum += r
    data.px_sumsq += r * r
    data.px_cnt += valid
    numeric = {2: "u1", 4: "<u2", 6: "<u4", 1: "i1", 3: "<i2", 5: "<i4", 7: "<f4", 8: "<f8"}
    for f in msg.fields:
        if f.name in ("x", "y", "z") or f.datatype not in numeric:
            continue
        dt = np.dtype(numeric[f.datatype])
        vals = buf[:, f.offset : f.offset + dt.itemsize].copy().view(dt)[:, 0].astype(float)
        if valid.any():
            data.cloud_field_mean[f.name].append(float(np.nanmean(vals[valid])))
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
    """Read one bag. `cfg` is campaign.yaml (uses its `topics`, `sensor`, `mocap` and `preflight` parts)."""
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    from .mocap_config import MocapConfig
    from .sensors import load_sensor

    topics = cfg["topics"]
    sensor_topics = load_sensor(cfg).topics  # the sensor's imu / odometry / cloud (each optional)
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
        names = wanted_tf | {mocap.topic, topics.get("markers") or "", topics.get("rosout", "/rosout")}
        names |= {sensor_topics.get(k) or "" for k in ("odometry", "imu", "cloud")}
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
            elif conn.topic == sensor_topics.get("odometry"):
                data.odom.t_stamp.append(_stamp(msg.header))
                data.odom.t_recv.append(t_recv)
                data.odom.p.append(_xyz(msg.pose.pose.position))
                data.odom.q.append(_xyzw(msg.pose.pose.orientation))
            elif conn.topic == sensor_topics.get("imu"):
                data.imu_t.append(_stamp(msg.header))
                data.imu_gyro.append(_xyz(msg.angular_velocity))
            elif conn.topic == sensor_topics.get("cloud"):
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
