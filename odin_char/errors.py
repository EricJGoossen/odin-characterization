"""Phase 2 analysis steps 1-2: compose Odin's map-frame pose and compute body-aligned error.

Frames: W mocap world, K mocap rigid body (base_link), I Odin IMU, O Odin odom, M Odin map.
Odin publishes odom->map with parent `odom` and child `map`, i.e. T_OM (the pose of the map
frame in odom) -- the *inverse* of ROS's usual map->odom. Hence T_MI = T_OM^-1 @ T_OI.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bagio import BagData
from .se3 import Pose, PoseSeries, rot_z, so3_log, wrap, yaw_of

AXES = ["along", "cross", "vertical", "roll", "pitch", "yaw"]


@dataclass
class Calibration:
    time_offset: float  # mocap_time = odin_stamp + time_offset
    X: Pose  # T_KI
    T_WM: Pose | None = None


def odin_map_pose(bag: BagData, frames: dict, use_arrival: bool = False):
    """T_MI at every odom->imu sample, holding the latest odom->map (zero-order hold).
    Returns (t, T_MI, has_map, map_age)."""
    odom_imu = bag.tf_series(frames["odin_odom"], frames["odin_imu"], use_arrival)
    try:
        odom_map = bag.tf_series(frames["odin_odom"], frames["odin_map"], use_arrival)
    except KeyError:
        n = len(odom_imu)
        return odom_imu.t, odom_imu.pose, np.zeros(n, bool), np.full(n, np.inf)
    T_OM, has_map, age = odom_map.zoh(odom_imu.t)
    if frames.get("odom_map_is_ros_convention", False):
        # driver publishes map->odom the ROS way: the stream is T_MO, so T_OM is its inverse
        T_OM = T_OM.inv()
    T_MI = T_OM.inv() @ odom_imu.pose
    return odom_imu.t, T_MI, has_map, np.where(has_map, age, np.inf)


@dataclass
class ErrorTable:
    t: np.ndarray  # sample time (mocap clock)
    err: np.ndarray  # (N, 6) in AXES order: metres / radians
    pos_M: np.ndarray  # true position in map frame
    speed: np.ndarray  # horizontal speed (m/s)
    yaw_rate: np.ndarray  # rad/s
    map_age: np.ndarray  # age of held odom->map (s)
    heading: np.ndarray  # track-frame heading in the map frame (rad)
    est_M: Pose
    true_M: Pose
    run: np.ndarray  # integer run id per sample
    meta: dict = field(default_factory=dict)  # run id -> dict(session, phase, speed_class, ...)

    _ARRAYS = ("t", "err", "pos_M", "speed", "yaw_rate", "map_age", "heading", "run")

    def __len__(self) -> int:
        return len(self.t)

    def select(self, mask: np.ndarray) -> "ErrorTable":
        kw = {k: getattr(self, k)[mask] for k in self._ARRAYS}
        return ErrorTable(est_M=self.est_M[mask], true_M=self.true_M[mask], meta=self.meta, **kw)

    def where(self, **kv) -> "ErrorTable":
        ok = [r for r, m in self.meta.items() if all(m.get(k) == v for k, v in kv.items())]
        return self.select(np.isin(self.run, ok))

    def with_err(self, err: np.ndarray) -> "ErrorTable":
        out = self.select(np.ones(len(self), bool))
        out.err = err
        return out

    @staticmethod
    def concat(tables: list["ErrorTable"]) -> "ErrorTable":
        meta: dict = {}
        for tb in tables:
            meta.update(tb.meta)
        kw = {k: np.concatenate([getattr(tb, k) for tb in tables]) for k in ErrorTable._ARRAYS}
        cat_pose = lambda k: Pose(
            np.concatenate([getattr(tb, k).R for tb in tables]), np.concatenate([getattr(tb, k).p for tb in tables])
        )
        return ErrorTable(est_M=cat_pose("est_M"), true_M=cat_pose("true_M"), meta=meta, **kw)

    def world_position_error(self, err: np.ndarray | None = None) -> np.ndarray:
        """Map-frame translation error (undo the track-frame rotation)."""
        e = self.err if err is None else err
        return (rot_z(self.heading) @ e[:, :3, None])[..., 0]

    def arc_length(self) -> np.ndarray:
        s = np.zeros(len(self))
        d = np.linalg.norm(np.diff(self.pos_M[:, :2], axis=0), axis=1)
        d[np.diff(self.run) != 0] = 0
        s[1:] = np.cumsum(d)
        return s

    def segments(self, dt: float, tol: float = 0.25) -> list[np.ndarray]:
        """Index arrays of contiguous, evenly spaced samples (same run, no gaps)."""
        if len(self.t) == 0:
            return []
        br = np.where((np.abs(np.diff(self.t) - dt) > tol * dt) | (np.diff(self.run) != 0))[0] + 1
        return [s for s in np.split(np.arange(len(self.t)), br) if len(s) > 1]


def body_aligned(delta_p: np.ndarray, delta_theta: np.ndarray, heading: np.ndarray) -> np.ndarray:
    Rt = np.swapaxes(rot_z(heading), -1, -2)
    return np.concatenate([(Rt @ delta_p[..., None])[..., 0], (Rt @ delta_theta[..., None])[..., 0]], axis=-1)


def pose_error(est: Pose, true: Pose, heading: np.ndarray) -> np.ndarray:
    """Body-aligned error of est relative to true.

    Translation: p_est - p_true; rotation: Log(R_est R_true^T). Both rotated into the track frame
    (x along direction of travel, z up). Using the position difference rather than the translation
    part of Log(T_est T_true^-1) avoids a lever arm about the map origin."""
    dp = est.p - true.p
    dth = so3_log(est.R @ np.swapaxes(true.R, -1, -2))
    return body_aligned(dp, dth, heading)


def compute_errors(
    bag: BagData,
    frames: dict,
    calib: Calibration,
    run_id: int,
    meta: dict,
    tau_d: float = 0.0,
    rate: float = 10.0,
    use_arrival: bool = False,
    mocap_max_gap: float = 0.02,
    min_speed_for_track: float = 0.05,
) -> ErrorTable:
    assert calib.T_WM is not None, "T_WM must be fitted before computing errors"
    t_est, T_MI, has_map, map_age = odin_map_pose(bag, frames, use_arrival)
    if not use_arrival:
        t_est = t_est + calib.time_offset
    mocap = bag.mocap_series()

    # consumer view: at each grid time take the latest estimate
    grid = np.arange(max(t_est[0], mocap.t[0]) + 0.1, min(t_est[-1], mocap.t[-1]) - 0.1, 1 / rate)
    idx = np.searchsorted(t_est, grid, side="right") - 1
    ok = (idx >= 0) & has_map[np.clip(idx, 0, None)]
    idx = np.clip(idx, 0, len(t_est) - 1)
    t_s = t_est[idx]  # the estimate's own time
    t_truth = t_s - tau_d

    T_WK, valid = mocap.interp(t_truth, mocap_max_gap)
    T_WI = T_WK @ calib.X
    true_M = calib.T_WM.inv() @ T_WI
    est_M = T_MI[idx]

    # velocities in map frame from truth
    h = 0.05
    Ka, va = mocap.interp(t_truth - h, mocap_max_gap)
    Kb, vb = mocap.interp(t_truth + h, mocap_max_gap)
    v_W = (Kb.p - Ka.p) / (2 * h)
    v_M = (calib.T_WM.R.T @ v_W[..., None])[..., 0]
    speed = np.linalg.norm(v_M[:, :2], axis=1)
    RM = calib.T_WM.R.T
    body_yaw = yaw_of(RM @ T_WK.R)
    yaw_rate = wrap(yaw_of(Kb.R) - yaw_of(Ka.R)) / (2 * h)
    heading = np.where(speed > min_speed_for_track, np.arctan2(v_M[:, 1], v_M[:, 0]), body_yaw)

    err = pose_error(est_M, true_M, heading)
    keep = ok & valid & va & vb
    # drop duplicate consumer samples of the same estimate? no: the consumer really sees them.
    return ErrorTable(
        t=grid[keep],
        err=err[keep],
        pos_M=true_M.p[keep],
        speed=speed[keep],
        yaw_rate=yaw_rate[keep],
        map_age=(map_age[idx] + (grid - t_s))[keep],
        heading=heading[keep],
        est_M=est_M[keep],
        true_M=true_M[keep],
        run=np.full(int(keep.sum()), run_id),
        meta={run_id: meta},
    )
