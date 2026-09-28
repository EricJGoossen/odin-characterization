"""Phase 1: IMU-mocap time offset, hand-eye (mocap body -> Odin IMU), map-to-mocap alignment."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from .se3 import Pose, PoseSeries, so3_exp, so3_log, umeyama


# --------------------------------------------------------------------------- 1a time offset
@dataclass
class TimeOffsetResult:
    offset: float  # mocap_time = odin_time + offset
    peak_corr: float
    lags: np.ndarray
    corr: np.ndarray


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    return np.convolve(x, np.ones(n) / n, mode="same")


def imu_mocap_time_offset(
    imu_t: np.ndarray,
    gyro: np.ndarray,
    mocap: PoseSeries,
    max_lag: float = 0.5,
    rate: float = 200.0,
    max_gap: float = 0.02,
) -> TimeOffsetResult:
    """Cross-correlate |omega| from Odin's IMU with |omega| from differentiated mocap
    orientation. The norm is invariant to the (unknown) mounting rotation."""
    imu_t = np.asarray(imu_t)
    gnorm = np.linalg.norm(np.asarray(gyro), axis=1)
    lo = max(imu_t[0], mocap.t[0]) + max_lag + 0.05
    hi = min(imu_t[-1], mocap.t[-1]) - max_lag - 0.05
    if hi - lo < 10:
        raise ValueError("need >= 10 s of overlapping IMU and mocap data")
    t = np.arange(lo, hi, 1 / rate)
    smooth = max(1, int(0.02 * rate))
    g = _moving_average(np.interp(t, imu_t, gnorm), smooth)

    tm = np.arange(lo - max_lag, hi + max_lag, 1 / rate)
    w = mocap.body_rates(tm, 2 / rate)
    _, valid = mocap.interp(tm, max_gap)
    mnorm = np.linalg.norm(w, axis=1)
    mnorm[~valid] = np.nan
    mnorm = _moving_average(np.nan_to_num(mnorm), smooth)

    def corr_at(lag: float) -> float:
        m = np.interp(t + lag, tm, mnorm)
        a, b = g - g.mean(), m - m.mean()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    lags = np.arange(-max_lag, max_lag, 1 / rate)
    corr = np.array([corr_at(L) for L in lags])
    k = int(np.argmax(corr))
    res = minimize_scalar(
        lambda L: -corr_at(L),
        bounds=(lags[max(k - 2, 0)], lags[min(k + 2, len(lags) - 1)]),
        method="bounded",
        options={"xatol": 1e-5},
    )
    return TimeOffsetResult(float(res.x), -float(res.fun), lags, corr)


def pose_mocap_time_offset(odom: PoseSeries, mocap: PoseSeries, max_lag: float = 0.5) -> TimeOffsetResult:
    """Same |omega| cross-correlation, using angular rate differentiated from Odin's odometry.
    Odometry is published later than the IMU (tau_d), so hand-eye pairs must use this lag."""
    t = np.arange(odom.t[0] + 0.05, odom.t[-1] - 0.05, 0.005)
    return imu_mocap_time_offset(t, odom.body_rates(t, 0.05), mocap, max_lag)


# --------------------------------------------------------------------------- 1b hand-eye
@dataclass
class HandEyeResult:
    X: Pose  # T_KI: mocap body (base_link) -> Odin IMU
    rot_rms_deg: float
    trans_rms_mm: float
    n_pairs: int
    weak_directions: list  # names of parameters constrained mainly by the prior
    singular_values: np.ndarray
    motion_only_std: dict  # 1-sigma of each parameter from motion alone (m / rad)

    @property
    def passed(self) -> bool:
        return self.rot_rms_deg < 0.5 and self.trans_rms_mm < 5.0


_PARAM_NAMES = ["tx", "ty", "tz", "rx", "ry", "rz"]


def relative_motion_pairs(
    mocap: PoseSeries,
    odin: PoseSeries,
    win_min: float = 0.5,
    win_max: float = 2.0,
    step: float = 0.1,
    max_gap: float = 0.02,
    min_rot_deg: float = 3.0,
    min_trans: float = 0.03,
    seed: int = 0,
) -> tuple[Pose, Pose]:
    rng = np.random.default_rng(seed)
    lo = max(mocap.t[0], odin.t[0])
    hi = min(mocap.t[-1], odin.t[-1]) - win_max
    t0 = np.arange(lo, hi, step)
    t1 = t0 + rng.uniform(win_min, win_max, size=len(t0))
    Wa, va = mocap.interp(t0, max_gap)
    Wb, vb = mocap.interp(t1, max_gap)
    Oa, oa = odin.interp(t0, 0.2)
    Ob, ob = odin.interp(t1, 0.2)
    A = Wa.inv() @ Wb
    B = Oa.inv() @ Ob
    moving = (np.degrees(np.linalg.norm(so3_log(A.R), axis=1)) > min_rot_deg) | (
        np.linalg.norm(A.p, axis=1) > min_trans
    )
    keep = va & vb & oa & ob & moving
    return A[keep], B[keep]


def hand_eye(
    A: Pose,
    B: Pose,
    prior: Pose | None = None,
    prior_sigma_t: float = 0.05,
    prior_sigma_rot_deg: float = 10.0,
    sigma_t: float = 0.003,
    sigma_rot_deg: float = 0.3,
) -> HandEyeResult:
    """Solve A X = X B for X = T_KI by robust nonlinear least squares.

    Ground robots mostly yaw, so some directions (e.g. the vertical lever arm) are not
    observable from motion; a weak prior (CAD/mount geometry) pins those."""
    if prior is None:
        prior = Pose.identity()
    s_rot = np.radians(sigma_rot_deg)
    prior_vec = prior.to_vec()
    prior_w = np.array([1 / prior_sigma_t] * 3 + [1 / np.radians(prior_sigma_rot_deg)] * 3)

    def motion_residual(x: np.ndarray) -> np.ndarray:
        X = Pose.from_vec(x)
        AX = A @ X
        XB = X @ B
        r_rot = so3_log(np.swapaxes(AX.R, -1, -2) @ XB.R) / s_rot
        r_t = (AX.p - XB.p) / sigma_t
        return np.concatenate([r_rot, r_t], axis=1).ravel()

    def residual(x: np.ndarray) -> np.ndarray:
        r_prior = (x - prior_vec) * prior_w
        r_prior[3:] = so3_log(so3_exp(x[3:]) @ so3_exp(prior_vec[3:]).T) * prior_w[3:]
        return np.concatenate([motion_residual(x), r_prior])

    best = None
    for yaw in np.radians([0, 90, 180, 270]):
        x0 = prior_vec.copy()
        x0[3:] = so3_log(so3_exp([0, 0, yaw]) @ so3_exp(prior_vec[3:]))
        sol = least_squares(residual, x0, loss="soft_l1", f_scale=3.0)
        if best is None or sol.cost < best.cost:
            best = sol

    x = best.x
    X = Pose.from_vec(x)
    AX, XB = A @ X, X @ B
    rot_err = np.degrees(np.linalg.norm(so3_log(np.swapaxes(AX.R, -1, -2) @ XB.R), axis=1))
    t_err = np.linalg.norm(AX.p - XB.p, axis=1) * 1000
    lsq = least_squares(motion_residual, x, max_nfev=1)
    J, r = lsq.jac, lsq.fun
    sv = np.linalg.svd(J, compute_uv=False)
    # motion-only standard deviation of each parameter (scaled by the actual residual level);
    # a parameter is "weak" if motion pins it no better than half the prior does
    s2 = float(r @ r) / max(len(r) - 6, 1)
    cov = np.linalg.pinv(J.T @ J) * s2
    std = np.sqrt(np.maximum(np.diag(cov), 0))
    weak = [n for n, s, w in zip(_PARAM_NAMES, std, prior_w) if s > 0.5 / w]
    return HandEyeResult(
        X=X,
        rot_rms_deg=float(np.sqrt(np.mean(rot_err**2))),
        trans_rms_mm=float(np.sqrt(np.mean(t_err**2))),
        n_pairs=len(A),
        weak_directions=weak,
        singular_values=sv,
        motion_only_std=dict(zip(_PARAM_NAMES, std)),
    )


# --------------------------------------------------------------------------- 1d T_WM
@dataclass
class AlignmentResult:
    T_WM: Pose
    pos_rms: float
    rot_rms_deg: float
    n: int


def align_map_to_mocap(est_MI: Pose, true_WI: Pose, rot_weight_m_per_deg: float = 0.01) -> AlignmentResult:
    """One SE(3) alignment over all TRACK samples: true_WI ~= T_WM @ est_MI.
    Umeyama on positions, then refined with orientations and a robust loss."""
    T0 = umeyama(est_MI.p, true_WI.p)
    w = rot_weight_m_per_deg / np.radians(1.0)

    def residual(x):
        T = Pose.from_vec(x)
        E = T @ est_MI
        r_p = E.p - true_WI.p
        r_r = so3_log(np.swapaxes(E.R, -1, -2) @ true_WI.R) * w
        return np.concatenate([r_p, r_r], axis=1).ravel()

    sol = least_squares(residual, T0.to_vec(), loss="soft_l1", f_scale=0.05)
    T = Pose.from_vec(sol.x)
    E = T @ est_MI
    return AlignmentResult(
        T_WM=T,
        pos_rms=float(np.sqrt(np.mean(np.sum((E.p - true_WI.p) ** 2, axis=1)))),
        rot_rms_deg=float(
            np.degrees(np.sqrt(np.mean(np.sum(so3_log(np.swapaxes(E.R, -1, -2) @ true_WI.R) ** 2, axis=1))))
        ),
        n=len(est_MI),
    )
