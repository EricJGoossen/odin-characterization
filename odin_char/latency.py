"""Latency tau_d: Phase 2 regression and Phase 4 cross-correlation check."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .bagio import BagData
from .errors import Calibration, ErrorTable, odin_map_pose
from .se3 import yaw_of


@dataclass
class Regression:
    tau: float
    stderr: float
    intercept: float
    n: int


@dataclass
class LatencyResult:
    along: Regression
    yaw: Regression
    tau_d: float  # inverse-variance combination
    tau_d_stderr: float


def _ols(x: np.ndarray, y: np.ndarray) -> Regression:
    A = np.stack([x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = max(len(x) - 2, 1)
    cov = np.linalg.inv(A.T @ A) * (resid @ resid / dof)
    # error = -rate * tau_d  ->  tau_d = -slope
    return Regression(tau=float(-coef[0]), stderr=float(np.sqrt(cov[0, 0])), intercept=float(coef[1]), n=len(x))


def _demean_by_bin(keys: np.ndarray, *arrays: np.ndarray) -> list[np.ndarray]:
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = np.bincount(inv)
    out = []
    for a in arrays:
        mean = np.bincount(inv, a) / n
        out.append(a - mean[inv])
    return out


def regress_latency(
    tab: ErrorTable, min_speed: float = 0.1, min_yaw_rate: float = 0.1, bin_res: float | None = 0.35
) -> LatencyResult:
    """Slope of error vs rate. With bin_res, the regression is within position bins (a fixed
    effect per bin), so a static error field correlated with the path (e.g. yaw rate vs lobe of the
    figure-8) cannot leak into tau_d; the slope then comes from the speed sweeps at the same places."""

    def fit(m, rate, err, keys_extra):
        x, y = rate[m], err[m]
        if bin_res:
            keys = np.column_stack([np.floor(tab.pos_M[m, :2] / bin_res), keys_extra[m]])
            x, y = _demean_by_bin(keys, x, y)
            reg = _ols(x, y)
            return Regression(reg.tau, reg.stderr, 0.0, reg.n)
        return _ols(x, y)

    # heading sign keeps the two crossing directions of the figure-8 in separate bins
    direction = np.sign(np.cos(tab.heading))
    m = tab.speed > min_speed
    along = fit(m, tab.speed, tab.err[:, 0], direction)
    m = np.abs(tab.yaw_rate) > min_yaw_rate
    if m.sum() > 20:
        yaw = fit(m, tab.yaw_rate, tab.err[:, 5], np.sign(tab.yaw_rate))
    else:
        yaw = Regression(np.nan, np.inf, np.nan, int(m.sum()))
    ests = [(r.tau, r.stderr) for r in (along, yaw) if np.isfinite(r.tau) and r.stderr > 0]
    w = np.array([1 / s**2 for _, s in ests])
    tau = float(np.sum(w * np.array([t for t, _ in ests])) / w.sum())
    return LatencyResult(along, yaw, tau, float(1 / np.sqrt(w.sum())))


@dataclass
class XcorrLatency:
    tau_position: float
    tau_yaw: float


def xcorr_latency(
    bag: BagData, frames: dict, calib: Calibration, use_arrival: bool = False, max_lag: float = 0.3
) -> XcorrLatency:
    """Phase 4: shift truth to best match Odin (position along the dominant motion axis, and yaw)."""
    t_est, T_MI, has_map, _ = odin_map_pose(bag, frames, use_arrival)
    if not use_arrival:
        t_est = t_est + calib.time_offset
    t_est, T_MI = t_est[has_map], T_MI[has_map]
    mocap = bag.mocap_series()
    lo, hi = max(t_est[0], mocap.t[0]) + max_lag, min(t_est[-1], mocap.t[-1]) - max_lag
    sel = (t_est > lo) & (t_est < hi)
    t, est = t_est[sel], T_MI[sel]

    def truth(tau):
        T_WK, valid = mocap.interp(t - tau, 0.02)
        return calib.T_WM.inv() @ (T_WK @ calib.X), valid

    T0, _ = truth(0.0)
    c = T0.p[:, :2] - T0.p[:, :2].mean(0)
    axis = np.linalg.svd(c, full_matrices=False)[2][0]
    moving_pos = np.ptp(c @ axis) > 0.1
    yaw_est = np.unwrap(yaw_of(est.R))

    def cost_pos(tau):
        T, v = truth(tau)
        d = (est.p[v, :2] - T.p[v, :2]) @ axis
        return np.var(d)

    def cost_yaw(tau):
        T, v = truth(tau)
        d = yaw_est[v] - np.unwrap(yaw_of(T.R))[v]
        d = (d + np.pi) % (2 * np.pi) - np.pi
        return np.var(d)

    def best(cost):
        grid = np.linspace(-max_lag, max_lag, 121)
        k = int(np.argmin([cost(g) for g in grid]))
        r = minimize_scalar(cost, bounds=(grid[max(k - 1, 0)], grid[min(k + 1, 120)]), method="bounded")
        return float(r.x)

    tau_p = best(cost_pos) if moving_pos else np.nan
    yaw_span = np.ptp(np.unwrap(yaw_of(T0.R)))
    tau_y = best(cost_yaw) if yaw_span > 0.3 else np.nan
    return XcorrLatency(tau_p, tau_y)
