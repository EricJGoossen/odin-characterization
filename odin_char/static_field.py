"""Phase 2 step 4: static error field eps_s(p), binned by position and GP-interpolated."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from .errors import AXES, ErrorTable


@dataclass
class Bins:
    centers: np.ndarray  # (B, 2)
    mean: np.ndarray  # (B, 6)
    var: np.ndarray  # (B, 6)
    count: np.ndarray  # (B,)
    visits: np.ndarray  # (B,) distinct passes through the bin: the effective sample count,
    # since consecutive samples in one pass share the same (correlated) dynamic error


def bin_errors(tab: ErrorTable, res: float, min_count: int = 5, origin=(0.0, 0.0)) -> Bins:
    ij = np.floor((tab.pos_M[:, :2] - np.asarray(origin)) / res).astype(int)
    keys, inv, count = np.unique(ij, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    B = len(keys)
    s = np.zeros((B, 6))
    s2 = np.zeros((B, 6))
    np.add.at(s, inv, tab.err)
    np.add.at(s2, inv, tab.err**2)
    new_visit = np.ones(len(inv), bool)
    new_visit[1:] = (inv[1:] != inv[:-1]) | (tab.run[1:] != tab.run[:-1]) | (np.diff(tab.t) > 1.0)
    visits = np.bincount(inv[new_visit], minlength=B)
    mean = s / count[:, None]
    var = np.maximum(s2 / count[:, None] - mean**2, 0)
    ok = count >= min_count
    centers = (keys + 0.5) * res + np.asarray(origin)
    return Bins(centers[ok], mean[ok], var[ok], count[ok], visits[ok])


@dataclass
class StaticField:
    res: float
    enabled: np.ndarray = field(default_factory=lambda: np.zeros(6, bool))
    gps: list = field(default_factory=lambda: [None] * 6)
    bins: Bins | None = None
    r2: np.ndarray = field(default_factory=lambda: np.full(6, np.nan))

    @staticmethod
    def fit(tab: ErrorTable, res: float, min_count: int = 5, axes_enabled=None) -> "StaticField":
        bins = bin_errors(tab, res, min_count)
        f = StaticField(res=res, bins=bins)
        f.enabled = np.ones(6, bool) if axes_enabled is None else np.asarray(axes_enabled, bool)
        for a in range(6):
            if not f.enabled[a] or len(bins.centers) < 3:
                f.enabled[a] = False
                continue
            y = bins.mean[:, a]
            scale = float(np.var(y)) + 1e-12
            kernel = ConstantKernel(scale, (1e-8, 1e2)) * RBF(1.0, (0.2, 20.0)) + WhiteKernel(
                scale * 0.1, (1e-10, 1e2)
            )
            gp = GaussianProcessRegressor(
                kernel, alpha=bins.var[:, a] / np.maximum(bins.visits, 1) + 1e-12, normalize_y=False, n_restarts_optimizer=2, random_state=0
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(bins.centers, y)
            f.gps[a] = gp
        return f

    def predict(self, pos: np.ndarray) -> np.ndarray:
        out = np.zeros((len(pos), 6))
        for a in range(6):
            if self.enabled[a] and self.gps[a] is not None:
                out[:, a] = self._predict_mean(self.gps[a], pos[:, :2])
        return out

    def sample(self, pos: np.ndarray, rng) -> np.ndarray:
        """One draw of the field from its GP posterior (smooth part only). Drawn on the bin
        centres and carried to `pos` as a deviation from the mean at the nearest centre."""
        from scipy.linalg import solve_triangular

        out = self.predict(pos)
        if self.bins is None or len(self.bins.centers) == 0:
            return out
        C = self.bins.centers
        d2 = ((pos[:, None, :2] - C[None]) ** 2).sum(-1)
        nearest = np.argmin(d2, axis=1)
        for a in range(6):
            gp = self.gps[a]
            if not self.enabled[a] or gp is None:
                continue
            k = gp.kernel_.k1
            Ks = k(C, gp.X_train_)
            v = solve_triangular(gp.L_, Ks.T, lower=True)
            cov = k(C) - v.T @ v
            w, V = np.linalg.eigh((cov + cov.T) / 2)
            dev = V @ (np.sqrt(np.clip(w, 0, None)) * rng.standard_normal(len(w)))
            out[:, a] += dev[nearest]
        return out

    @staticmethod
    def _predict_mean(gp: GaussianProcessRegressor, x: np.ndarray) -> np.ndarray:
        # mean of the smooth part only (exclude the white-noise kernel)
        k = gp.kernel_.k1
        return k(x, gp.X_train_) @ gp.alpha_

    def export_grid(self, lo, hi, step: float = 0.1) -> dict:
        xs = np.arange(lo[0], hi[0] + 1e-9, step)
        ys = np.arange(lo[1], hi[1] + 1e-9, step)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        pred = self.predict(np.stack([X.ravel(), Y.ravel()], 1)).reshape(len(xs), len(ys), 6)
        return {"x": xs, "y": ys, "field": pred, "axes": np.array(AXES), "enabled": self.enabled}


def cross_session_r2(tab: ErrorTable, res: float, min_count: int = 5) -> tuple[np.ndarray, list]:
    """Leave-one-session-out: fit on the other sessions, score per-bin means of the held-out one.
    Returns (mean R^2 per axis, per-fold R^2 list)."""
    sessions = sorted({m["session"] for m in tab.meta.values()})
    folds = []
    for s in sessions:
        train = tab.select(np.array([tab.meta[r]["session"] != s for r in tab.run]))
        test = tab.where(session=s)
        if len(train) == 0 or len(test) == 0:
            continue
        f = StaticField.fit(train, res, min_count)
        tb = bin_errors(test, res, min_count)
        pred = f.predict(tb.centers)
        r2 = np.full(6, np.nan)
        for a in range(6):
            ss = np.sum((tb.mean[:, a] - tb.mean[:, a].mean()) ** 2)
            if ss > 0 and f.enabled[a]:
                r2[a] = 1 - np.sum((tb.mean[:, a] - pred[:, a]) ** 2) / ss
        folds.append(r2)
    return np.nanmean(np.array(folds), axis=0) if folds else np.full(6, np.nan), folds


def leave_one_session_out_residual(tab: ErrorTable, field_obj: StaticField, res: float, min_count: int = 5) -> np.ndarray:
    """eta = eps - eps_s(p) where eps_s is fitted without the sample's own session."""
    eta = tab.err.copy()
    sessions = sorted({m["session"] for m in tab.meta.values()})
    sess_of = np.array([tab.meta[r]["session"] for r in tab.run])
    if len(sessions) < 2:
        return eta - field_obj.predict(tab.pos_M)
    for s in sessions:
        m = sess_of == s
        f = StaticField.fit(tab.select(~m), res, min_count, axes_enabled=field_obj.enabled)
        eta[m] -= f.predict(tab.pos_M[m])
    return eta
