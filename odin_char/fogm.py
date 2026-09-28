"""Phase 2 step 5 / Phase 3: first-order Gauss-Markov fit per axis with Student-t innovations.

eta_{k+1} = a eta_k + e_k,  e_k ~ t_nu(0, s),  a = exp(-dt / tau_c),  P = Var(eta)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


@dataclass
class AxisFit:
    a: float
    tau_c: float
    P: float
    nu: float  # inf => Gaussian innovations
    scale: float  # innovation scale (t scale, or Gaussian std)
    loglik_gain_t: float
    ljung_box_p: float
    acf: np.ndarray
    acf_model: np.ndarray
    slow_tail: bool
    second_component: dict | None
    run_length_ok: bool
    min_run_over_tau: float
    n: int
    innovations: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))

    def summary(self) -> dict:
        return {
            "a": self.a,
            "tau_c": self.tau_c,
            "P": self.P,
            "sigma": float(np.sqrt(self.P)),
            "nu": None if not np.isfinite(self.nu) else self.nu,
            "innovation_scale": self.scale,
            "loglik_gain_t": self.loglik_gain_t,
            "ljung_box_p": self.ljung_box_p,
            "slow_acf_tail": self.slow_tail,
            "second_component": self.second_component,
            "run_length_ok": self.run_length_ok,
            "min_run_over_tau_c": self.min_run_over_tau,
            "n": self.n,
        }


def segment_acf(x_segments: list[np.ndarray], max_lag: int) -> np.ndarray:
    """Pooled autocorrelation over several segments (no correlation across gaps)."""
    allx = np.concatenate(x_segments)
    mu, var = allx.mean(), allx.var()
    acf = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        num, cnt = 0.0, 0
        for s in x_segments:
            if len(s) > k:
                d = s - mu
                num += np.sum(d[: len(s) - k] * d[k:])
                cnt += len(s) - k
        acf[k] = num / cnt / var if cnt else np.nan
    return acf


def ljung_box(x_segments: list[np.ndarray], lags: int) -> float:
    n = sum(len(s) for s in x_segments)
    r = segment_acf(x_segments, lags)[1:]
    Q = n * (n + 2) * np.sum(r**2 / (n - np.arange(1, lags + 1)))
    return float(stats.chi2.sf(Q, lags))


def fit_axis(segments: list[np.ndarray], dt: float, loglik_threshold: float = 10.0, run_durations=None) -> AxisFit:
    segments = [s for s in segments if len(s) > 2]
    x0 = np.concatenate([s[:-1] for s in segments])
    x1 = np.concatenate([s[1:] for s in segments])
    allx = np.concatenate(segments)
    a = float(np.clip((x0 @ x1) / (x0 @ x0), 1e-6, 1 - 1e-9))
    tau_c = -dt / np.log(a)
    P = float(np.var(allx))
    e = x1 - a * x0
    e_segments = [s[1:] - a * s[:-1] for s in segments]

    # innovation distribution
    sd = float(np.std(e))
    ll_norm = float(np.sum(stats.norm.logpdf(e, 0, sd)))
    nu, _, sc = stats.t.fit(e, floc=0)
    ll_t = float(np.sum(stats.t.logpdf(e, nu, 0, sc)))
    gain = ll_t - ll_norm
    use_t = gain > loglik_threshold and nu < 100
    nu_out, scale_out = (float(nu), float(sc)) if use_t else (np.inf, sd)

    lb_lags = int(min(20, max(5, len(e) // 50)))
    lb_p = ljung_box(e_segments, lb_lags)

    max_lag = int(min(max(5 * tau_c / dt, 10), max(len(s) for s in segments) // 2))
    acf = segment_acf(segments, max_lag)
    model = a ** np.arange(max_lag + 1)
    # slow tail: beyond 3 tau_c the empirical ACF stays well above the single-exponential model
    k3 = int(3 * tau_c / dt)
    band = 2 / np.sqrt(len(allx))
    slow = bool(k3 < max_lag and np.nanmean(acf[k3:] - model[k3:]) > max(0.1, 2 * band))
    second = None
    if slow:
        lags = np.arange(max_lag + 1) * dt
        try:
            f = lambda L, w, t1, t2: w * np.exp(-L / t1) + (1 - w) * np.exp(-L / t2)
            (w, t1, t2), _ = curve_fit(f, lags, acf, p0=[0.7, tau_c, 10 * tau_c], bounds=([0, 1e-3, 1e-3], [1, 1e4, 1e5]))
            if t1 > t2:
                w, t1, t2 = 1 - w, t2, t1
            second = {"weight_fast": float(w), "tau_fast": float(t1), "tau_slow": float(t2),
                      "P_fast": float(w * P), "P_slow": float((1 - w) * P)}
        except RuntimeError:
            second = None

    durations = run_durations if run_durations else [len(s) * dt for s in segments]
    run_over_tau = min(durations) / tau_c
    return AxisFit(
        a=a,
        tau_c=float(tau_c),
        P=P,
        nu=nu_out,
        scale=scale_out,
        loglik_gain_t=gain,
        ljung_box_p=lb_p,
        acf=acf,
        acf_model=model,
        slow_tail=slow,
        second_component=second,
        run_length_ok=bool(run_over_tau >= 50),
        min_run_over_tau=float(run_over_tau),
        n=len(allx),
        innovations=e,
    )


def fit_fogm(eta: np.ndarray, segments: list[np.ndarray], dt: float, loglik_threshold: float = 10.0, run_durations=None) -> list[AxisFit]:
    """eta: (N, 6); segments: index arrays of contiguous samples."""
    return [fit_axis([eta[s, a] for s in segments], dt, loglik_threshold, run_durations) for a in range(eta.shape[1])]
