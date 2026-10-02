"""Phase 6: hold-out validation of the Tier 0 model by Monte Carlo along real trajectories."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .errors import AXES, ErrorTable
from .fogm import segment_acf
from .model import Tier0Model

RPE_LENGTHS = (0.5, 1.0, 2.0, 4.0, 8.0)

# which Tier 1 term a failing metric points to
TIER1_HINTS = {
    "rmse": "speed-dependent drift/registration terms or geometry-dependent covariance H(p)",
    "cdf": "heavier-tailed innovations or a mixture/outlier term",
    "acf": "second FOGM component or a slow drift term",
    "rpe": "speed-dependent drift (error that grows with distance travelled)",
    "events": "event model (lambda_lost / lambda_rec) or stationary correction gating",
}


def rpe_curve(tab: ErrorTable, dp_M: np.ndarray, dt: float, lengths=RPE_LENGTHS) -> np.ndarray:
    """Mean relative translation error over path segments of each length."""
    s = tab.arc_length()
    out = np.full(len(lengths), np.nan)
    segs = tab.segments(dt)
    for li, L in enumerate(lengths):
        vals = []
        for seg in segs:
            ss = s[seg]
            j = np.searchsorted(ss, ss + L)
            ok = j < len(seg)
            i = np.where(ok)[0][::5]
            if len(i):
                vals.append(np.linalg.norm(dp_M[seg[j[i]], :2] - dp_M[seg[i], :2], axis=1))
        if vals:
            out[li] = float(np.mean(np.concatenate(vals)))
    return out


@dataclass
class ValidationResult:
    passed: bool
    metrics: dict
    failing: list


def validate(model: Tier0Model, held: ErrorTable, n_mc: int = 50, real_lost_events: int = 0,
             seed: int = 0, acf_tau_multiple: float = 5.0, band=(5, 95)) -> ValidationResult:
    rng = np.random.default_rng(seed)
    dt = model.dt
    # rank tests need enough realisations to be able to reach the per-axis level at all
    n_mc = max(n_mc, int(np.ceil(3 / ((100 - band[1] + band[0]) / 100 / 6))))
    segs = held.segments(dt)
    idx = np.concatenate(segs) if segs else np.zeros(0, int)
    real = held.err
    duration = len(idx) * dt

    tau_max = max(-dt / np.log(p.a) for p in model.moving)
    max_lag = int(max(2, min(acf_tau_multiple * tau_max / dt, max(len(s) for s in segs) // 2)))
    real_rmse = np.sqrt(np.mean(real[idx] ** 2, axis=0))
    real_acf = np.array([segment_acf([real[s, a] for s in segs], max_lag) for a in range(6)])
    real_rpe = rpe_curve(held, held.world_position_error(), dt)

    sims_rmse, sims_acf, sims_rpe, sims_q, pooled = [], [], [], [], [[] for _ in range(6)]
    qs = np.linspace(0.05, 0.95, 19)
    for _ in range(n_mc):
        sim = model.simulate_error(held.pos_M, held.speed, held.yaw_rate, segs, rng, parameter_uncertainty=True)
        sims_rmse.append(np.sqrt(np.mean(sim[idx] ** 2, axis=0)))
        sims_acf.append([segment_acf([sim[s, a] for s in segs], max_lag) for a in range(6)])
        sims_rpe.append(rpe_curve(held, held.world_position_error(sim), dt))
        sims_q.append(np.quantile(sim[idx], qs, axis=0))
        for a in range(6):
            pooled[a].append(sim[idx, a][::10])
    sims_rmse, sims_acf, sims_rpe, sims_q = map(np.array, (sims_rmse, sims_acf, sims_rpe, sims_q))
    lo, hi = band
    # The protocol's per-axis 5-95% bands are reported as-is. The verdict uses rank tests of the
    # real data among the Monte Carlo realisations (exchangeable under a correct model, so the
    # p-values are exact): a global-envelope max-deviation statistic for curves (ACF, QQ, RPE),
    # whose neighbouring points are strongly correlated, and a two-sided rank for scalars. The
    # level is family-wise over the six axes.
    alpha = (100 - hi + lo) / 100  # 0.10
    alpha_axis = alpha / 6

    def band_of(x, a, b):
        return np.nanpercentile(x, a, 0), np.nanpercentile(x, b, 0)

    def rank_p_scalar(real_v, sims_v):
        # two-sided, normal approximation of the simulated distribution (RMSE over thousands of
        # samples is close to normal); continuous, so it works with any number of realisations
        sd = float(np.std(sims_v)) + 1e-15
        return float(2 * stats.norm.sf(abs(real_v - float(np.mean(sims_v))) / sd))

    def rank_p_curve(real_c, sims_c):
        mu = np.nanmean(sims_c, 0)
        sd = np.nanstd(sims_c, 0) + 1e-12
        T = lambda c: np.nanmax(np.abs(c - mu) / sd)
        t_real = T(real_c)
        t_sims = np.array([T(s) for s in sims_c])
        return float((1 + np.sum(t_sims >= t_real)) / (len(sims_c) + 1))

    metrics: dict = {"band_percentiles": [lo, hi], "alpha_family": alpha, "alpha_per_axis": alpha_axis}
    failing = []
    # RMSE
    r_lo, r_hi = band_of(sims_rmse, lo, hi)
    metrics["rmse"] = {}
    for i, a in enumerate(AXES):
        p = rank_p_scalar(real_rmse[i], sims_rmse[:, i])
        metrics["rmse"][a] = {"real": float(real_rmse[i]), "sim_band": [float(r_lo[i]), float(r_hi[i])],
                              "inside_5_95": bool(r_lo[i] <= real_rmse[i] <= r_hi[i]), "p": p, "pass": p > alpha_axis}
    if not all(v["pass"] for v in metrics["rmse"].values()):
        failing.append("rmse")
    # CDF / QQ
    real_q = np.quantile(real[idx], qs, axis=0)
    q_lo, q_hi = band_of(sims_q, lo, hi)
    metrics["cdf"] = {}
    for i, a in enumerate(AXES):
        inside = np.mean((real_q[:, i] >= q_lo[:, i]) & (real_q[:, i] <= q_hi[:, i]))
        ks = stats.ks_2samp(real[idx, i][::10], np.concatenate(pooled[i]))
        p = rank_p_curve(real_q[:, i], sims_q[:, :, i])
        metrics["cdf"][a] = {"quantiles_inside_5_95": float(inside), "ks_p": float(ks.pvalue), "p": p, "pass": p > alpha_axis}
    if not all(v["pass"] for v in metrics["cdf"].values()):
        failing.append("cdf")
    # ACF (lag 0 is always 1: excluded from the statistic)
    a_lo, a_hi = band_of(sims_acf, lo, hi)
    inside = np.mean((real_acf >= a_lo - 1e-9) & (real_acf <= a_hi + 1e-9), axis=1)
    metrics["acf"] = {}
    for i, a in enumerate(AXES):
        p = rank_p_curve(real_acf[i, 1:], sims_acf[:, i, 1:])
        metrics["acf"][a] = {"fraction_inside_5_95": float(inside[i]), "p": p, "pass": p > alpha_axis}
    if not all(v["pass"] for v in metrics["acf"].values()):
        failing.append("acf")
    # RPE
    p_lo, p_hi = band_of(sims_rpe, lo, hi)
    fin = np.isfinite(real_rpe)
    p_rpe = rank_p_curve(real_rpe[fin], sims_rpe[:, fin]) if fin.any() else 1.0
    metrics["rpe"] = {"lengths_m": list(RPE_LENGTHS), "real": real_rpe.tolist(), "sim_lo": p_lo.tolist(),
                      "sim_hi": p_hi.tolist(), "inside_5_95": bool(np.all((real_rpe[fin] >= p_lo[fin]) & (real_rpe[fin] <= p_hi[fin]))),
                      "p": p_rpe, "pass": p_rpe > alpha}
    if not metrics["rpe"]["pass"]:
        failing.append("rpe")
    # events
    mu = model.lambda_lost * duration
    p_ev = float(min(1.0, 2 * min(stats.poisson.cdf(real_lost_events, mu), stats.poisson.sf(real_lost_events - 1, mu)))) if mu > 0 else (1.0 if real_lost_events == 0 else 0.0)
    metrics["events"] = {"real_lost": real_lost_events, "expected": mu, "p": p_ev, "pass": p_ev > alpha}
    if not metrics["events"]["pass"]:
        failing.append("events")

    metrics["_curves"] = {"real_acf": real_acf, "acf_lo": a_lo, "acf_hi": a_hi, "qs": qs, "real_q": real_q,
                          "q_lo": q_lo, "q_hi": q_hi, "dt": dt}
    return ValidationResult(not failing, metrics, failing)