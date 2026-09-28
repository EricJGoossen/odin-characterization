"""Phase 5: INIT / TRACK / LOST event rates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from .errors import ErrorTable


# --------------------------------------------------------------------------- INIT
@dataclass
class InitTrial:
    offset_m: float
    heading_deg: float
    moving: bool
    time_to_fix: float | None  # None = no fix before timeout
    timeout: float
    fix_pos_err: float | None = None
    fix_yaw_err_deg: float | None = None

    @property
    def success(self) -> bool:
        return self.time_to_fix is not None

    @property
    def correct(self) -> bool | None:
        if not self.success or self.fix_pos_err is None:
            return None
        return self.fix_pos_err < 0.20 and abs(self.fix_yaw_err_deg) < 5.0


@dataclass
class InitResult:
    n: int
    n_success: int
    lambda_fix: float  # 1/s, censored-exponential MLE
    fix_delay_min: float  # shortest observed time-to-fix (a dead time the exponential can't model)
    lambda_fix_shifted: float  # exponential rate after subtracting fix_delay_min
    ks_p: float  # KS test of time-to-fix against the fitted exponential (successes only)
    ks_p_shifted: float
    logistic: dict  # coefficients for P(success | offset, heading, moving)
    p_wrong: float | None


def _logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 1e-2) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), X])

    def nll(w):
        z = Xb @ w
        return np.sum(np.logaddexp(0, z) - y * z) + l2 * np.sum(w[1:] ** 2)

    return minimize(nll, np.zeros(Xb.shape[1]), method="BFGS").x


def analyze_init(trials: list[InitTrial]) -> InitResult:
    succ = [t for t in trials if t.success]
    T = np.array([t.time_to_fix for t in succ])
    exposure = sum(t.time_to_fix if t.success else t.timeout for t in trials)
    lam = len(succ) / exposure if exposure > 0 else np.nan
    ks_p = float(stats.kstest(T, "expon", args=(0, 1 / lam)).pvalue) if len(T) >= 3 else np.nan
    d0 = float(T.min()) if len(T) else np.nan
    exp_shift = sum((t.time_to_fix if t.success else t.timeout) - d0 for t in trials) if len(T) else np.nan
    lam_s = len(succ) / exp_shift if len(T) and exp_shift > 0 else np.nan
    ks_s = float(stats.kstest(T - d0, "expon", args=(0, 1 / lam_s)).pvalue) if len(T) >= 3 and np.isfinite(lam_s) else np.nan

    X = np.array([[t.offset_m, t.heading_deg, float(t.moving)] for t in trials])
    y = np.array([float(t.success) for t in trials])
    if 0 < y.sum() < len(y):
        w = _logistic_fit(X, y)
        logistic = {"intercept": w[0], "offset_m": w[1], "heading_deg": w[2], "moving": w[3]}
    else:
        logistic = {"intercept": np.inf if y.all() else -np.inf, "offset_m": 0.0, "heading_deg": 0.0, "moving": 0.0,
                    "note": "all trials " + ("succeeded" if y.all() else "failed")}
    checked = [t.correct for t in succ if t.correct is not None]
    p_wrong = float(1 - np.mean(checked)) if checked else None
    return InitResult(len(trials), len(succ), float(lam), d0, float(lam_s), ks_p, ks_s,
                      {k: float(v) for k, v in logistic.items() if k != "note"} | ({"note": logistic["note"]} if "note" in logistic else {}),
                      p_wrong)


# --------------------------------------------------------------------------- LOST (natural)
@dataclass
class LostEvent:
    t: float
    run: int
    reason: str  # "map_stale" | "error_jump"


def detect_lost(
    tab: ErrorTable,
    map_update_times: dict,
    stale_s: float = 1.0,
    jump_m: float = 0.20,
    merge_s: float = 2.0,
) -> list[LostEvent]:
    """map_update_times: run id -> array of odom->map message times (mocap clock)."""
    events: list[LostEvent] = []
    for run in np.unique(tab.run):
        m = tab.run == run
        t = tab.t[m]
        pe = np.linalg.norm(tab.err[m, :3], axis=1)
        cand = []
        upd = np.sort(np.asarray(map_update_times.get(int(run), [])))
        if len(upd) > 1:
            upd = upd[(upd >= t[0]) & (upd <= t[-1])]
            gaps = np.diff(upd)
            cand += [(upd[i] + stale_s, "map_stale") for i in np.where(gaps > stale_s)[0]]
        jumps = np.abs(np.diff(pe)) > jump_m
        cand += [(t[i + 1], "error_jump") for i in np.where(jumps & (np.diff(t) < 0.5))[0]]
        cand.sort()
        last = -np.inf
        for tc, why in cand:
            if tc - last > merge_s:
                events.append(LostEvent(float(tc), int(run), why))
            last = tc
    return events


def lost_rate(n_events: int, tracked_time_s: float) -> dict:
    if n_events == 0:
        return {"lambda_lost": 0.0, "lambda_lost_upper95": 3.0 / tracked_time_s, "n": 0, "T": tracked_time_s,
                "note": "no spontaneous losses; rule of three upper bound"}
    lo, hi = stats.chi2.ppf(0.025, 2 * n_events) / 2, stats.chi2.ppf(0.975, 2 * (n_events + 1)) / 2
    return {"lambda_lost": n_events / tracked_time_s, "lambda_lost_ci95": [lo / tracked_time_s, hi / tracked_time_s],
            "n": n_events, "T": tracked_time_s}


# --------------------------------------------------------------------------- LOST (induced)
@dataclass
class OutageTrial:
    cover_s: float
    t_cover: float
    t_uncover: float
    max_err: float
    err_growth_rate: float  # m/s during the outage
    time_to_recover: float | None


def analyze_outage(tab: ErrorTable, t_cover: float, t_uncover: float, map_updates: np.ndarray,
                   recover_err_m: float = 0.20, horizon_s: float = 30.0) -> OutageTrial:
    pe = np.linalg.norm(tab.err[:, :3], axis=1)
    during = (tab.t >= t_cover) & (tab.t <= t_uncover)
    base = np.median(pe[(tab.t < t_cover) & (tab.t > t_cover - 5)]) if np.any(tab.t < t_cover) else 0.0
    if during.sum() > 2:
        slope = float(np.polyfit(tab.t[during] - t_cover, pe[during] - base, 1)[0])
        mx = float(pe[during].max())
    else:
        slope, mx = np.nan, np.nan
    after = (tab.t > t_uncover) & (tab.t < t_uncover + horizon_s)
    upd_after = np.asarray(map_updates)[(np.asarray(map_updates) > t_uncover)]
    rec = None
    ok = np.where(after & (pe < max(recover_err_m, 3 * base)))[0]
    if len(ok) and len(upd_after):
        t_ok = max(tab.t[ok[0]], upd_after[0])
        rec = float(t_ok - t_uncover)
    return OutageTrial(t_uncover - t_cover, t_cover, t_uncover, mx, slope, rec)


def recovery_rate(trials: list[OutageTrial]) -> dict:
    rec = np.array([t.time_to_recover for t in trials if t.time_to_recover is not None])
    by_dur: dict = {}
    for t in trials:
        d = by_dur.setdefault(round(t.cover_s), {"max_err": [], "growth": [], "recover": []})
        d["max_err"].append(t.max_err)
        d["growth"].append(t.err_growth_rate)
        d["recover"].append(t.time_to_recover)
    return {
        "lambda_rec": float(len(rec) / rec.sum()) if len(rec) and rec.sum() > 0 else None,
        "n_recovered": int(len(rec)),
        "n_trials": len(trials),
        "by_cover_duration": {
            k: {"max_err_mean": float(np.nanmean(v["max_err"])), "err_growth_mps": float(np.nanmean(v["growth"])),
                "recover_s": [None if r is None else float(r) for r in v["recover"]]}
            for k, v in sorted(by_dur.items())
        },
    }
