"""Per-relocalization registration offset (the Tier 1 "registration" term, needed in odin1).

Odin publishes odom->map once it relocalizes into the prior map and then corrects it only rarely,
so every relocalization lands with its own constant map-frame offset b that holds until the next
one. Seen as a time series inside one session this is a constant; across sessions it is the
dominant error term (2-5 cm, 5-10 mrad in campaign odin1), and a model without it under-predicts
any session it was not fitted on.

Model: b ~ N(0, diag(sigma_b^2)) in the map frame (x, y, z, rx, ry, rz), redrawn at every
relocalization (driver start, re-registration after an outage), added to the FOGM error.

A relocalization instance is a stretch of constant-ish odom->map: a driver restart puts the odom
origin at the new start pose, so odom->map jumps by metres; map corrections inside an instance
are centimetres.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .errors import AXES, ErrorTable, map_to_track, track_to_map
from .se3 import so3_log

MAP_AXES = ["x", "y", "z", "rx", "ry", "rz"]


@dataclass
class Segment:
    run: int
    t0: float
    t1: float
    p0: np.ndarray
    R0: np.ndarray
    p1: np.ndarray
    R1: np.ndarray
    instance: int = -1


def split_odom_map(run_id: int, t: np.ndarray, p: np.ndarray, R: np.ndarray, jump_m: float = 0.10,
                   jump_deg: float = 2.0, gap_s: float = 8.0) -> list[Segment]:
    """Cut one run's odom->map stream at jumps (relocalizations) and long gaps (driver restarts)."""
    if len(t) == 0:
        return []
    dp = np.linalg.norm(np.diff(p, axis=0), axis=1)
    dr = np.degrees(np.linalg.norm(so3_log(R[1:] @ np.swapaxes(R[:-1], -1, -2)), axis=1))
    cut = np.where((dp > jump_m) | (dr > jump_deg) | (np.diff(t) > gap_s))[0] + 1
    bounds = np.concatenate([[0], cut, [len(t)]])
    return [Segment(run_id, float(t[a]), float(t[b - 1]), p[a], R[a], p[b - 1], R[b - 1])
            for a, b in zip(bounds[:-1], bounds[1:])]


def link_instances(segments: list[Segment], jump_m: float = 0.10, jump_deg: float = 2.0) -> list[Segment]:
    """Give consecutive segments (time order) the same instance when odom->map carries on
    continuously from one to the next (same driver, same registration)."""
    segments = sorted(segments, key=lambda s: s.t0)
    k = -1
    prev = None
    for s in segments:
        same = False
        if prev is not None:
            dp = np.linalg.norm(s.p0 - prev.p1)
            dr = np.degrees(np.linalg.norm(so3_log(s.R0 @ prev.R1.T)))
            same = dp <= jump_m and dr <= jump_deg
        if not same:
            k += 1
        s.instance = k
        prev = s
    return segments


def label(tab: ErrorTable, segments: list[Segment], pad_s: float = 0.5) -> np.ndarray:
    """Instance id per error sample (-1 where no segment covers it)."""
    out = np.full(len(tab), -1)
    for s in segments:
        m = (tab.run == s.run) & (tab.t >= s.t0 - pad_s) & (tab.t <= s.t1 + pad_s)
        out[m & (out < 0)] = s.instance
    return out


@dataclass
class InstanceBias:
    instance: int
    runs: list
    n: int
    duration_s: float
    mean_M: np.ndarray  # map-frame 6-vector


def instance_biases(tab: ErrorTable, inst: np.ndarray, dt: float, min_samples: int = 30) -> list[InstanceBias]:
    eM = track_to_map(tab.err, tab.heading)
    out = []
    for k in np.unique(inst[inst >= 0]):
        m = inst == k
        if m.sum() < min_samples:
            continue
        out.append(InstanceBias(int(k), sorted({int(r) for r in tab.run[m]}), int(m.sum()), float(m.sum() * dt),
                                eM[m].mean(0)))
    return out


def debias(tab: ErrorTable, inst: np.ndarray, biases: list[InstanceBias]) -> np.ndarray:
    """Track-frame error with each instance's map-frame mean removed (samples of instances too
    short to estimate keep their error)."""
    eM = track_to_map(tab.err, tab.heading)
    for b in biases:
        eM[inst == b.instance] -= b.mean_M
    return map_to_track(eM, tab.heading)


def fit_sigma(biases: list[InstanceBias], fogm_P: np.ndarray | None = None, fogm_tau: np.ndarray | None = None,
              wrong_m: float = 0.20, wrong_deg: float = 5.0) -> dict:
    """sigma_b per map axis from the spread of instance means.

    Each mean also carries FOGM noise of variance ~ P * min(1, 2 tau_c / T); that part is
    subtracted (a short INIT trial says less about b than a 200 s session)."""
    if len(biases) < 2:
        return {"status": "need >= 2 relocalizations", "n_instances": len(biases)}
    # wrong fixes (the protocol's INIT definition: > wrong_m or > wrong_deg from the median
    # registration) are a different event, a bad relocalization, not a draw from the spread
    M_all = np.array([b.mean_M for b in biases])
    d = M_all - np.median(M_all, axis=0)
    off_m = np.linalg.norm(d[:, :3], axis=1)
    off_deg = np.degrees(np.linalg.norm(d[:, 3:], axis=1))
    bad = (off_m > wrong_m) | (off_deg > wrong_deg)
    wrong = {b.instance for b, w in zip(biases, bad) if w}
    good = [b for b in biases if b.instance not in wrong]
    wrong_info = [{"instance": b.instance, "runs": b.runs, "duration_s": b.duration_s,
                   "mean": dict(zip(MAP_AXES, b.mean_M)), "offset_m": float(om), "offset_deg": float(od)}
                  for b, w, om, od in zip(biases, bad, off_m, off_deg) if w]
    biases = good
    M = np.array([b.mean_M for b in biases])
    T = np.array([b.duration_s for b in biases])
    var = M.var(axis=0, ddof=1)
    noise = np.zeros(6)
    if fogm_P is not None and fogm_tau is not None:
        # track-frame FOGM variance is shared between x/y in the map frame as heading turns
        Pm = np.array([0.5 * (fogm_P[0] + fogm_P[1])] * 2 + [fogm_P[2]] + [0.5 * (fogm_P[3] + fogm_P[4])] * 2 + [fogm_P[5]])
        taum = np.array([max(fogm_tau[0], fogm_tau[1])] * 2 + [fogm_tau[2]] + [max(fogm_tau[3], fogm_tau[4])] * 2 + [fogm_tau[5]])
        noise = np.mean(Pm[None, :] * np.minimum(1.0, 2 * taum[None, :] / T[:, None]), axis=0)
    sig = np.sqrt(np.maximum(var - noise, 0.0))
    # robust companion: MAD of the means
    mad = 1.4826 * np.median(np.abs(M - np.median(M, axis=0)), axis=0)
    k = len(biases) - 1
    ci = [np.sqrt(np.maximum(k * var / stats.chi2.ppf(q, k) - noise, 0.0)) for q in (0.975, 0.025)]
    return {
        "sigma": dict(zip(MAP_AXES, sig)),
        "sigma_raw_spread": dict(zip(MAP_AXES, np.sqrt(var))),
        "sigma_ci95": {a: [float(ci[0][i]), float(ci[1][i])] for i, a in enumerate(MAP_AXES)},
        "sigma_mad": dict(zip(MAP_AXES, mad)),
        "fogm_part_removed": dict(zip(MAP_AXES, np.sqrt(noise))),
        "n_instances": len(biases),
        "instances": [{"instance": b.instance, "runs": b.runs, "duration_s": b.duration_s,
                       "mean": dict(zip(MAP_AXES, b.mean_M))} for b in biases],
        "wrong_fixes": wrong_info,
        "p_wrong_fix": len(wrong_info) / (len(wrong_info) + len(biases)),
        "p_wrong_fix_ci95": [float(stats.beta.ppf(0.025, len(wrong_info), len(biases) + 1)) if wrong_info else 0.0,
                             float(stats.beta.ppf(0.975, len(wrong_info) + 1, len(biases)))],
        "frame": "map (x, y, z metres; rx, ry, rz radians)",
        "redraw": "at every relocalization (driver start) and re-registration after an outage",
    }


def sample_track(sigma: np.ndarray, heading: np.ndarray, inst: np.ndarray, rng, scale: float = 1.0) -> np.ndarray:
    """One draw of b per instance, expressed in the track frame of every sample."""
    eM = np.zeros((len(heading), 6))
    for k in np.unique(inst):
        eM[inst == k] = np.sqrt(scale) * sigma * rng.standard_normal(6)
    return map_to_track(eM, heading)


__all__ = ["AXES", "MAP_AXES", "Segment", "split_odom_map", "link_instances", "label", "instance_biases",
           "debias", "fit_sigma", "sample_track"]
