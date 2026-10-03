"""Phase 0 preflight: does mocap's IR disturb Odin, or Odin's 940 nm laser disturb mocap?"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .bagio import BagData


def _compare(a, b, name: str, alpha: float = 0.01, min_rel: float = 0.05) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return {"metric": name, "status": "not recorded"}
    p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    rel = float(abs(a.mean() - b.mean()) / (abs(b.mean()) + 1e-12))
    significant = p < alpha and rel > min_rel
    return {"metric": name, "on_mean": float(a.mean()), "off_mean": float(b.mean()), "rel_diff": rel, "p": p,
            "significant": bool(significant)}


def odin_side(mocap_on: BagData, mocap_off: BagData) -> list[dict]:
    """Odin cloud statistics with mocap strobes on vs off."""
    out = [
        _compare(mocap_on.cloud_valid, mocap_off.cloud_valid, "cloud_raw valid points per cloud"),
        _compare(mocap_on.cloud_range_mean, mocap_off.cloud_range_mean, "mean range of valid points (m)"),
        _compare(mocap_on.cloud_plane_std, mocap_off.cloud_plane_std, "flat-wall range noise (m)"),
    ]
    out.append(_pixel_noise(mocap_on, mocap_off))
    for f in sorted(set(mocap_on.cloud_field_mean) & set(mocap_off.cloud_field_mean)):
        out.append(_compare(mocap_on.cloud_field_mean[f], mocap_off.cloud_field_mean[f], f"mean point {f}"))
    return out


def _pixel_noise(on: BagData, off: BagData, min_rel: float = 0.10) -> dict:
    """Static scene: per-pixel temporal range std, compared on the pixels valid in both runs."""
    if on.px_cnt is None or off.px_cnt is None or len(on.px_cnt) != len(off.px_cnt):
        return {"metric": "per-pixel range noise (m)", "status": "not recorded"}
    both = (on.px_cnt >= 20) & (off.px_cnt >= 20)
    if both.sum() < 100:
        return {"metric": "per-pixel range noise (m)", "status": "too few common pixels"}

    def std(b):
        m = b.px_sum[both] / b.px_cnt[both]
        return np.sqrt(np.maximum(b.px_sumsq[both] / b.px_cnt[both] - m * m, 0.0))

    s_on, s_off = std(on), std(off)
    p = float(stats.wilcoxon(s_on, s_off).pvalue)
    med_on, med_off = float(np.median(s_on)), float(np.median(s_off))
    rel = abs(med_on - med_off) / (med_off + 1e-12)
    return {"metric": "per-pixel range noise (m), median over pixels", "on_mean": med_on, "off_mean": med_off,
            "rel_diff": rel, "p": p, "pixels": int(both.sum()), "significant": bool(p < 0.01 and rel > min_rel)}


def mocap_side(odin_on: BagData, odin_off: BagData, expected_markers: int | None) -> list[dict]:
    """Mocap marker counts (ghosts) and body-pose jitter with Odin's laser on vs off."""
    out = []
    if odin_on.marker_n and odin_off.marker_n:
        out.append(_compare(odin_on.marker_n, odin_off.marker_n, "visible marker count"))
        if expected_markers:
            g_on = np.maximum(np.array(odin_on.marker_n) - expected_markers, 0)
            g_off = np.maximum(np.array(odin_off.marker_n) - expected_markers, 0)
            out.append(_compare(g_on, g_off, "ghost markers per frame"))
    else:
        out.append({"metric": "marker residuals / ghost markers", "status": "not in bag: check in the mocap software"})

    def jitter(b: BagData):
        s = b.mocap_series()
        return np.linalg.norm(s.pose.p - np.median(s.pose.p, axis=0), axis=1)

    try:
        out.append(_compare(jitter(odin_on), jitter(odin_off), "static rigid-body position jitter (m)"))
    except KeyError:
        pass
    return out


def passed(results: list[dict]) -> bool:
    return not any(r.get("significant") for r in results)
