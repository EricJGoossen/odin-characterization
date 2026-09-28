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
        _compare(mocap_on.cloud_n, mocap_off.cloud_n, "cloud_raw point count"),
        _compare(mocap_on.cloud_plane_std, mocap_off.cloud_plane_std, "flat-wall range noise (m)"),
    ]
    for f in sorted(set(mocap_on.cloud_field_mean) & set(mocap_off.cloud_field_mean)):
        out.append(_compare(mocap_on.cloud_field_mean[f], mocap_off.cloud_field_mean[f], f"mean point {f}"))
    return out


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
