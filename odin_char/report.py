"""Markdown report + figures mirroring the protocol's checklist."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from .errors import AXES  # noqa: E402


def _box(ok) -> str:
    if ok is None:
        return "[ ] (not run)"
    return "[x] PASS" if ok else "[ ] **FAIL**"


def _f(x, fmt="{:.4g}"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return fmt.format(x)


def _figures(d: Path, figs: dict, eta: np.ndarray, tab) -> list[str]:
    d.mkdir(exist_ok=True)
    made = []
    if "latency" in figs:
        sp, ea, yr, ey, lat = figs["latency"]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(sp[::5], ea[::5] * 100, ".", ms=2, alpha=0.4)
        xs = np.linspace(0, sp.max(), 10)
        ax[0].plot(xs, (lat.along.intercept - lat.along.tau * xs) * 100, "r")
        ax[0].set(xlabel="speed (m/s)", ylabel="along-track error (cm)", title=f"tau_d={lat.along.tau*1000:.1f} ms")
        ax[1].plot(yr[::5], np.degrees(ey[::5]), ".", ms=2, alpha=0.4)
        if np.isfinite(lat.yaw.tau):
            xs = np.linspace(yr.min(), yr.max(), 10)
            ax[1].plot(xs, np.degrees(lat.yaw.intercept - lat.yaw.tau * xs), "r")
        ax[1].set(xlabel="yaw rate (rad/s)", ylabel="yaw error (deg)", title=f"tau_d={lat.yaw.tau*1000:.1f} ms")
        fig.tight_layout()
        fig.savefig(d / "latency.png", dpi=110)
        plt.close(fig)
        made.append("latency.png")
    if "fogm" in figs:
        fits = figs["fogm"]
        fig, ax = plt.subplots(2, 6, figsize=(18, 6))
        for i, (a, f) in enumerate(zip(AXES, fits)):
            lag = np.arange(len(f.acf)) * (1 / 10)
            ax[0, i].plot(lag, f.acf, label="empirical")
            ax[0, i].plot(lag, f.acf_model, "--", label="AR(1)")
            ax[0, i].set_title(f"{a} ACF, tau_c={f.tau_c:.2f}s")
            e = f.innovations
            if np.isfinite(f.nu):
                q = stats.t.ppf(np.linspace(0.005, 0.995, 200), f.nu, 0, f.scale)
            else:
                q = stats.norm.ppf(np.linspace(0.005, 0.995, 200), 0, f.scale)
            ax[1, i].plot(q, np.quantile(e, np.linspace(0.005, 0.995, 200)), ".")
            ax[1, i].plot(q, q, "k--", lw=0.8)
            ax[1, i].set_title(f"QQ innovations ({'t nu=%.1f' % f.nu if np.isfinite(f.nu) else 'Gaussian'})")
        ax[0, 0].legend()
        fig.tight_layout()
        fig.savefig(d / "fogm.png", dpi=100)
        plt.close(fig)
        made.append("fogm.png")
    if figs.get("field") is not None:
        fld = figs["field"]
        b = fld.bins
        fig, ax = plt.subplots(1, 6, figsize=(20, 3.6))
        for i, a in enumerate(AXES):
            sc = ax[i].scatter(b.centers[:, 0], b.centers[:, 1], c=b.mean[:, i] * (100 if i < 3 else 57.3), s=30, cmap="coolwarm")
            plt.colorbar(sc, ax=ax[i])
            ax[i].set_title(f"{a} bin mean ({'cm' if i < 3 else 'deg'})" + ("" if fld.enabled[i] else " [off]"))
            ax[i].set_aspect("equal")
        fig.tight_layout()
        fig.savefig(d / "static_field.png", dpi=100)
        plt.close(fig)
        made.append("static_field.png")
    if "validation" in figs:
        v = figs["validation"]
        fig, ax = plt.subplots(2, 6, figsize=(18, 6))
        for i, a in enumerate(AXES):
            lag = np.arange(v["real_acf"].shape[1]) * v["dt"]
            ax[0, i].fill_between(lag, v["acf_lo"][i], v["acf_hi"][i], alpha=0.3, label="sim 5-95%")
            ax[0, i].plot(lag, v["real_acf"][i], "k", label="real")
            ax[0, i].set_title(f"{a} ACF")
            ax[1, i].fill_between(v["qs"], v["q_lo"][:, i], v["q_hi"][:, i], alpha=0.3)
            ax[1, i].plot(v["qs"], v["real_q"][:, i], "k")
            ax[1, i].set_title(f"{a} quantiles")
        ax[0, 0].legend()
        fig.tight_layout()
        fig.savefig(d / "validation.png", dpi=100)
        plt.close(fig)
        made.append("validation.png")
    if "time_offset" in figs:
        lags, corr = figs["time_offset"]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(lags * 1000, corr)
        ax.set(xlabel="lag (ms)", ylabel="corr |omega|")
        fig.tight_layout()
        fig.savefig(d / "time_offset.png", dpi=100)
        plt.close(fig)
        made.append("time_offset.png")
    return made


def write_report(res_dir: Path, out: dict, model, figs: dict, eta, tab):
    made = _figures(res_dir / "figures", figs, eta, tab)
    L = ["# Odin1 Tier 0 characterization report", ""]

    p0 = out.get("phase0", {})
    L += ["## Phase 0: IR interference preflight", f"- {_box(p0.get('pass'))} no significant difference in either direction"]
    for side in ("odin_side", "mocap_side"):
        for r in p0.get(side, []):
            if "p" in r:
                L.append(f"  - {side}: {r['metric']}: on {_f(r['on_mean'])} vs off {_f(r['off_mean'])} (p={_f(r['p'])}, rel {_f(r['rel_diff'])})"
                         + (" **SIGNIFICANT**" if r["significant"] else ""))
            else:
                L.append(f"  - {side}: {r['metric']}: {r.get('status')}")

    p1 = out.get("phase1", {})
    he = p1.get("hand_eye", {})
    L += ["", "## Phase 1: calibration",
          f"- IMU-mocap time offset: {_f(p1.get('time_offset_s', np.nan) * 1000 if 'time_offset_s' in p1 else None)} ms (peak corr {_f(p1.get('time_offset_peak_corr'))})",
          f"- {_box(he.get('pass'))} hand-eye residual rot {_f(he.get('rot_rms_deg'))} deg (< 0.5), trans {_f(he.get('trans_rms_mm'))} mm (< 5)",
          f"  - parameters constrained mainly by the CAD prior (not observable from planar motion): {he.get('prior_dominated_parameters')}",
          f"- T_WM (frozen, one alignment for all runs): pos rms {_f(p1.get('T_WM', {}).get('pos_rms_m'))} m, rot rms {_f(p1.get('T_WM', {}).get('rot_rms_deg'))} deg"]

    p2 = out.get("phase2", {})
    lat = p2.get("latency", {})
    sf = p2.get("static_field", {})
    L += ["", "## Phase 2: repeated path",
          f"- Latency tau_d = **{_f(lat.get('tau_d', np.nan) * 1000)} ms** +- {_f(lat.get('tau_d_stderr', np.nan) * 1000)} ({lat.get('reference')}-to-truth)",
          f"  - from along-track vs speed: {_f(lat.get('along_track', {}).get('tau', np.nan) * 1000)} ms; from yaw vs yaw rate: {_f(lat.get('yaw', {}).get('tau', np.nan) * 1000)} ms",
          f"- {_box(sf.get('kept'))} static field predicts held-out session (R^2 > 0.5 on >= 1 axis): "
          + ", ".join(f"{a} {_f(sf.get('cross_session_r2', {}).get(a))}" for a in AXES)
          + ("" if sf.get("kept") else "  -> eps_s = 0"),
          "", "| axis | sigma | tau_c (s) | nu | Ljung-Box p | ACF exp. | t over Gaussian | run >= 50 tau_c |",
          "|---|---|---|---|---|---|---|---|"]
    for a, s in p2.get("fogm_moving", {}).items():
        unit = 100 if a in AXES[:3] else 57.2958
        L.append(f"| {a} | {_f(s['sigma'] * unit)} {'cm' if unit == 100 else 'deg'} | {_f(s['tau_c'])} | {_f(s['nu'])} | "
                 f"{_f(s['ljung_box_p'])} {'ok' if s['ljung_box_p'] > 0.05 else '**corr. left**'} | "
                 f"{'slow tail -> 2nd FOGM' if s['slow_acf_tail'] else 'ok'} | {_f(s['loglik_gain_t'], '{:.1f}')} | "
                 f"{'ok' if s['run_length_ok'] else '**add laps** (%.0f)' % s['min_run_over_tau_c']} |")
    t1 = p2.get("tier1_speed_screen")
    if t1:
        L.append(f"- Tier 1 speed screen: {'**FLAG** (P differs > 30% across speeds)' if t1['flag'] else 'no flag'}: "
                 + ", ".join(f"{a} x{_f(v, '{:.2f}')}" for a, v in t1["max_over_min"].items()))

    p3 = out.get("phase3", {})
    L += ["", "## Phase 3: stationary holds"]
    if "locations" in p3:
        for loc, v in p3["locations"].items():
            if "fogm" in v:
                u = v["odom_to_map_updates"]
                L.append(f"- {loc}: sigma along/cross/yaw = {_f(v['fogm']['along']['sigma']*100)} cm / {_f(v['fogm']['cross']['sigma']*100)} cm / "
                         f"{_f(np.degrees(v['fogm']['yaw']['sigma']))} deg; odom->map updates: median {_f(u['median_interval_s'])} s, max {_f(u['max_interval_s'])} s"
                         + (" **stops while still**" if u["stops_when_still"] else ""))
        s = p3.get("tier1_location_screen")
        if s:
            L.append(f"- Tier 1 location screen: {'**FLAG** (P varies > 2x)' if s['flag'] else 'no flag'}")
        L.append(f"- Tier 1 stationary gating: {'**FLAG**' if p3.get('tier1_stationary_gating_flag') else 'no flag'}")
    else:
        L.append(f"- {p3.get('status')}")

    p4 = out.get("phase4", {})
    L += ["", "## Phase 4: latency cross-check",
          f"- {_box(p4.get('pass'))} xcorr tau_d {_f(p4.get('tau_d_xcorr', np.nan) * 1000)} ms vs Phase 2 {_f(p4.get('tau_d_phase2', np.nan) * 1000)} ms "
          f"(diff {_f(p4.get('difference_ms'))} ms, must be < 10)"]

    p5 = out.get("phase5", {})
    L += ["", "## Phase 5: events"]
    if "init" in p5:
        i = p5["init"]
        L += [f"- INIT: {i['n_success']}/{i['n']} fixes; lambda_fix = {_f(i['lambda_fix'])} /s (KS p {_f(i['ks_p'])}); "
              f"min fix delay {_f(i['fix_delay_min'])} s, rate after delay {_f(i['lambda_fix_shifted'])} /s (KS p {_f(i['ks_p_shifted'])})",
              f"- p(success) logistic: {i['logistic']}", f"- p_wrong (fix > 20 cm or > 5 deg): {_f(i['p_wrong'])}"]
    if "lost_natural" in p5:
        n = p5["lost_natural"]
        L.append(f"- Natural LOST: {n['n']} events in {_f(n['T'] / 3600)} h tracked -> "
                 + (f"lambda_lost <= {_f(n['lambda_lost_upper95'])} /s (rule of three)" if n["n"] == 0 else f"lambda_lost = {_f(n['lambda_lost'])} /s"))
    if "lost_induced" in p5:
        r = p5["lost_induced"]
        L.append(f"- Induced outages: {r['n_recovered']}/{r['n_trials']} recovered, lambda_rec = {_f(r['lambda_rec'])} /s")
        for k, v in r["by_cover_duration"].items():
            L.append(f"  - {k} s cover: max error {_f(v['max_err_mean'])} m, growth {_f(v['err_growth_mps'])} m/s, recover {v['recover_s']}")

    p6 = out.get("phase6", {})
    L += ["", "## Phase 6: hold-out validation"]
    if "metrics" in p6:
        m = p6["metrics"]
        L.append(f"- held-out session: {p6['held_out_session']}")
        L.append(f"- {_box(all(v['pass'] for v in m['rmse'].values()))} RMSE per axis inside simulated 5-95% band")
        L.append(f"- {_box(all(v['pass'] for v in m['cdf'].values()))} error CDF / QQ (KS p: "
                 + ", ".join(f"{a} {_f(v['ks_p'])}" for a, v in m["cdf"].items()) + ")")
        L.append(f"- {_box(all(v['pass'] for v in m['acf'].values()))} ACF inside band to 5 tau_c")
        L.append(f"- {_box(m['rpe']['pass'])} RPE vs segment length inside band")
        L.append(f"- {_box(m['events']['pass'])} event counts plausible ({m['events']['real_lost']} real vs {_f(m['events']['expected'])} expected)")
        L.append("")
        tier1 = [p2.get("tier1_speed_screen", {}).get("flag"), p3.get("tier1_location_screen", {}).get("flag"), p3.get("tier1_stationary_gating_flag")]
        if p6["pass"] and not any(tier1):
            L.append("**Tier 0 passes and no Tier 1 screen triggered: stop here.**")
        else:
            L.append("**Tier 0 is not sufficient.** Next terms to add:")
            for k, v in p6.get("tier1_suggestions", {}).items():
                L.append(f"- {k} failed -> {v}")
            if tier1[0]:
                L.append("- speed screen -> speed-dependent drift and registration terms")
            if tier1[1]:
                L.append("- location screen -> raycast registration covariance H(p)")
            if tier1[2]:
                L.append("- stationary updates -> stationary correction gating")
    else:
        L.append(f"- {p6.get('status')}")

    L += ["", "## Outputs", "- `model.yaml`: fitted Tier 0 parameters", "- `static_field_grid.npz`: eps_s on a 0.1 m grid (map frame)",
          "- `calibration.json`: time offset, hand-eye X, frozen T_WM", "- `results.json`: every number above", ""]
    L += [f"![{f}](figures/{f})" for f in made]
    (res_dir / "report.md").write_text("\n".join(L) + "\n")
