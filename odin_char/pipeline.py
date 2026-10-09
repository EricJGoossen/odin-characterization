"""Run the full Tier 0 analysis over a campaign directory written by `odin_char.robot.campaign`.

    python -m odin_char.pipeline <campaign_dir> [--config campaign.yaml]

Outputs in <campaign_dir>/results: model.yaml, static_field_grid.npz, calibration.json,
report.md (with the protocol's pass/fail checks) and figures/.
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from . import events as ev
from . import preflight
from . import registration as reg
from .bagio import BagData, read_bag
from .calib import (
    align_map_to_mocap,
    hand_eye,
    imu_mocap_time_offset,
    pose_mocap_time_offset,
    relative_motion_pairs,
)
from .errors import AXES, Calibration, ErrorTable, compute_errors
from .fogm import fit_fogm
from .latency import regress_latency, xcorr_latency
from .model import AxisParams, Tier0Model
from .se3 import Pose
from .sensors import load_sensor, resolve_config
from .static_field import StaticField, cross_session_r2, leave_one_session_out_residual
from .validate import TIER1_HINTS, validate


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


class Campaign:
    def __init__(self, root: Path, cfg: dict):
        self.root = root
        self.cfg = cfg = resolve_config(cfg)
        self.sensor = load_sensor(cfg)
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.runs = {r["id"]: r for r in self.manifest["runs"]}
        self.cache = root / "cache"
        self.cache.mkdir(exist_ok=True)
        a = cfg.get("analysis", {})
        self.rate = a.get("consumer_rate_hz", 10.0)
        self.dt = 1.0 / self.rate
        self.use_arrival = a.get("latency_reference", "stamp") == "arrival"
        self.grid_res = a.get("static_field_grid_m", 0.35)
        self.truth_lowpass_hz = a.get("truth_lowpass_hz")
        self.use_registration = bool(a.get("registration_term", True))
        self.min_bin = a.get("static_field_min_samples", 5)

    def runs_of(self, phase: str) -> list[dict]:
        # runs marked "excluded" (e.g. an abandoned session) stay in the manifest for the record;
        # their driver start can still be an INIT trial (looked up via self.runs)
        return [r for r in self.manifest["runs"] if r["phase"] == phase and not r.get("excluded")]

    def bag(self, run: dict) -> BagData:
        mem = self.__dict__.setdefault("_bags", {})
        if run["id"] in mem:
            return mem[run["id"]]
        c = self.cache / f"run{run['id']}.pkl"
        if c.exists():
            b = pickle.loads(c.read_bytes())
        else:
            b = read_bag(self.root / run["bag"], self.cfg)
            b.tf = dict(b.tf)
            b.cloud_field_mean = dict(b.cloud_field_mean)
            c.write_bytes(pickle.dumps(b))
        b.mocap_lowpass_hz = self.truth_lowpass_hz
        mem[run["id"]] = b
        return b

    def meta(self, run: dict) -> dict:
        return {k: run.get(k) for k in ("phase", "session", "speed_class", "location", "name")}

    def errors(self, runs: list[dict], calib: Calibration, tau_d: float) -> ErrorTable:
        tabs = [
            compute_errors(self.bag(r), self.sensor, calib, r["id"], self.meta(r), tau_d, self.rate, self.use_arrival)
            for r in runs
        ]
        return ErrorTable.concat([t for t in tabs if len(t)])

    def reloc_segments(self, calib: Calibration) -> list:
        """Registration (odom->map) stretches of every run, linked into relocalization instances (cached)."""
        key = round(calib.time_offset, 9)
        if getattr(self, "_segs_key", None) == key:
            return self._segs
        segs = []
        for r in self.manifest["runs"]:
            if r["phase"] == "preflight" or r.get("excluded"):
                continue
            try:
                s = self.sensor.registration(self.bag(r), self.use_arrival)
            except KeyError:
                continue
            t = s.t + (0.0 if self.use_arrival else calib.time_offset)
            segs += reg.split_odom_map(r["id"], t, s.pose.p, s.pose.R)
        self._segs, self._segs_key = reg.link_instances(segs), key
        return self._segs

    def instances(self, tab: ErrorTable, calib: Calibration) -> np.ndarray:
        return reg.label(tab, self.reloc_segments(calib))

    def debiased(self, tab: ErrorTable, calib: Calibration) -> ErrorTable:
        """tab with each relocalization's mean map-frame offset removed (no-op when the
        registration term is off)."""
        if not self.use_registration or len(tab) == 0:
            return tab
        inst = self.instances(tab, calib)
        return tab.with_err(reg.debias(tab, inst, reg.instance_biases(tab, inst, self.dt)))

    def map_updates(self, run: dict, calib: Calibration) -> np.ndarray:
        b = self.bag(run)
        try:
            s = self.sensor.registration(b, self.use_arrival)
        except KeyError:
            return np.zeros(0)
        return s.t + (0.0 if self.use_arrival else calib.time_offset)


# ------------------------------------------------------------------------------------ steps
def step_preflight(c: Campaign, out: dict):
    # campaigns recorded before the sensor split call the sensor conditions odin_on / odin_off
    legacy = {"odin_on": "sensor_on", "odin_off": "sensor_off"}
    runs = {legacy.get(r.get("condition"), r.get("condition")): r for r in c.runs_of("preflight")}
    res = {}
    if "mocap_on" in runs and "mocap_off" in runs:
        res["sensor_side"] = preflight.sensor_side(c.bag(runs["mocap_on"]), c.bag(runs["mocap_off"]))
    if "sensor_on" in runs and "sensor_off" in runs:
        res["mocap_side"] = preflight.mocap_side(
            c.bag(runs["sensor_on"]), c.bag(runs["sensor_off"]), c.cfg.get("preflight", {}).get("expected_markers")
        )
    if res:
        res["pass"] = all(preflight.passed(v) for v in res.values())
    out["phase0"] = res or {"status": "no preflight runs"}


def step_calibration(c: Campaign, out: dict) -> Calibration:
    fixed = c.cfg.get("calibration_override")
    if fixed:
        X = Pose.from_vec(np.array(fixed["X"]))
        out["phase1"] = {"status": "calibration taken from config override"}
        return Calibration(float(fixed["time_offset"]), X)
    runs = c.runs_of("calib")
    if not runs:
        raise SystemExit("no Phase 1 calibration run in manifest and no calibration_override in config")
    b = c.bag(runs[-1])
    mocap = b.mocap_series()
    to = imu_mocap_time_offset(np.array(b.imu_t), np.array(b.imu_gyro), mocap)
    odom_raw = c.sensor.odometry(b)
    odom_lag = pose_mocap_time_offset(odom_raw, mocap)
    odom = odom_raw.shifted(odom_lag.offset)
    A, B = relative_motion_pairs(mocap, odom)
    pr = c.cfg.get("hand_eye_prior")
    prior = Pose.from_vec(np.array(pr["X"])) if pr else None
    he = hand_eye(A, B, prior, **({k: pr[k] for k in ("prior_sigma_t", "prior_sigma_rot_deg") if pr and k in pr}))
    out["phase1"] = {
        "time_offset_s": to.offset,
        "time_offset_peak_corr": to.peak_corr,
        "odometry_lag_vs_mocap_s": odom_lag.offset - to.offset,
        "hand_eye": {
            "X_T_base_to_imu": he.X.to_vec(),
            "rot_rms_deg": he.rot_rms_deg,
            "trans_rms_mm": he.trans_rms_mm,
            "n_pairs": he.n_pairs,
            "prior_dominated_parameters": he.weak_directions,
            "motion_only_std": he.motion_only_std,
            "pass": he.passed,
        },
    }
    out["_figs"]["time_offset"] = (to.lags, to.corr)
    return Calibration(to.offset, he.X)


def step_alignment(c: Campaign, calib: Calibration, tau_d: float, out: dict | None,
                   runs: list[dict] | None = None) -> Calibration:
    ident = Calibration(calib.time_offset, calib.X, Pose.identity())
    tab = c.errors(c.runs_of("phase2") if runs is None else runs, ident, tau_d)  # true_M == T_WI when T_WM = I
    tab = tab.select(tab.map_age < 1.0)  # TRACK only
    al = align_map_to_mocap(tab.est_M, tab.true_M)
    if out is None:
        return Calibration(calib.time_offset, calib.X, al.T_WM)
    out.setdefault("phase1", {})["T_WM"] = {
        "vec": al.T_WM.to_vec(),
        "pos_rms_m": al.pos_rms,
        "rot_rms_deg": al.rot_rms_deg,
        "n": al.n,
        "latency_used_s": tau_d,
    }
    return Calibration(calib.time_offset, calib.X, al.T_WM)


def fogm_block(fits) -> dict:
    return {a: f.summary() for a, f in zip(AXES, fits)}


def run_durations(tab: ErrorTable) -> list[float]:
    return [float(np.ptp(tab.t[tab.run == r])) for r in np.unique(tab.run)]


def fit_model(c: Campaign, calib: Calibration, p2_runs: list[dict], out: dict | None, label: str,
              field_axes=None) -> tuple:
    """Latency -> static field -> FOGM on the given Phase 2 runs. Returns (model, corrected table, raw table, info).

    Each relocalization's constant map-frame offset is removed first (registration.py): it is a
    separate model term, and left in it would look like a static field that changes between
    sessions and like extra slow FOGM variance."""
    raw = c.errors(p2_runs, calib, 0.0)
    raw = c.debiased(raw.select(raw.map_age < 1.0), calib)
    lat = regress_latency(raw)
    tau = lat.tau_d
    tab = c.errors(p2_runs, calib, tau)
    tab = c.debiased(tab.select(tab.map_age < 1.0), calib)

    a = c.cfg.get("analysis", {})
    r2, folds = cross_session_r2(tab, c.grid_res, c.min_bin)
    thr = a.get("static_field_r2_threshold", 0.5)
    passing = np.nan_to_num(r2, nan=-np.inf) > thr
    if field_axes is not None:
        enabled = np.asarray(field_axes, bool)  # decided on all sessions (hold-out refit)
    elif a.get("static_field_per_axis", True):
        enabled = passing
    else:
        enabled = np.ones(6, bool) if passing.any() else np.zeros(6, bool)
    field = StaticField.fit(tab, c.grid_res, c.min_bin, axes_enabled=enabled) if enabled.any() else None
    if field is not None:
        field.r2 = r2
    eta = leave_one_session_out_residual(tab, field, c.grid_res, c.min_bin) if field else tab.err.copy()

    nominal = np.array([tab.meta[r].get("speed_class") in (None, "1x") for r in tab.run])
    tn = tab.select(nominal)
    fits = fit_fogm(eta[nominal], tn.segments(c.dt), c.dt, a.get("t_loglik_threshold", 10.0), run_durations(tn))
    model = Tier0Model(c.dt, tau, [AxisParams.from_fit(f) for f in fits], None, field, tau_d_stderr=lat.tau_d_stderr)

    info = {
        "latency": {
            "along_track": asdict(lat.along),
            "yaw": asdict(lat.yaw),
            "tau_d": lat.tau_d,
            "tau_d_stderr": lat.tau_d_stderr,
            "reference": "arrival" if c.use_arrival else "stamp",
        },
        "static_field": {
            "cross_session_r2": dict(zip(AXES, r2)),
            "folds": folds,
            "kept": bool(enabled.any()),
            "enabled_axes": [ax for ax, e in zip(AXES, enabled) if e],
        },
        "fogm_moving": fogm_block(fits),
    }
    # Tier 1 speed screen: latency-corrected P per speed class
    per_speed = {}
    for sc in sorted({m.get("speed_class") for m in tab.meta.values() if m.get("speed_class")}):
        m = np.array([tab.meta[r].get("speed_class") == sc for r in tab.run])
        if m.sum() > 100:
            per_speed[sc] = np.var(eta[m], axis=0)
    if len(per_speed) >= 2:
        P = np.array(list(per_speed.values()))
        ratio = P.max(0) / np.maximum(P.min(0), 1e-18)
        info["tier1_speed_screen"] = {
            "P_by_speed": {k: dict(zip(AXES, v)) for k, v in per_speed.items()},
            "max_over_min": dict(zip(AXES, ratio)),
            "flag": bool((ratio > 1.3).any()),
        }
    if out is not None:
        out[label] = info
        out["_figs"]["latency"] = (raw.speed, raw.err[:, 0], raw.yaw_rate, raw.err[:, 5], lat)
        out["_figs"]["fogm"] = fits
        out["_figs"]["field"] = field
    return model, tab, raw, info, eta, fits


def step_stationary(c: Campaign, calib: Calibration, model: Tier0Model, out: dict):
    runs = c.runs_of("phase3")
    if not runs:
        out["phase3"] = {"status": "no stationary runs"}
        return
    res, Ps, all_eta, all_segs, durations = {}, [], [], [], []
    offset = 0
    for r in runs:
        tab = c.errors([r], calib, model.tau_d)
        tab = tab.select(tab.map_age < 1.0)
        if len(tab) < 50:
            res[r.get("location", r["name"])] = {"status": "too little tracked data"}
            continue
        # At a fixed spot the static field is the hold's mean error (holds are >> tau_c); the
        # Phase 2 GP is only trustworthy where the figure-8 went, so report both.
        spot_mean = tab.err.mean(0)
        gp_pred = model.static_field.predict(tab.pos_M[:1])[0] if model.static_field else np.zeros(6)
        eta = tab.err - spot_mean
        segs = tab.segments(c.dt)
        fits = fit_fogm(eta, segs, c.dt, run_durations=run_durations(tab))
        upd = c.map_updates(r, calib)
        upd = upd[(upd > tab.t[0]) & (upd < tab.t[-1])]
        gaps = np.diff(upd) if len(upd) > 1 else np.array([np.inf])
        res[r.get("location", r["name"])] = {
            "fogm": fogm_block(fits),
            "static_error_at_spot": dict(zip(AXES, spot_mean)),
            "phase2_field_prediction": dict(zip(AXES, gp_pred)),
            "odom_to_map_updates": {"n": int(len(upd)), "median_interval_s": float(np.median(gaps)),
                                     "max_interval_s": float(np.max(gaps)),
                                     "stops_when_still": bool(len(upd) < 2 or np.max(gaps) > 1.0)},
        }
        Ps.append([f.P for f in fits])
        all_eta.append(eta)
        all_segs += [s + offset for s in segs]
        durations += run_durations(tab)
        offset += len(tab)
    out["phase3"] = {"locations": res}
    if Ps:
        Ps = np.array(Ps)
        spread = Ps.max(0) / np.maximum(Ps.min(0), 1e-18)
        out["phase3"]["tier1_location_screen"] = {"max_over_min": dict(zip(AXES, spread)), "flag": bool((spread > 2).any())}
        out["phase3"]["tier1_stationary_gating_flag"] = any(
            v.get("odom_to_map_updates", {}).get("stops_when_still") for v in res.values()
        )
        pooled = fit_fogm(np.concatenate(all_eta), all_segs, c.dt, run_durations=durations)
        out["phase3"]["fogm_stationary_pooled"] = fogm_block(pooled)
        model.stationary = [AxisParams.from_fit(f) for f in pooled]


def registration_fit(c: Campaign, calib: Calibration, model: Tier0Model, exclude_runs=(), min_samples: int = 30) -> dict:
    """sigma_b from the mean error of every relocalization instance in the campaign (all phases:
    sessions, holds, latency, INIT trials and outage runs each start from their own fix).
    Instances that contain any of `exclude_runs` are left out (hold-out validation)."""
    runs = [r for r in c.manifest["runs"]
            if r["phase"] not in ("preflight", "calib", "mapping") and not r.get("excluded")]
    tabs = []
    for r in runs:
        try:
            t = c.errors([r], calib, model.tau_d)
        except (KeyError, ValueError):
            continue
        t = t.select(t.map_age < 1.0)
        if len(t):
            tabs.append(t)
    if not tabs:
        return {"status": "no tracked runs", "n_instances": 0}
    tab = ErrorTable.concat(tabs)
    inst = c.instances(tab, calib)
    bad = {int(k) for k in np.unique(inst[np.isin(tab.run, list(exclude_runs))])}
    biases = [b for b in reg.instance_biases(tab, inst, c.dt, min_samples) if b.instance not in bad]
    P = np.array([p.P for p in model.moving])
    tau = np.array([-c.dt / np.log(min(max(p.a, 1e-6), 1 - 1e-9)) for p in model.moving])
    fit = reg.fit_sigma(biases, P, tau)
    for b in fit.get("instances", []) + fit.get("wrong_fixes", []):
        b["run_names"] = [c.runs[r]["name"] for r in b["runs"]]
    return fit


def step_registration(c: Campaign, calib: Calibration, model: Tier0Model, out: dict):
    if not c.use_registration:
        out["registration"] = {"status": "disabled (analysis.registration_term: false)"}
        return
    fit = registration_fit(c, calib, model)
    out["registration"] = fit
    if "sigma" in fit:
        model.registration_sigma = np.array([fit["sigma"][a] for a in reg.MAP_AXES])
        model.registration = fit
        model.events["p_wrong_fix"] = fit["p_wrong_fix"]
        model.events["wrong_fix_offsets"] = [w["mean"] for w in fit["wrong_fixes"]]


def step_latency_check(c: Campaign, calib: Calibration, tau_d: float, out: dict):
    runs = c.runs_of("phase4")
    if not runs:
        out["phase4"] = {"status": "no latency runs"}
        return
    taus = {"position": [], "yaw": []}
    for r in runs:
        x = xcorr_latency(c.bag(r), c.sensor, calib, c.use_arrival)
        if np.isfinite(x.tau_position):
            taus["position"].append(x.tau_position)
        if np.isfinite(x.tau_yaw):
            taus["yaw"].append(x.tau_yaw)
    allt = taus["position"] + taus["yaw"]
    t4 = float(np.median(allt)) if allt else np.nan
    out["phase4"] = {
        "tau_position_s": taus["position"],
        "tau_yaw_s": taus["yaw"],
        "tau_d_xcorr": t4,
        "tau_d_phase2": tau_d,
        "difference_ms": (t4 - tau_d) * 1000,
        "pass": bool(abs(t4 - tau_d) < 0.010),
    }


def refine_cover_times(b: BagData, t_cover: float, t_uncover: float, calib: Calibration) -> tuple[float, float]:
    """The announced cover/uncover times include the operator's reaction time; when the cloud was
    recorded, use the frames where the sensor's point count collapses instead."""
    if len(b.cloud_n) < 20:
        return t_cover, t_uncover
    t = np.asarray(b.cloud_t) + calib.time_offset
    n = np.asarray(b.cloud_n, float)
    win = (t > t_cover - 2.0) & (t < t_uncover + 2.0)
    low = win & (n < 0.3 * np.median(n))
    if not low.any():
        return t_cover, t_uncover
    period = float(np.median(np.diff(t)))
    return float(t[low].min()), float(t[low].max() + period)


def natural_losses(c: Campaign, tab: ErrorTable, calib: Calibration) -> list:
    """LOST events in tracked data, minus deliberate driver restarts (INIT trials and session
    starts relaunch the sensor inside a recorded run; odom->map stops then too, but nothing was lost)."""
    restarts = [(tr["t_launch"] - 15.0, (tr.get("t_log_success") or tr["t_launch"] + tr["timeout"]) + 2.0)
                for tr in c.manifest.get("init_trials", [])]
    ups = {int(r): c.map_updates(c.runs[int(r)], calib) for r in np.unique(tab.run)}
    return [e for e in ev.detect_lost(tab, ups) if not any(a <= e.t <= b for a, b in restarts)]


def outage_reregistration(c: Campaign, calib: Calibration, run: dict, tab: ErrorTable, t_cover: float,
                          t_uncover: float) -> dict:
    """What a covered window does to the pose output: how long the odometry stops, and how far the
    map-frame offset moved between before the cover and after output resumes (Odin, for one, keeps
    its odom->map, so dead-reckoning drift during the outage stays in the estimate)."""
    b = c.bag(run)
    oi = c.sensor.odometry(b, c.use_arrival).t
    oi = oi + (0.0 if c.use_arrival else calib.time_offset)
    win = oi[(oi > t_cover - 1.0) & (oi < t_uncover + 15.0)]
    gaps = np.diff(win)
    gap = float(gaps.max()) if len(gaps) else np.nan
    resume = float(win[1:][np.argmax(gaps)]) if len(gaps) else t_uncover
    eM = reg.track_to_map(tab.err, tab.heading)
    pre = (tab.t > t_cover - 4.0) & (tab.t < t_cover - 0.3)
    post = (tab.t > max(resume, t_uncover) + 1.0) & (tab.t < max(resume, t_uncover) + 5.0)
    if pre.sum() < 5 or post.sum() < 5:
        return {"run": run["name"], "output_gap_s": gap, "jump_pos_m": np.nan, "jump_rot_deg": np.nan}
    d = eM[post].mean(0) - eM[pre].mean(0)
    return {"run": run["name"], "output_gap_s": gap, "jump_pos_m": float(np.linalg.norm(d[:3])),
            "jump_rot_deg": float(np.degrees(np.linalg.norm(d[3:]))), "jump_map": dict(zip(reg.MAP_AXES, d))}


def step_events(c: Campaign, calib: Calibration, model: Tier0Model, out: dict, tracked: list[ErrorTable]):
    res: dict = {}
    trials = []
    for tr in c.manifest.get("init_trials", []):
        run = c.runs[tr["run"]]
        b = c.bag(run)
        fix_t = None
        try:
            # the bag can start before the driver restart: odom->map from the previous driver is not this fix
            s = c.sensor.registration(b, use_arrival=True)
            after = s.t[s.t > tr["t_launch"]]
            fix_t = float(after[0]) if len(after) else None
        except KeyError:
            pass
        if tr.get("t_log_success") is not None:
            fix_t = tr["t_log_success"] if fix_t is None else min(fix_t, tr["t_log_success"])
        ttf = None if fix_t is None else fix_t - tr["t_launch"]
        if ttf is not None and ttf > tr["timeout"]:
            ttf = None
        pos_err = yaw_err = None
        if ttf is not None:
            try:
                tab = c.errors([run], calib, model.tau_d)
                tab = tab.select((tab.t > fix_t + 0.5) & (tab.t < fix_t + 2.0))
                if len(tab):
                    pos_err = float(np.median(np.linalg.norm(tab.err[:, :3], axis=1)))
                    yaw_err = float(np.degrees(np.median(tab.err[:, 5])))
            except (KeyError, ValueError):
                pass
        trials.append(ev.InitTrial(tr["offset_m"], tr["heading_deg"], bool(tr["moving"]), ttf, tr["timeout"], pos_err, yaw_err))
    if trials:
        ir = ev.analyze_init(trials)
        res["init"] = asdict(ir)
        res["init"]["trials"] = [asdict(t) | {"correct": t.correct} for t in trials]
        model.events["lambda_fix"] = ir.lambda_fix
        model.events["fix_delay_min_s"] = ir.fix_delay_min
        model.events["lambda_fix_after_delay"] = ir.lambda_fix_shifted
        model.events["p_success_logistic"] = ir.logistic
        model.events["p_wrong"] = ir.p_wrong

    # natural losses over all tracked Phase 2/3 data
    n_lost, T = 0, 0.0
    lost_list = []
    for tab in tracked:
        found = natural_losses(c, tab, calib)
        lost_list += [asdict(e) for e in found]
        n_lost += len(found)
        T += sum(run_durations(tab))
    lr = ev.lost_rate(n_lost, T)
    res["lost_natural"] = lr | {"events": lost_list}
    model.lambda_lost = lr["lambda_lost"]
    model.events["lambda_lost"] = lr["lambda_lost"]
    if "lambda_lost_upper95" in lr:
        model.events["lambda_lost_upper95"] = lr["lambda_lost_upper95"]

    outs, rereg = [], []
    for o in c.manifest.get("outages", []):
        run = c.runs[o["run"]]
        tab = c.errors([run], calib, model.tau_d)
        if len(tab):
            t0, t1 = refine_cover_times(c.bag(run), o["t_cover"], o["t_uncover"], calib)
            outs.append(ev.analyze_outage(tab, t0, t1, c.map_updates(run, calib)))
            rereg.append(outage_reregistration(c, calib, run, tab, t0, t1) | {"cover_s": o["cover_s"]})
    if outs:
        rr = ev.recovery_rate(outs)
        res["lost_induced"] = rr
        model.events["lambda_rec"] = rr["lambda_rec"]
    if rereg:
        res["outage_reregistration"] = rereg
        by = {}
        for r in rereg:
            by.setdefault(r["cover_s"], []).append(r)
        model.events["outage_effect_by_cover_s"] = {
            str(k): {"pose_output_gap_s": float(np.nanmean([r["output_gap_s"] for r in v])),
                     "offset_jump_m": float(np.nanmean([r["jump_pos_m"] for r in v])),
                     "offset_jump_deg": float(np.nanmean([r["jump_rot_deg"] for r in v]))}
            for k, v in sorted(by.items())}
    out["phase5"] = res or {"status": "no event data"}


def step_validation(c: Campaign, calib: Calibration, out: dict, full: Tier0Model):
    p2 = c.runs_of("phase2")
    sessions = sorted({r["session"] for r in p2}, key=str)
    if len(sessions) < 2:
        out["phase6"] = {"status": "need >= 2 Phase 2 sessions for hold-out validation"}
        return
    held = c.cfg.get("analysis", {}).get("holdout_session", sessions[-1])
    train = [r for r in p2 if r["session"] != held]
    test = [r for r in p2 if r["session"] == held]
    axes = full.static_field.enabled if full.static_field is not None else np.zeros(6, bool)
    # nothing fitted on the held-out session: T_WM is re-aligned on the training runs only (the
    # full-campaign T_WM averages the held-out registration in), and sigma_b leaves out every
    # relocalization the held-out runs belong to
    calib = step_alignment(c, calib, full.tau_d, None, runs=train)
    model, tab_train, *_ = fit_model(c, calib, train, None, "", field_axes=axes)
    model.stationary = full.stationary
    reg_scale = 1.0
    reg_note = "no registration term"
    if full.registration_sigma is not None:
        rf = registration_fit(c, calib, model, exclude_runs={r["id"] for r in test})
        if "sigma" in rf:
            model.registration_sigma = np.array([rf["sigma"][a] for a in reg.MAP_AXES])
            model.registration = rf
            # the held-out offset is measured against a T_WM that itself averages k training
            # registrations: var(b_test - mean(b_train)) = sigma^2 (1 + 1/k)
            it = c.instances(tab_train, calib)
            k = len([i for i in np.unique(it[it >= 0]) if (it == i).sum() >= 30])
            reg_scale = 1.0 + 1.0 / max(k, 1)
            reg_note = f"sigma_b from {rf['n_instances']} relocalizations outside the held-out session; T_WM from {k} -> variance x{reg_scale:.2f}"
    raw = c.errors(test, calib, 0.0)
    raw = raw.select(raw.map_age < 1.0)
    n_real = len(natural_losses(c, raw, calib))
    trt = c.errors(train, calib, model.tau_d)
    trt = trt.select(trt.map_age < 1.0)
    n_train = len(natural_losses(c, trt, calib))
    Ttrain = sum(run_durations(trt))
    model.lambda_lost = n_train / Ttrain if Ttrain else 0.0
    vr = validate(model, raw, n_mc=c.cfg.get("analysis", {}).get("mc_realizations", 50), real_lost_events=n_real,
                  instance=c.instances(raw, calib), registration_scale=reg_scale)
    curves = vr.metrics.pop("_curves")
    out["_figs"]["validation"] = curves
    out["phase6"] = {
        "held_out_session": held,
        "registration": reg_note,
        "pass": vr.passed,
        "failing_metrics": vr.failing,
        "tier1_suggestions": {m: TIER1_HINTS[m] for m in vr.failing},
        "metrics": vr.metrics,
    }


# ------------------------------------------------------------------------------------ main
def analyze(root: Path, cfg: dict) -> dict:
    c = Campaign(root, cfg)
    res_dir = root / "results"
    res_dir.mkdir(exist_ok=True)
    out: dict = {"_figs": {}}

    step_preflight(c, out)
    calib = step_calibration(c, out)
    if not c.runs_of("phase2"):
        raise SystemExit("no Phase 2 runs: cannot fit T_WM / Tier 0")
    # T_WM once with tau=0, estimate latency, then refit T_WM once with the latency and freeze it
    calib = step_alignment(c, calib, 0.0, out)
    raw0 = c.errors(c.runs_of("phase2"), calib, 0.0)
    tau0 = regress_latency(c.debiased(raw0.select(raw0.map_age < 1.0), calib)).tau_d
    calib = step_alignment(c, calib, tau0, out)

    model, tab, raw, info, eta, fits = fit_model(c, calib, c.runs_of("phase2"), out, "phase2")
    step_stationary(c, calib, model, out)
    step_registration(c, calib, model, out)
    step_latency_check(c, calib, model.tau_d, out)
    tracked = [tab]
    if c.runs_of("phase3"):
        t3 = c.errors(c.runs_of("phase3"), calib, model.tau_d)
        tracked.append(t3.select(t3.map_age < 1.0))
    step_events(c, calib, model, out, tracked)
    step_validation(c, calib, out, model)

    # ---- write outputs
    calib_out = {
        "time_offset_s": calib.time_offset,
        "X_T_base_to_imu": {"vec_txyz_rotvec": calib.X.to_vec(), "matrix": np.block([[calib.X.R, calib.X.p[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])},
        "T_WM_mocap_to_map": {"vec_txyz_rotvec": calib.T_WM.to_vec(), "frozen": True},
    }
    (res_dir / "calibration.json").write_text(json.dumps(_jsonable(calib_out), indent=2))
    (res_dir / "model.yaml").write_text(yaml.safe_dump(_jsonable(model.to_dict()), sort_keys=False))
    if model.static_field is not None:
        lo, hi = tab.pos_M[:, :2].min(0) - 0.5, tab.pos_M[:, :2].max(0) + 0.5
        np.savez(res_dir / "static_field_grid.npz", **model.static_field.export_grid(lo, hi))
    figs = out.pop("_figs")
    out["sensor"] = {"type": c.sensor.type, "name": c.sensor.name}
    (res_dir / "results.json").write_text(json.dumps(_jsonable(out), indent=2))
    from .report import write_report

    write_report(res_dir, out, model, figs, eta, tab)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign_dir", type=Path)
    ap.add_argument("--config", type=Path, default=None, help="default: <campaign_dir>/campaign.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load((args.config or args.campaign_dir / "campaign.yaml").read_text())
    out = analyze(args.campaign_dir, cfg)
    print(f"done: {args.campaign_dir / 'results' / 'report.md'}")
    print("Phase 6 pass:", out.get("phase6", {}).get("pass"))


if __name__ == "__main__":
    main()
