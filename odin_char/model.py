"""The fitted Tier 0 model and a simulator for it (used by Phase 6 and by the sim team).

eps = eps_s(p) + eta (+ latency: the output is the pose delayed by tau_d)
eta per body-aligned axis: AR(1) with Student-t innovations, separate moving/stationary fits.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .errors import AXES
from .static_field import StaticField


@dataclass
class AxisParams:
    a: float
    P: float
    nu: float  # inf = Gaussian
    scale: float
    second: dict | None = None  # two-component FOGM, if needed

    @staticmethod
    def from_fit(fit) -> "AxisParams":
        return AxisParams(fit.a, fit.P, fit.nu, fit.scale, fit.second_component)


@dataclass
class Tier0Model:
    dt: float
    tau_d: float
    moving: list  # 6 x AxisParams
    stationary: list | None
    static_field: StaticField | None
    lambda_lost: float = 0.0
    events: dict = field(default_factory=dict)
    moving_speed_threshold: float = 0.05
    tau_d_stderr: float = 0.0

    # ------------------------------------------------------------------ simulation
    def _innovation(self, p: AxisParams, rng, size):
        if np.isfinite(p.nu):
            return p.scale * rng.standard_t(p.nu, size)
        return p.scale * rng.standard_normal(size)

    def _components(self, p: AxisParams):
        """[(a, innovation sampler, stationary variance)] for one or two FOGM components."""
        if p.second:
            s = p.second
            out = []
            for tau, P in ((s["tau_fast"], s["P_fast"]), (s["tau_slow"], s["P_slow"])):
                a = np.exp(-self.dt / tau)
                sd = np.sqrt(P * (1 - a**2))
                out.append((a, lambda rng, n, sd=sd: sd * rng.standard_normal(n), P))
            return out
        return [(p.a, lambda rng, n, p=p: self._innovation(p, rng, n), p.P)]

    def simulate_eta(self, moving_mask: np.ndarray, segments: list[np.ndarray], rng) -> np.ndarray:
        from scipy.signal import lfilter

        eta = np.zeros((len(moving_mask), 6))
        for ax in range(6):
            pm = self.moving[ax]
            ps = self.stationary[ax] if self.stationary else pm
            for seg in segments:
                reg = moving_mask[seg]
                cuts = np.where(np.diff(reg.astype(int)) != 0)[0] + 1
                state = None
                for piece in np.split(np.arange(len(seg)), cuts):
                    comps = self._components(pm if reg[piece[0]] else ps)
                    if state is None or len(state) != len(comps):
                        state = [np.sqrt(P) * rng.standard_normal() for _, _, P in comps]
                        first_given = True
                    else:
                        first_given = False
                    total = np.zeros(len(piece))
                    new_state = []
                    for (a, draw, _), x0 in zip(comps, state):
                        e = draw(rng, len(piece))
                        if first_given:
                            e[0] = 0.0
                            y = lfilter([1.0], [1.0, -a], e, zi=[x0])[0]
                        else:
                            y = lfilter([1.0], [1.0, -a], e, zi=[a * x0])[0]
                        total += y
                        new_state.append(y[-1])
                    state = new_state
                    eta[seg[piece], ax] = total
        return eta

    def simulate_error(self, pos: np.ndarray, speed: np.ndarray, yaw_rate: np.ndarray,
                       segments: list[np.ndarray], rng, parameter_uncertainty: bool = False) -> np.ndarray:
        """Body-aligned error seen by a consumer along a true trajectory (includes latency).

        parameter_uncertainty: also draw tau_d and eps_s from their estimation uncertainty
        (used for hold-out validation bands; the sim itself uses the point estimates)."""
        eps = self.simulate_eta(speed > self.moving_speed_threshold, segments, rng)
        if self.static_field is not None:
            eps += self.static_field.sample(pos, rng) if parameter_uncertainty else self.static_field.predict(pos)
        tau = self.tau_d + (self.tau_d_stderr * rng.standard_normal() if parameter_uncertainty else 0.0)
        eps[:, 0] -= speed * tau
        eps[:, 5] -= yaw_rate * tau
        return eps

    # ------------------------------------------------------------------ export
    def to_dict(self) -> dict:
        def ax(lst):
            if lst is None:
                return None
            return {
                name: {
                    "a": p.a,
                    "tau_c": float(-self.dt / np.log(p.a)),
                    "P": p.P,
                    "sigma": float(np.sqrt(p.P)),
                    "nu": None if not np.isfinite(p.nu) else p.nu,
                    "innovation_scale": p.scale,
                    "second_component": p.second,
                }
                for name, p in zip(AXES, lst)
            }

        sf = self.static_field
        return {
            "dt": self.dt,
            "tau_d": self.tau_d,
            "units": "metres / radians; axes are body-aligned (along-track, cross-track, vertical, roll, pitch, yaw)",
            "dynamic_error": {"moving": ax(self.moving), "stationary": ax(self.stationary)},
            "static_field": None
            if sf is None
            else {
                "enabled_axes": [a for a, e in zip(AXES, sf.enabled) if e],
                "grid_resolution": sf.res,
                "cross_session_r2": dict(zip(AXES, [None if not np.isfinite(r) else float(r) for r in sf.r2])),
                "grid_file": "static_field_grid.npz",
            },
            "events": self.events,
            "moving_speed_threshold": self.moving_speed_threshold,
        }
