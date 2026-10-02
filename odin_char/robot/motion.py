"""ROS-free motion layer: reference trajectories, a mocap-feedback controller, the geofence and
identification of the command -> actual velocity gains.

All poses are planar (x, y, yaw) in the mocap world frame. Nothing here reads wheel odometry:
feedback and gain identification use mocap only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RATE = 100.0  # Hz, sampling of precomputed references


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# ============================================================================ references
@dataclass
class Ref:
    p: np.ndarray  # (2,)
    yaw: float
    v: np.ndarray  # (2,) world-frame velocity
    w: float  # yaw rate


class Trajectory:
    """A reference sampled at RATE Hz. Velocities are differentiated from the samples so the
    feedforward is always consistent with the path."""

    def __init__(self, xy: np.ndarray, yaw: np.ndarray, label: str = ""):
        self.xy = np.asarray(xy, float)
        self.yaw = np.unwrap(np.asarray(yaw, float))
        self.t = np.arange(len(self.xy)) / RATE
        self.v = np.gradient(self.xy, axis=0) * RATE if len(self.xy) > 1 else np.zeros_like(self.xy)
        self.w = np.gradient(self.yaw) * RATE if len(self.yaw) > 1 else np.zeros_like(self.yaw)
        self.label = label

    @property
    def duration(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0

    def sample(self, t: float) -> Ref:
        t = float(np.clip(t, 0, self.duration))
        k = t * RATE
        i = int(min(np.floor(k), len(self.t) - 2)) if len(self.t) > 1 else 0
        a = k - i if len(self.t) > 1 else 0.0
        j = min(i + 1, len(self.t) - 1)
        lerp = lambda arr: arr[i] * (1 - a) + arr[j] * a
        return Ref(lerp(self.xy), float(lerp(self.yaw)), lerp(self.v), float(lerp(self.w)))

    @property
    def start(self) -> np.ndarray:
        return np.array([*self.xy[0], self.yaw[0]])

    @property
    def end(self) -> np.ndarray:
        return np.array([*self.xy[-1], self.yaw[-1]])

    def max_speed(self) -> float:
        return float(np.linalg.norm(self.v, axis=1).max()) if len(self.v) else 0.0

    def __add__(self, other: "Trajectory") -> "Trajectory":
        y2 = other.yaw + np.round((self.yaw[-1] - other.yaw[0]) / (2 * np.pi)) * 2 * np.pi
        return Trajectory(np.vstack([self.xy, other.xy[1:]]), np.concatenate([self.yaw, y2[1:]]),
                          self.label or other.label)


def trapezoid(distance: float, v_max: float, a_max: float) -> np.ndarray:
    """Arc-length samples s(t) at RATE Hz for a rest-to-rest trapezoidal (or triangular) move."""
    distance = abs(distance)
    if distance < 1e-9:
        return np.zeros(2)
    t_acc = v_max / a_max
    if a_max * t_acc**2 > distance:  # triangular
        t_acc = np.sqrt(distance / a_max)
        v_max = a_max * t_acc
    t_cruise = (distance - a_max * t_acc**2) / v_max
    T = 2 * t_acc + t_cruise
    t = np.arange(0, T + 1 / RATE, 1 / RATE)
    s = np.where(
        t < t_acc,
        0.5 * a_max * t**2,
        np.where(
            t < t_acc + t_cruise,
            0.5 * a_max * t_acc**2 + v_max * (t - t_acc),
            distance - 0.5 * a_max * np.clip(T - t, 0, None) ** 2,
        ),
    )
    s[-1] = distance
    return s


def hold(pose, seconds: float) -> Trajectory:
    n = max(int(seconds * RATE), 2)
    return Trajectory(np.tile(pose[:2], (n, 1)), np.full(n, pose[2]), "hold")


def goto(start, goal, v_max: float, a_max: float, w_max: float, alpha_max: float) -> Trajectory:
    """Straight-line translation and shortest rotation, synchronised to finish together."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    d = goal[:2] - start[:2]
    L = float(np.linalg.norm(d))
    dyaw = float(wrap(goal[2] - start[2]))
    s_t = trapezoid(L, v_max, a_max)
    s_r = trapezoid(dyaw, w_max, alpha_max)
    n = max(len(s_t), len(s_r))
    # stretch the shorter profile so both end together (keeps it rest-to-rest)
    u_t = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(s_t)), s_t / max(L, 1e-9)) if L > 1e-9 else np.zeros(n)
    u_r = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(s_r)), s_r / max(abs(dyaw), 1e-9)) if abs(dyaw) > 1e-9 else np.zeros(n)
    xy = start[:2] + u_t[:, None] * d
    yaw = start[2] + u_r * dyaw
    return Trajectory(xy, yaw, "goto")


def _resample_path(xy: np.ndarray, ds: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 1e-12])
    xy, s = xy[keep], s[keep]
    sq = np.arange(0, s[-1], ds)
    sq = np.append(sq, s[-1])
    return np.stack([np.interp(sq, s, xy[:, 0]), np.interp(sq, s, xy[:, 1])], 1), sq


def follow_path(xy: np.ndarray, speed: float, a_max: float, heading: str | float = "tangent") -> Trajectory:
    """Traverse a polyline at `speed` with a rest-to-rest trapezoidal profile."""
    P, S = _resample_path(np.asarray(xy, float))
    s_t = trapezoid(S[-1], speed, a_max)
    x = np.interp(s_t, S, P[:, 0])
    y = np.interp(s_t, S, P[:, 1])
    if heading == "tangent":
        tang = np.gradient(P, axis=0)
        path_yaw = np.unwrap(np.arctan2(tang[:, 1], tang[:, 0]))
        yaw = np.interp(s_t, S, path_yaw)
    else:
        yaw = np.full(len(s_t), float(heading))
    return Trajectory(np.stack([x, y], 1), yaw, "path")


def chaikin(xy: np.ndarray, iterations: int = 4, closed: bool = False) -> np.ndarray:
    """Corner cutting: rounds polyline corners so a tangent heading stays continuous."""
    P = np.asarray(xy, float)
    for _ in range(iterations):
        Q = 0.75 * P[:-1] + 0.25 * P[1:]
        R = 0.25 * P[:-1] + 0.75 * P[1:]
        mid = np.empty((2 * len(Q), 2))
        mid[0::2], mid[1::2] = Q, R
        P = mid if closed else np.vstack([P[0], mid, P[-1]])
    return P


def figure8_path(center, half_length: float, half_width: float, rotation: float, laps: int,
                 n: int = 2000) -> np.ndarray:
    """Lemniscate of Gerono traversed `laps` times; one lap turns both directions."""
    s = np.linspace(0, 2 * np.pi * laps, n * laps, endpoint=True)
    x = half_length * np.sin(s)
    y = half_width * np.sin(s) * np.cos(s)
    c, sn = np.cos(rotation), np.sin(rotation)
    return np.stack([center[0] + c * x - sn * y, center[1] + sn * x + c * y], 1)


def figure8(center, half_length, half_width, rotation, laps, speed, a_max) -> Trajectory:
    tr = follow_path(figure8_path(center, half_length, half_width, rotation, laps), speed, a_max)
    tr.label = "figure8"
    return tr


def lawnmower_path(lo, hi, spacing: float, loops: int) -> np.ndarray:
    """Boustrophedon lanes over the box plus `loops` perimeter loops (loop closures), rounded."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    ys = np.arange(lo[1], hi[1] + 1e-9, spacing)
    pts = []
    for i, y in enumerate(ys):
        xs = (lo[0], hi[0]) if i % 2 == 0 else (hi[0], lo[0])
        pts += [(xs[0], y), (xs[1], y)]
    perim = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1]), (lo[0], lo[1])]
    pts += [pts[-1]] + perim * loops
    return chaikin(np.array(pts), 4)


def sinusoid_axis(pose, axis_yaw: float, freqs, peak_speed: float, seconds_each: float,
                  ramp_cycles: float = 1.0, max_amplitude: float = 0.5) -> Trajectory:
    """Back-and-forth along one axis at several frequencies, amplitude ramped in/out."""
    xs, ys, yaws = [], [], []
    d = np.array([np.cos(axis_yaw), np.sin(axis_yaw)])
    for f in freqs:
        A = min(peak_speed / (2 * np.pi * f), max_amplitude)
        t = np.arange(0, seconds_each, 1 / RATE)
        env = np.clip(np.minimum(t, seconds_each - t) * f / ramp_cycles, 0, 1)
        env = np.sin(env * np.pi / 2) ** 2
        u = A * env * np.sin(2 * np.pi * f * t)
        xs.append(pose[0] + u * d[0])
        ys.append(pose[1] + u * d[1])
        yaws.append(np.full(len(t), pose[2]))
    return Trajectory(np.stack([np.concatenate(xs), np.concatenate(ys)], 1), np.concatenate(yaws), "sinusoid")


def yaw_sinusoid(pose, freqs, peak_rate: float, seconds_each: float, ramp_cycles: float = 1.0,
                 max_amplitude: float = 1.5) -> Trajectory:
    yaws = []
    for f in freqs:
        A = min(peak_rate / (2 * np.pi * f), max_amplitude)
        t = np.arange(0, seconds_each, 1 / RATE)
        env = np.clip(np.minimum(t, seconds_each - t) * f / ramp_cycles, 0, 1)
        env = np.sin(env * np.pi / 2) ** 2
        yaws.append(pose[2] + A * env * np.sin(2 * np.pi * f * t))
    yaw = np.concatenate(yaws)
    return Trajectory(np.tile(pose[:2], (len(yaw), 1)), yaw, "yaw_sinusoid")


def calib_motion(pose, duration: float, radius: float, peak_rate: float = 2.5, peak_speed: float = 0.5) -> Trajectory:
    """Phase 1: rich rotation (yaw-rate chirp in both directions) plus slow translations, so the
    time offset (|omega| cross-correlation) and the hand-eye solve are well conditioned."""
    t = np.arange(0, duration, 1 / RATE)
    env = np.clip(np.minimum(t, duration - t) / 3.0, 0, 1)
    f = 0.05 + 0.25 * t / duration  # chirp 0.05 -> 0.3 Hz
    phase = 2 * np.pi * np.cumsum(f) / RATE
    yaw_rate = peak_rate * env * np.sin(phase)
    yaw = pose[2] + np.cumsum(yaw_rate) / RATE
    ax = min(radius, peak_speed / (2 * np.pi * 0.11))
    ay = min(radius, peak_speed / (2 * np.pi * 0.07))
    x = pose[0] + ax * env * np.sin(2 * np.pi * 0.11 * t)
    y = pose[1] + ay * env * np.sin(2 * np.pi * 0.07 * t + 1.0) - ay * env * np.sin(1.0)
    return Trajectory(np.stack([x, y], 1), yaw, "calib")


def line_back_and_forth(pose, length: float, speed: float, a_max: float, seconds: float) -> Trajectory:
    """Slow shuttle along the robot's heading (used while the sensor is covered)."""
    d = np.array([np.cos(pose[2]), np.sin(pose[2])])
    a = np.asarray(pose[:2]) - d * length / 2
    b = np.asarray(pose[:2]) + d * length / 2
    one = follow_path(np.array([a, b]), speed, a_max, heading=pose[2]) + follow_path(np.array([b, a]), speed, a_max, heading=pose[2])
    start = goto(pose, [*a, pose[2]], speed, a_max, 1.0, 2.0)
    tr = start
    while tr.duration < seconds:
        tr = tr + one
    return tr + goto(tr.end, pose, speed, a_max, 1.0, 2.0)


# ============================================================================ safety
@dataclass
class Fence:
    lo: np.ndarray
    hi: np.ndarray
    margin: float

    def inside(self, p, margin: float | None = None) -> bool:
        m = self.margin if margin is None else margin
        p = np.asarray(p)[:2]
        return bool(np.all(p >= self.lo + m) and np.all(p <= self.hi - m))

    def stop_point_inside(self, p, v, a_max: float) -> bool:
        v = np.asarray(v)[:2]
        sp = float(np.linalg.norm(v))
        stop = np.asarray(p)[:2] + (v / sp * sp**2 / (2 * a_max) if sp > 1e-6 else 0)
        return self.inside(stop, self.margin / 2)

    def trajectory_inside(self, tr: Trajectory) -> bool:
        return bool(np.all(tr.xy >= self.lo + self.margin) and np.all(tr.xy <= self.hi - self.margin))


# ============================================================================ control
@dataclass
class ControllerConfig:
    kp_pos: float = 2.0
    ki_pos: float = 0.2
    kp_yaw: float = 3.0
    max_speed: float = 1.8
    max_accel: float = 1.5
    max_yaw_rate: float = 3.0
    max_yaw_accel: float = 6.0
    max_track_err: float = 0.5  # m: abort if exceeded for max_track_err_s
    max_track_err_s: float = 0.5
    integrator_limit: float = 0.3  # m/s


@dataclass
class Controller:
    cfg: ControllerConfig
    G: np.ndarray = field(default_factory=lambda: np.eye(3))  # actual = G @ command (body vx, vy, wz)
    _i: np.ndarray = field(default_factory=lambda: np.zeros(2))
    _last: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _err_since: float | None = None

    def reset(self):
        self._i[:] = 0
        self._last[:] = 0
        self._err_since = None

    def update(self, t: float, pose: np.ndarray, ref: Ref, dt: float) -> tuple[np.ndarray, str | None]:
        """Returns (command [vx, vy, wz] in the body frame, abort reason or None)."""
        c = self.cfg
        e = ref.p - pose[:2]
        en = float(np.linalg.norm(e))
        if en > c.max_track_err:
            self._err_since = t if self._err_since is None else self._err_since
            if t - self._err_since > c.max_track_err_s:
                return np.zeros(3), f"tracking error {en:.2f} m"
        else:
            self._err_since = None
        self._i = np.clip(self._i + c.ki_pos * e * dt, -c.integrator_limit, c.integrator_limit)
        v_w = ref.v + c.kp_pos * e + self._i
        w = ref.w + c.kp_yaw * float(wrap(ref.yaw - pose[2]))
        # rotate into the body frame at the mid-tick heading
        yaw_mid = pose[2] + 0.5 * w * dt
        cy, sy = np.cos(yaw_mid), np.sin(yaw_mid)
        v_b = np.array([cy * v_w[0] + sy * v_w[1], -sy * v_w[0] + cy * v_w[1]])
        sp = float(np.linalg.norm(v_b))
        if sp > c.max_speed:
            v_b *= c.max_speed / sp
        w = float(np.clip(w, -c.max_yaw_rate, c.max_yaw_rate))
        des = np.array([v_b[0], v_b[1], w])
        # acceleration limits on the commanded body velocity
        dv = des[:2] - self._last[:2]
        dvn = float(np.linalg.norm(dv))
        if dvn > c.max_accel * dt * 2:
            dv *= c.max_accel * dt * 2 / dvn
        dw = float(np.clip(des[2] - self._last[2], -c.max_yaw_accel * dt * 2, c.max_yaw_accel * dt * 2))
        out = np.array([self._last[0] + dv[0], self._last[1] + dv[1], self._last[2] + dw])
        self._last = out
        return np.linalg.solve(self.G, out), None


# ============================================================================ sysid
def sysid_plan(v: float = 0.3, w: float = 1.0, pulse_s: float = 1.0, gap_s: float = 0.7):
    """Command pulses (body vx, vy, wz) with rests; each +/- pair returns the robot near start."""
    plan = []
    for cmd in ([v, 0, 0], [-v, 0, 0], [0, v, 0], [0, -v, 0], [0, 0, w], [0, 0, -w]):
        plan.append((np.array(cmd, float), pulse_s))
        plan.append((np.zeros(3), gap_s))
    return plan


@dataclass
class SysIdResult:
    G: np.ndarray
    delay_s: float
    residual: float
    ok: bool
    reason: str


def fit_command_gains(t: np.ndarray, cmd: np.ndarray, meas: np.ndarray,
                      delays=np.arange(0.0, 0.31, 0.02), settle_s: float = 0.4) -> SysIdResult:
    """Least-squares G with meas(t) ~= G cmd(t - delay), using only samples where the delayed
    command has been constant for settle_s (steady state). meas is mocap body velocity."""
    best = None
    for d in delays:
        c = np.stack([np.interp(t - d, t, cmd[:, k]) for k in range(3)], 1)
        changed = np.r_[True, np.any(np.abs(np.diff(c, axis=0)) > 1e-9, axis=1)]
        last_change = np.maximum.accumulate(np.where(changed, t, -np.inf))
        steady = (t - last_change) > settle_s
        excited = np.linalg.norm(c, axis=1) > 1e-6
        m = steady & excited
        if m.sum() < 10:
            continue
        X, *_ = np.linalg.lstsq(c[m], meas[m], rcond=None)
        r = float(np.sqrt(np.mean((c[m] @ X - meas[m]) ** 2)))
        if best is None or r < best[2]:
            best = (X.T, d, r)
    if best is None:
        return SysIdResult(np.eye(3), 0.0, np.inf, False, "not enough steady-state data")
    G, d, r = best
    diag = np.diag(G)
    if np.any(diag < 0.3) or np.any(diag > 3.0):
        return SysIdResult(G, d, r, False, f"implausible gains {np.round(diag, 2)}: is the remote in "
                           "left-UP/right-MID and the chassis velocity bridge running?")
    return SysIdResult(G, float(d), r, True, "ok")
