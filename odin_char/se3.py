"""Vectorised SO(3)/SE(3) helpers.

Poses are stored as (R, p) with R of shape (..., 3, 3) and p of shape (..., 3). T_AB maps
coordinates in frame B into frame A (i.e. it is the pose of B expressed in A).
Quaternions are (x, y, z, w), matching ROS.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def quat_to_R(q: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(np.asarray(q, dtype=float)).as_matrix()


def R_to_quat(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_quat()


def so3_log(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_rotvec()


def so3_exp(w: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(w, dtype=float)).as_matrix()


def yaw_of(R: np.ndarray) -> np.ndarray:
    """Heading of the body x axis projected on the world xy plane."""
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def rot_z(yaw: np.ndarray) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=float)
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.zeros(yaw.shape + (3, 3))
    out[..., 0, 0] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    out[..., 1, 1] = c
    out[..., 2, 2] = 1.0
    return out


def wrap(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Pose:
    """A (possibly batched) rigid transform."""

    R: np.ndarray
    p: np.ndarray

    @staticmethod
    def identity(n: int | None = None) -> "Pose":
        if n is None:
            return Pose(np.eye(3), np.zeros(3))
        return Pose(np.tile(np.eye(3), (n, 1, 1)), np.zeros((n, 3)))

    @staticmethod
    def from_quat(p, q) -> "Pose":
        return Pose(quat_to_R(q), np.asarray(p, dtype=float))

    @staticmethod
    def from_vec(x: np.ndarray) -> "Pose":
        """From [tx, ty, tz, rx, ry, rz] (translation + rotation vector)."""
        x = np.asarray(x, dtype=float)
        return Pose(so3_exp(x[..., 3:]), x[..., :3].copy())

    def to_vec(self) -> np.ndarray:
        return np.concatenate([self.p, so3_log(self.R)], axis=-1)

    def __matmul__(self, other: "Pose") -> "Pose":
        R = self.R @ other.R
        p = (self.R @ other.p[..., None])[..., 0] + self.p
        return Pose(R, p)

    def inv(self) -> "Pose":
        Rt = np.swapaxes(self.R, -1, -2)
        return Pose(Rt, -(Rt @ self.p[..., None])[..., 0])

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (self.R @ np.asarray(x)[..., None])[..., 0] + self.p

    def __getitem__(self, idx) -> "Pose":
        return Pose(self.R[idx], self.p[idx])

    def __len__(self) -> int:
        return self.p.shape[0]


class PoseSeries:
    """Time-stamped pose sequence with gap-aware interpolation."""

    def __init__(self, t: np.ndarray, pose: Pose):
        order = np.argsort(t)
        t = np.asarray(t, dtype=float)[order]
        keep = np.concatenate([[True], np.diff(t) > 0])
        self.t = t[keep]
        self.pose = pose[order][keep]

    def __len__(self) -> int:
        return len(self.t)

    def shifted(self, dt: float) -> "PoseSeries":
        return PoseSeries(self.t + dt, self.pose)

    def interp(self, tq: np.ndarray, max_gap: float = np.inf) -> tuple[Pose, np.ndarray]:
        """Interpolate at tq. Returns (poses, valid) where invalid samples fall outside the
        series or inside a gap longer than max_gap (never interpolate across gaps)."""
        tq = np.asarray(tq, dtype=float)
        valid = (tq >= self.t[0]) & (tq <= self.t[-1])
        idx = np.clip(np.searchsorted(self.t, tq, side="right") - 1, 0, len(self.t) - 2)
        gap = self.t[idx + 1] - self.t[idx]
        valid &= gap <= max_gap
        tc = np.clip(tq, self.t[0], self.t[-1])
        w = ((tc - self.t[idx]) / gap)[:, None]
        p = self.pose.p[idx] * (1 - w) + self.pose.p[idx + 1] * w
        slerp = Slerp(self.t, Rotation.from_matrix(self.pose.R))
        R = slerp(tc).as_matrix()
        return Pose(R, p), valid

    def zoh(self, tq: np.ndarray) -> tuple[Pose, np.ndarray, np.ndarray]:
        """Latest sample at or before tq (zero-order hold). Returns (poses, valid, age)."""
        tq = np.asarray(tq, dtype=float)
        idx = np.searchsorted(self.t, tq, side="right") - 1
        valid = idx >= 0
        idx = np.clip(idx, 0, len(self.t) - 1)
        return self.pose[idx], valid, tq - self.t[idx]

    def velocity(self, tq: np.ndarray, dt: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
        """World-frame linear velocity and yaw rate by central differences."""
        a, _ = self.interp(tq - dt)
        b, _ = self.interp(tq + dt)
        v = (b.p - a.p) / (2 * dt)
        w = wrap(yaw_of(b.R) - yaw_of(a.R)) / (2 * dt)
        return v, w

    def body_rates(self, tq: np.ndarray, dt: float) -> np.ndarray:
        """Body-frame angular velocity (rad/s) from differentiated orientation."""
        a, _ = self.interp(tq - dt / 2)
        b, _ = self.interp(tq + dt / 2)
        return so3_log(np.swapaxes(a.R, -1, -2) @ b.R) / dt


def umeyama(src: np.ndarray, dst: np.ndarray) -> Pose:
    """Least-squares rigid T with dst ~= T.apply(src) (no scale)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    return Pose(R, mu_d - R @ mu_s)
