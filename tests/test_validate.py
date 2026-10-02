"""Phase 6 must accept the true model and reject wrong ones."""
import copy

import numpy as np
import pytest

from odin_char.errors import ErrorTable
from odin_char.model import AxisParams, Tier0Model
from odin_char.se3 import Pose
from odin_char.validate import validate

DT = 0.1


def _truth_model():
    sig = np.array([0.02, 0.015, 0.01, 0.003, 0.003, 0.006])
    tau = np.array([1.5, 1.0, 2.0, 0.8, 0.8, 1.2])
    axes = []
    for s, t in zip(sig, tau):
        a = np.exp(-DT / t)
        axes.append(AxisParams(a, s**2, np.inf, s * np.sqrt(1 - a**2)))
    return Tier0Model(DT, 0.045, axes, None, None)


def _table(model, seed):
    n_runs, n = 3, 1500
    t = np.concatenate([np.arange(n) * DT + 1000 * r for r in range(n_runs)])
    run = np.repeat(np.arange(n_runs), n)
    s = np.arange(n * n_runs) * DT
    pos = np.stack([3 * np.sin(0.1 * s), 1.8 * np.sin(0.1 * s) * np.cos(0.1 * s), 0 * s], 1)
    speed = np.full(len(t), 0.8)
    yaw_rate = 0.5 * np.sin(0.2 * s)
    tab = ErrorTable(t=t, err=np.zeros((len(t), 6)), pos_M=pos, speed=speed, yaw_rate=yaw_rate,
                     map_age=np.zeros(len(t)), heading=np.zeros(len(t)), est_M=Pose.identity(len(t)),
                     true_M=Pose.identity(len(t)), run=run, meta={r: {"session": "s3"} for r in range(n_runs)})
    tab.err = model.simulate_error(pos, speed, yaw_rate, tab.segments(DT), np.random.default_rng(seed))
    return tab


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_true_model_passes(seed):
    m = _truth_model()
    r = validate(m, _table(m, seed), n_mc=50, seed=seed + 100)
    assert r.passed, r.failing


def test_wrong_sigma_fails():
    m = _truth_model()
    held = _table(m, 21)
    bad = copy.deepcopy(m)
    p = bad.moving[0]
    bad.moving[0] = AxisParams(p.a, p.P * 0.4, p.nu, p.scale * np.sqrt(0.4))
    r = validate(bad, held, n_mc=50, seed=5)
    # total-error RMSE is dominated by the latency bias here; the distribution/RPE checks catch it
    assert not r.passed


def test_wrong_correlation_time_fails():
    m = _truth_model()
    held = _table(m, 22)
    bad = copy.deepcopy(m)
    p = bad.moving[1]
    a = np.exp(-DT / 0.2)
    bad.moving[1] = AxisParams(a, p.P, p.nu, np.sqrt(p.P * (1 - a**2)))
    r = validate(bad, held, n_mc=50, seed=6)
    assert not r.passed and "acf" in r.failing


def test_missing_latency_fails():
    m = _truth_model()
    held = _table(m, 23)
    bad = copy.deepcopy(m)
    bad.tau_d = 0.0
    r = validate(bad, held, n_mc=50, seed=7)
    assert not r.passed
