import numpy as np
import pytest

from odin_char.robot.motion import (
    RATE,
    Controller,
    ControllerConfig,
    Fence,
    calib_motion,
    figure8,
    fit_command_gains,
    goto,
    lawnmower_path,
    follow_path,
    sinusoid_axis,
    sysid_plan,
    trapezoid,
    wrap,
    yaw_sinusoid,
)


def test_trapezoid_limits():
    s = trapezoid(5.0, 1.0, 0.5)
    v = np.diff(s) * RATE
    a = np.diff(v) * RATE
    assert s[-1] == pytest.approx(5.0)
    assert v.max() <= 1.0 + 1e-6
    assert np.abs(a).max() <= 0.5 * 1.05


def test_figure8_speed_and_both_turn_directions():
    tr = figure8((0, 0), 3.0, 1.8, 0.0, 2, 0.8, 1.0)
    sp = np.linalg.norm(tr.v, axis=1)
    assert sp.max() == pytest.approx(0.8, rel=0.03)
    assert tr.w.max() > 0.3 and tr.w.min() < -0.3
    # heading follows the direction of travel
    moving = sp > 0.2
    trav = np.arctan2(tr.v[moving, 1], tr.v[moving, 0])
    assert np.abs(wrap(trav - tr.yaw[moving])).max() < 0.05


def test_goto_ends_at_goal_at_rest():
    tr = goto([0, 0, 0], [1.0, -0.5, 2.0], 0.5, 1.0, 1.0, 2.0)
    assert np.allclose(tr.end[:2], [1.0, -0.5], atol=1e-3)
    assert wrap(tr.end[2] - 2.0) == pytest.approx(0, abs=1e-3)
    assert np.linalg.norm(tr.v[-1]) < 0.05


def test_sinusoids_respect_peaks():
    tr = sinusoid_axis([0, 0, 0.5], 0.5, [0.5, 1, 2], 0.6, 10)
    assert np.linalg.norm(tr.v, axis=1).max() <= 0.6 * 1.05
    ty = yaw_sinusoid([0, 0, 0], [0.5, 1, 2], 1.5, 10)
    assert np.abs(ty.w).max() <= 1.5 * 1.05


def test_calib_and_lawnmower_stay_in_box():
    fence = Fence(np.array([-4.0, -3.0]), np.array([4.0, 3.0]), 0.4)
    assert fence.trajectory_inside(calib_motion([0, 0, 0], 150, 1.0))
    path = lawnmower_path([-3.4, -2.4], [3.4, 2.4], 1.0, 2)
    assert fence.trajectory_inside(follow_path(path, 0.4, 0.5))


def test_fence_stopping_point():
    fence = Fence(np.array([-4.0, -3.0]), np.array([4.0, 3.0]), 0.4)
    assert fence.stop_point_inside([3.0, 0.0], [0.5, 0.0], 1.5)
    assert not fence.stop_point_inside([3.4, 0.0], [1.8, 0.0], 1.5)


def _simulate(tr, G_true, delay_steps=2, dt=0.02, controller=None):
    """Kinematic robot: body velocity = G_true @ command (delayed)."""
    ctl = controller or Controller(ControllerConfig())
    pose = tr.start.copy()
    queue = [np.zeros(3)] * delay_steps
    errs = []
    t = 0.0
    while t < tr.duration:
        ref = tr.sample(t)
        cmd, abort = ctl.update(t, pose, ref, dt)
        assert abort is None
        queue.append(cmd)
        act = G_true @ queue.pop(0)
        c, s = np.cos(pose[2]), np.sin(pose[2])
        pose = pose + dt * np.array([c * act[0] - s * act[1], s * act[0] + c * act[1], act[2]])
        errs.append(np.linalg.norm(ref.p - pose[:2]))
        t += dt
    return np.array(errs)


def test_controller_tracks_figure8_with_identified_gains():
    tr = figure8((0, 0), 3.0, 1.8, 0.0, 1, 0.8, 1.0)
    G_true = np.diag([0.8, 0.75, 1.2])
    errs = _simulate(tr, G_true, controller=Controller(ControllerConfig(), G=G_true))
    assert errs.max() < 0.05
    # even with unidentified gains the feedback keeps it on the path
    errs = _simulate(tr, G_true)
    assert errs.max() < 0.25


def test_sysid_recovers_gains_and_delay():
    G_true = np.array([[0.8, 0.05, 0.0], [-0.03, 0.75, 0.0], [0.0, 0.0, 1.2]])
    dt = 0.01
    t, cmds = [], []
    now = 0.0
    for cmd, dur in sysid_plan():
        n = int(dur / dt)
        t += list(now + np.arange(n) * dt)
        cmds += [cmd] * n
        now += n * dt
    t, cmds = np.array(t), np.array(cmds)
    delayed = np.stack([np.interp(t - 0.08, t, cmds[:, k]) for k in range(3)], 1)
    rng = np.random.default_rng(0)
    meas = delayed @ G_true.T + rng.normal(0, 0.01, delayed.shape)
    r = fit_command_gains(t, cmds, meas)
    assert r.ok
    assert np.allclose(r.G, G_true, atol=0.03)
    assert r.delay_s == pytest.approx(0.08, abs=0.021)


def test_sysid_flags_robot_not_moving():
    t = np.arange(0, 10, 0.01)
    cmds = np.zeros((len(t), 3))
    cmds[100:300, 0] = 0.3
    r = fit_command_gains(t, cmds, np.zeros_like(cmds))
    assert not r.ok
