import numpy as np

from odin_char import registration as reg
from odin_char.bagio import BagData, lowpass_series
from odin_char.errors import ErrorTable, map_to_track, track_to_map
from odin_char.se3 import Pose, PoseSeries, rot_z


def _table(err, heading, run, t):
    n = len(t)
    I = Pose.identity(n)
    return ErrorTable(t=t, err=err, pos_M=np.zeros((n, 3)), speed=np.ones(n), yaw_rate=np.zeros(n),
                      map_age=np.zeros(n), heading=heading, est_M=I, true_M=I, run=run,
                      meta={int(r): {"session": "s1"} for r in np.unique(run)})


def test_track_map_roundtrip():
    rng = np.random.default_rng(0)
    e = rng.normal(size=(50, 6))
    h = rng.uniform(-np.pi, np.pi, 50)
    assert np.allclose(map_to_track(track_to_map(e, h), h), e)


def test_instances_split_and_link():
    t = np.arange(0, 20, 0.1)
    p = np.zeros((len(t), 3))
    p[100:] += [1.5, 0.0, 0.0]  # driver restart: odom->map jumps by metres
    R = np.tile(np.eye(3), (len(t), 1, 1))
    segs = reg.split_odom_map(1, t, p, R)
    assert len(segs) == 2
    # a following run continuing the second registration links to it
    segs += reg.split_odom_map(2, t + 30, p[100:101].repeat(len(t), 0), R)
    segs = reg.link_instances(segs)
    assert [s.instance for s in sorted(segs, key=lambda s: s.t0)] == [0, 1, 1]


def test_sigma_recovers_spread_and_flags_wrong_fix():
    rng = np.random.default_rng(1)
    sigma = np.array([0.03, 0.01, 0.015, 0.005, 0.004, 0.003])
    biases = [reg.InstanceBias(k, [k], 1000, 100.0, sigma * rng.standard_normal(6)) for k in range(40)]
    biases.append(reg.InstanceBias(99, [99], 300, 30.0, np.array([0.1, -0.1, -0.2, 0.03, 0.03, 0.0])))
    fit = reg.fit_sigma(biases)
    est = np.array([fit["sigma"][a] for a in reg.MAP_AXES])
    assert np.all(np.abs(est / sigma - 1) < 0.35)
    assert [w["instance"] for w in fit["wrong_fixes"]] == [99]


def test_debias_removes_instance_means_in_map_frame():
    rng = np.random.default_rng(2)
    n = 400
    t = np.arange(n) * 0.1
    h = np.linspace(0, 4 * np.pi, n)  # heading turns: a map-frame offset is not constant in the track frame
    inst = (np.arange(n) >= 200).astype(int)
    bM = np.where(inst[:, None] == 0, [0.03, -0.02, 0.01, 0, 0, 0.004], [-0.01, 0.04, 0.0, 0.002, 0, 0])
    noise = 0.001 * rng.standard_normal((n, 6))
    tab = _table(map_to_track(bM, h) + noise, h, np.ones(n, int), t)
    biases = reg.instance_biases(tab, inst, 0.1)
    out = reg.debias(tab, inst, biases)
    assert np.sqrt(np.mean(out**2)) < 0.002


def test_mocap_clean_dedup_and_backdate():
    # 360 Hz frames delivered in bursts of 3 every 3 frames, each published twice
    rng = np.random.default_rng(3)
    n = 3000
    tc = np.arange(n) / 360.0
    p = np.column_stack([0.8 * tc, 0.01 * np.sin(tc), np.zeros(n)]) + 1e-5 * rng.standard_normal((n, 3))
    q = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    burst_arrival = tc[(np.arange(n) // 3) * 3 + 2] + 0.004
    t_pub = burst_arrival + 1e-5 * (np.arange(n) % 3)
    b = BagData(path="synthetic")
    for copy in range(2):
        for i in range(n):
            b.mocap.t_stamp.append(t_pub[i] + copy * 2e-4 + 1e-7 * i)
            b.mocap.t_recv.append(t_pub[i] + copy * 2e-4)
            b.mocap.p.append(p[i])
            b.mocap.q.append(q[i])
    s = b.mocap_series()
    assert b.mocap_clean_stats["duplicates"] == n
    assert len(s.t) == n
    # back-dated stamps are uniform up to the constant delivery delay
    lag = s.t - tc
    assert np.ptp(lag[10:-10]) < 1e-3


def test_lowpass_keeps_slow_motion_removes_vibration():
    fs = 360.0
    t = np.arange(0, 20, 1 / fs)
    slow = 0.2 * np.sin(2 * np.pi * 0.5 * t)
    vib = 0.002 * np.sin(2 * np.pi * 8.0 * t)
    yaw = 0.05 * np.sin(2 * np.pi * 0.5 * t) + 0.007 * np.sin(2 * np.pi * 8.0 * t)
    P = Pose(rot_z(yaw), np.column_stack([slow + vib, np.zeros_like(t), np.zeros_like(t)]))
    f = lowpass_series(PoseSeries(t, P), 4.0)
    m = (f.t > 2) & (f.t < 18)
    x_true = 0.2 * np.sin(2 * np.pi * 0.5 * f.t[m])
    assert np.max(np.abs(f.pose.p[m, 0] - x_true)) < 3e-4
    yaw_f = np.arctan2(f.pose.R[m, 1, 0], f.pose.R[m, 0, 0])
    assert np.max(np.abs(yaw_f - 0.05 * np.sin(2 * np.pi * 0.5 * f.t[m]))) < 7e-4
