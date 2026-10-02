import json

import numpy as np
import pytest
import yaml

from odin_char.pipeline import analyze
from odin_char.se3 import Pose, so3_log

from .synth import make_campaign


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp("campaign")
    truth = make_campaign(root, seed=3, short=True)
    cfg = yaml.safe_load((root / "campaign.yaml").read_text())
    out = analyze(root, cfg)
    return root, truth, out


def test_time_offset(campaign):
    _, truth, out = campaign
    # mocap is mapped from the server clock to the host clock; the minimum network delay (0.8 ms)
    # remains and is absorbed here
    assert out["phase1"]["time_offset_s"] == pytest.approx(truth.time_offset + 0.0008, abs=0.002)


def test_hand_eye(campaign):
    _, truth, out = campaign
    he = out["phase1"]["hand_eye"]
    X = Pose.from_vec(np.array(he["X_T_base_to_imu"]))
    Xt = Pose.from_vec(truth.X)
    assert np.degrees(np.linalg.norm(so3_log(X.R @ Xt.R.T))) < 0.3
    assert np.linalg.norm(X.p[:2] - Xt.p[:2]) < 0.005
    # planar motion barely sees the vertical lever arm: either it's flagged or it's still right
    # (the synthetic CAD prior is 2 cm off in z, so tz may legitimately sit near the prior)
    assert "tz" in he["prior_dominated_parameters"] or abs(X.p[2] - Xt.p[2]) < 0.03
    assert he["motion_only_std"]["tz"] > he["motion_only_std"]["tx"]


def test_alignment(campaign):
    _, truth, out = campaign
    T = Pose.from_vec(np.array(out["phase1"]["T_WM"]["vec"]))
    Tt = Pose.from_vec(truth.T_WM)
    assert np.linalg.norm(T.p[:2] - Tt.p[:2]) < 0.02
    assert np.degrees(np.linalg.norm(so3_log(T.R @ Tt.R.T))) < 0.5


def test_latency(campaign):
    _, truth, out = campaign
    assert out["phase2"]["latency"]["tau_d"] == pytest.approx(truth.tau_d, abs=0.006)
    assert out["phase4"]["tau_d_xcorr"] == pytest.approx(truth.tau_d, abs=0.006)
    assert out["phase4"]["pass"]


def test_static_field_kept(campaign):
    _, _, out = campaign
    sf = out["phase2"]["static_field"]
    assert sf["kept"]
    assert sf["cross_session_r2"]["along"] > 0.5


def test_fogm(campaign):
    _, truth, out = campaign
    f = out["phase2"]["fogm_moving"]
    for i, a in enumerate(["along", "cross", "vertical", "roll", "pitch", "yaw"]):
        assert f[a]["sigma"] == pytest.approx(truth.sigma[i], rel=0.25), a
        assert f[a]["tau_c"] == pytest.approx(truth.tau_c[i], rel=0.35), a
    assert f["along"]["nu"] is not None and f["along"]["nu"] < 10  # heavy tails found
    assert f["cross"]["nu"] is None  # Gaussian kept


def test_stationary(campaign):
    _, truth, out = campaign
    pooled = out["phase3"]["fogm_stationary_pooled"]
    assert pooled["along"]["sigma"] == pytest.approx(truth.sigma[0] * truth.stationary_scale, rel=0.35)


def test_events(campaign):
    _, _, out = campaign
    p5 = out["phase5"]
    assert p5["init"]["n_success"] == 6
    assert p5["lost_induced"]["n_trials"] == 1
    assert p5["lost_natural"]["n"] == 0


def test_validation_passes(campaign):
    # data generated from the Tier 0 model itself must pass the hold-out validation
    _, _, out = campaign
    assert out["phase6"]["pass"], out["phase6"]["failing_metrics"]


def test_outputs_written(campaign):
    root, _, _ = campaign
    res = root / "results"
    model = yaml.safe_load((res / "model.yaml").read_text())
    assert model["tau_d"] > 0
    assert (res / "report.md").exists()
    assert (res / "static_field_grid.npz").exists()
    json.loads((res / "results.json").read_text())
