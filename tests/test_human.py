"""Human-driven campaign: orchestration with a simulated driver, and the live-feedback loop."""
import json
import time
from pathlib import Path

import numpy as np

from odin_char.robot.human import HumanCampaign
from odin_char.robot.motion import lawnmower_path

from .test_campaign_dryrun import FakeCmd, FakeDriver, FakeRec, World, small_cfg  # noqa: F401


class SimulatedHuman(HumanCampaign):
    """Replaces the person: instructions are 'followed' instantly by moving the simulated robot."""

    def ask(self, text):
        pass

    def prompt(self, text):
        return True

    def guide_to(self, target, what=""):
        self.ros.pose[:] = np.asarray(target, float)
        time.sleep(0.05)

    def monitor(self, instructions, status, min_s=0.0, target_s=None, done=None, track=None):
        lo, hi = self.fence.lo + self.fence.margin + 0.1, self.fence.hi - self.fence.margin - 0.1
        path = lawnmower_path(lo, hi, 1.0, 1)
        st = {"elapsed": 0.0, "dist": 0.0, "speed": 0.5, "yaw_rate": 0.3, "age": 0.0, "v": np.zeros(3),
              "last_p": None, "target_beeped": False}
        for k in range(1, len(path)):
            yaw = float(np.arctan2(*(path[k] - path[k - 1])[::-1]))
            self.ros.pose[:] = [path[k][0], path[k][1], yaw]
            st["pose"] = self.ros.pose.copy()
            st["dist"] += float(np.linalg.norm(path[k] - path[k - 1]))
            if track:
                track(st)
            assert isinstance(status(st), str)
            if done is not None and done(st):
                break
        self.ros.pose[:] = self.cfg["room"]["home"]
        time.sleep(0.15)  # let the mocap stream follow
        return 1.0


def test_human_day1_and_day2(tmp_path, small_cfg):
    small_cfg["robot"]["control"] = "human"
    small_cfg["phases"]["phase5"]["shutter_cmd_cover"] = None  # a person covers Odin
    small_cfg["phases"]["phase5"]["shutter_cmd_uncover"] = None
    root = tmp_path / "campaign"
    cmd = FakeCmd()
    driver = FakeDriver(small_cfg, root)
    world = World(cmd, driver)
    installed = Path(small_cfg["driver"]["base_config"])
    original = installed.read_text()
    c = SimulatedHuman(small_cfg, root, prompts=True, skip_warmup=True, io=(world, cmd, driver, FakeRec()))
    try:
        c.run("day1", redo=False)
        c.run("day2", redo=False)
    finally:
        c.close()
    # the driver reads the installed control_command.yaml: ours during runs, the user's after exit
    assert installed.read_text() == original
    assert not driver.original_backup.exists()
    assert len(list((root / "logs").glob("control_command_*.yaml"))) >= 10

    m = json.loads((root / "manifest.json").read_text())
    phases = [r["phase"] for r in m["runs"]]
    for p in ("preflight", "calib", "mapping", "phase2", "phase3", "phase4", "init", "outage"):
        assert p in phases, p
    p2 = [r for r in m["runs"] if r["phase"] == "phase2"]
    # day1: s1 (1x, 0.5x, 1.5x) + validation s2; day2: s3 (1x, 0.5x, 1.5x)
    assert {r["session"] for r in p2} == {"s1", "s2", "s3"} and len(p2) == 7
    assert [r["session"] for r in p2 if r.get("validation")] == ["s2"]
    assert {r["speed_class"] for r in p2} == {"1x", "0.5x", "1.5x"}
    assert len(m["init_trials"]) == 8 + 3
    real = [t for t in m["init_trials"] if t.get("key")]
    assert all("requested_offset_m" in t and t["offset_m"] >= 0 for t in real)
    assert len(m["outages"]) == 1 and m["outages"][0]["t_uncover"] > m["outages"][0]["t_cover"]
    st = json.loads((root / "state.json").read_text())
    assert len(st["mapping_trajectory"]) > 10 and st["mapping_coverage"] > 0.3
    assert "G" not in st  # nothing was ever commanded
    assert np.all(cmd.last == 0)


def test_survey_writes_room(tmp_path, small_cfg):
    import yaml

    from odin_char.robot.campaign import load_config

    root = tmp_path / "c"
    cmd = FakeCmd()
    driver = FakeDriver(small_cfg, root)
    world = World(cmd, driver)
    c = SimulatedHuman(small_cfg, root, prompts=True, skip_warmup=True, io=(world, cmd, driver, FakeRec()))
    cfgdir = tmp_path / "config"
    cfgdir.mkdir()
    (cfgdir / "campaign.yaml").write_text(yaml.safe_dump(small_cfg))
    c.config_dir = cfgdir
    try:
        # the simulated driver covers the fence area; the survey must find roughly those bounds
        orig = c.monitor

        def many_samples(*a, **k):
            for _ in range(30):
                orig(*a, **k)

        c.monitor = many_samples
        c.step_survey()
    finally:
        c.close()
    merged = load_config(cfgdir / "campaign.yaml")
    lo, hi = np.array(merged["room"]["fence_min"]), np.array(merged["room"]["fence_max"])
    assert np.all(lo < -1.5) and np.all(hi > [1.5, 0.9])
    assert len(merged["phases"]["phase3"]["locations"]) == 4
    p2 = merged["phases"]["phase2"]
    assert p2["half_length"] >= p2["half_width"] > 0 and p2["rotation_deg"] in (0.0, 90.0)
    for x, y, _ in merged["phases"]["phase3"]["locations"]:
        assert lo[0] < x < hi[0] and lo[1] < y < hi[1]


def test_monitor_feedback_loop(tmp_path, small_cfg):
    root = tmp_path / "c"
    cmd = FakeCmd()
    driver = FakeDriver(small_cfg, root)
    world = World(cmd, driver, pose=(0.3, -0.2, 0.0))
    c = HumanCampaign(small_cfg, root, prompts=True, skip_warmup=True, io=(world, cmd, driver, FakeRec()))
    try:
        time.sleep(0.2)
        # Enter before min_s is refused; a later Enter finishes
        presses = iter([0.2, 0.9])
        t0 = time.time()
        nxt = {"t": next(presses)}

        def ready():
            if nxt["t"] is not None and time.time() - t0 >= nxt["t"]:
                nxt["t"] = next(presses, None)
                return True
            return False

        c._stdin_ready = ready
        c._readline = lambda: "\n"
        el = c.monitor("test", lambda st: f"speed {st['speed']:.2f}", min_s=0.6)
        assert 0.6 <= el < 2.0
        # guide_to returns by itself once parked on target (no Enter needed)
        c._stdin_ready = lambda: False
        t1 = time.time()
        c.guide_to([0.3, -0.2, 0.0])
        assert time.time() - t1 < 3.0
    finally:
        c.close()
