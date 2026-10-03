"""Dry run of the whole campaign orchestration against a simulated sentry, mocap and Odin driver.

Runs in real time with a shrunken room/config (~3 min). Checks that every step completes, the
robot never leaves the fence, and the manifest is what the analysis expects.
"""
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from odin_char.robot.campaign import Campaign
from odin_char.robot.processes import OdinDriver
from odin_char.robot.ros_io import RosIO

G_TRUE = np.array([[0.85, 0.03, 0.0], [-0.02, 0.8, 0.0], [0.0, 0.0, 1.15]])


class FakeCmd:
    def __init__(self):
        self.last = np.zeros(3)
        self.t = 0.0

    def send(self, vx, vy, wz):
        self.last = np.array([vx, vy, wz])
        self.t = time.monotonic()

    def stop(self, repeats=5):
        self.send(0, 0, 0)


class FakeDriver(OdinDriver):
    def __init__(self, cfg, root):
        super().__init__(cfg)
        self.root = root
        self._alive = False
        self.starts = []

    @staticmethod
    def already_running():
        return False

    def start(self, config_path, log_path, mode):
        assert not self._alive
        assert yaml.safe_load(config_path.read_text())["register_keys"]["custom_map_mode"] == mode
        # what the real driver reads (it ignores config_path)
        assert yaml.safe_load(self.installed_config.read_text())["register_keys"]["custom_map_mode"] == mode
        self.events, self.mode, self.started_at, self.log_path = {}, mode, time.time(), log_path
        self._alive = True
        self.starts.append(mode)
        if mode == 2:
            t = self.started_at
            threading.Timer(0.8, lambda: self._alive and self.started_at == t and self.events.setdefault("relocalized", time.time())).start()

    def alive(self):
        return self._alive

    def stop(self):
        self._alive = False
        self.mode = None

    def save_map(self, timeout=600.0):
        # like the real driver, which may write somewhere other than the configured folder
        p = self.root / "driver_default_maps"
        p.mkdir(exist_ok=True)
        (p / "map_20261002.bin").write_bytes(b"map")
        self.event_lines["map_save_target"] = f"Map save triggered, will transfer to [{p}/map_20261002.bin]"
        return p / "map_20261002.bin"


class FakeRec:
    def start(self, path, with_cloud=False):
        path.mkdir(parents=True)
        self.path = path

    def stop(self):
        pass


class World:
    """Kinematic sentry: body velocity = G_TRUE @ (command delayed 40 ms), watchdog 150 ms."""

    def __init__(self, cmd: FakeCmd, driver: FakeDriver, pose=(0.0, 0.0, 0.0)):
        self.cmd, self.driver = cmd, driver
        self.pose = np.array(pose, float)
        self._lock = threading.Lock()
        from collections import deque

        self.mocap_hist = deque(maxlen=2000)
        self.last_imu_arrival = None
        self.first_map_tf_arrival = None
        self.last_map_tf_arrival = None
        self.max_abs = np.zeros(2)
        self.events = []
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    mocap = RosIO.mocap
    mocap_velocity = RosIO.mocap_velocity
    reset_map_watch = RosIO.reset_map_watch

    def event(self, text):
        self.events.append(text)

    def publish_cmd(self, cmd):
        pass

    def shutdown(self):
        self._run = False

    def _loop(self):
        q, last = [], time.monotonic()
        while self._run:
            now = time.monotonic()
            dt, last = now - last, now  # integrate real elapsed time (sleep overshoots)
            c = self.cmd.last if now - self.cmd.t < 0.15 else np.zeros(3)
            q.append((now, c))
            while len(q) > 1 and now - q[1][0] >= 0.04:  # 40 ms actuation delay
                q.pop(0)
            act = G_TRUE @ (q[0][1] if now - q[0][0] >= 0.04 else np.zeros(3))
            cy, sy = np.cos(self.pose[2]), np.sin(self.pose[2])
            self.pose += dt * np.array([cy * act[0] - sy * act[1], sy * act[0] + cy * act[1], act[2]])
            self.max_abs = np.maximum(self.max_abs, np.abs(self.pose[:2]))
            with self._lock:
                self.mocap_hist.append((now, now + 100.0, *self.pose))  # server clock offset
            if self.driver.alive():
                self.last_imu_arrival = time.time()
                if self.driver.mode == 2 and "relocalized" in self.driver.events and self.first_map_tf_arrival is None:
                    self.first_map_tf_arrival = time.time()
            time.sleep(0.005)


@pytest.fixture
def small_cfg(tmp_path):
    cfg = yaml.safe_load((Path(__file__).parents[1] / "config" / "campaign.yaml").read_text())
    base = tmp_path / "control_command.yaml"
    base.write_text(yaml.safe_dump({"register_keys": {"custom_map_mode": 1, "use_host_ros_time": 0}}))
    cfg["driver"]["base_config"] = str(base)
    cfg["driver"]["warmup_min"] = 0
    cfg["room"] = {"fence_min": [-3.0, -2.0], "fence_max": [3.0, 2.0], "margin": 0.3, "home": [0.0, 0.0, 0.0]}
    ph = cfg["phases"]
    ph["preflight"]["duration_s"] = 1
    ph["calib"]["duration_s"] = 8
    ph["mapping"].update(lane_spacing=1.5, speed=1.0, loops=1)
    ph["phase2"].update(half_length=1.2, half_width=0.6, nominal_laps=1, sweep_laps=1, nominal_speed=1.0)
    ph["phase3"].update(locations=[[1.0, 0.5, 0.0], [-1.0, -0.5, 1.57]], hold_s=1)
    ph["phase4"].update(frequencies_hz=[1.0], seconds_per_frequency=4)
    ph["phase5"].update(offsets_m=[0.0, 0.5], heading_offsets_deg=[0, 10], timeout_s=5,
                        cover_durations_s=[1], cover_repeats=1, shutter_cmd_cover="true", shutter_cmd_uncover="true")
    return cfg


def test_day1_dry_run(tmp_path, small_cfg):
    root = tmp_path / "campaign"
    cmd = FakeCmd()
    driver = FakeDriver(small_cfg, root)
    world = World(cmd, driver)
    c = Campaign(small_cfg, root, prompts=False, skip_warmup=True, io=(world, cmd, driver, FakeRec()))
    try:
        c.run("day1", redo=False)
        c.run("day2", redo=False)
    except Exception as e:
        log = root / "aborts.log"
        raise AssertionError(f"{e!r}\n{log.read_text() if log.exists() else ''}") from e
    finally:
        c.close()

    m = json.loads((root / "manifest.json").read_text())
    phases = [r["phase"] for r in m["runs"]]
    for p in ("preflight", "calib", "mapping", "phase2", "phase3", "phase4", "init", "outage"):
        assert p in phases, p
    assert {r["session"] for r in m["runs"] if r["phase"] == "phase2"} == {"s1", "s2"}
    assert len([r for r in m["runs"] if r["phase"] == "phase2"]) == 6  # 1x, 0.5x, 1.5x per session
    # 2 offsets x 2 headings x 2 behaviours, plus one INIT trial per session start
    assert len(m["init_trials"]) == 8 + 2
    assert all(t["t_log_success"] is not None for t in m["init_trials"])
    assert len(m["outages"]) == 1 and m["outages"][0]["t_uncover"] > m["outages"][0]["t_cover"]
    st = json.loads((root / "state.json").read_text())
    assert np.allclose(np.array(st["G"]), G_TRUE, atol=0.08)
    # never left the fence
    assert np.all(world.max_abs <= np.array([3.0, 2.0]) - 0.15)
    assert (root / "metadata.csv").exists()
    # every relocalization run used a fresh driver start in mode 2
    assert driver.starts.count(2) >= 10
