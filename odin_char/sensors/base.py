"""What the campaign and the analysis need from the sensor under test.

A sensor is a sub-package odin_char/sensors/<type>/ that exports `SENSOR`, a `Sensor` subclass. It
may also ship `defaults.yaml` (merged under the campaign's `sensor:` block) and `env.sh` (sourced by
scripts/run_campaign.sh to put the driver on the ROS path). See sensors/README.md.

Frames: S is the sensor body frame whose pose the sensor reports, O its odometry frame, M the map
frame it localises in (the prior map built in the `map` step).
"""
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from ..se3 import Pose, PoseSeries


def expand(p: str | None) -> Path | None:
    return None if p is None else Path(os.path.expanduser(str(p)))


def stop_group(proc: subprocess.Popen, timeout: float = 15.0):
    if proc.poll() is not None:
        return
    for sig, wait in ((signal.SIGINT, timeout), (signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(wait)
            return
        except subprocess.TimeoutExpired:
            continue


class Mode:
    """Driver modes the campaign asks for."""

    ODOMETRY = 0  # local odometry only (calibration, preflight)
    MAPPING = 1  # build a fresh map, saved with Driver.save_map()
    RELOCALIZE = 2  # localise in the saved map (everything that is scored)


class Driver:
    """Starts and stops the sensor's driver as a process group, logs its output with host
    timestamps, and records in `events` the first time each `watch` text appears in the log.

    Event keys the campaign reads: `relocalized` (fix in the prior map; an alternative to the
    sensor's fix tf), and `map_save_target` (shown when a map save fails).

    Override `prepare` when the driver needs a per-run config file, `save_map` when the sensor can
    build maps, and `device_serial` for the metadata."""

    def __init__(self, cfg: dict):  # the `driver:` part of the sensor block
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None
        self.log_path: Path | None = None
        self.config_path: Path | None = None
        self.started_at: float | None = None
        self.mode: int | None = None
        self.events: dict[str, float] = {}
        self.event_lines: dict[str, str] = {}
        self.watch: dict[str, str] = dict(cfg.get("watch") or {})  # event key -> log text

    def already_running(self) -> bool:
        """A driver started outside the campaign (pgrep -f driver.process_name)."""
        name = self.cfg.get("process_name")
        return bool(name) and subprocess.run(["pgrep", "-f", name], capture_output=True).returncode == 0

    def prepare(self, run_dir: Path, tag: str, mode: int, reloc_map: Path | None = None) -> Path | None:
        """Write whatever the driver reads for this run (archived under run_dir); returns its
        path, which `run_cmd` gets as {config}. The base driver needs nothing."""
        self.config_path = None
        return None

    def restore_config(self):
        """Undo anything prepare() changed outside the campaign directory (called on exit)."""

    def start(self, config_path: Path | None, log_path: Path, mode: int):
        if self.proc is not None and self.proc.poll() is None:
            raise RuntimeError("driver already running")
        if self.already_running():
            raise RuntimeError(f"another driver ({self.cfg.get('process_name')}) is running; stop it first")
        cmd = self.cfg["run_cmd"].format(config=str(config_path or ""), mode=mode)
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.events = {}
        self.event_lines = {}
        self.mode = mode
        self.started_at = time.time()
        # stdin=DEVNULL: child processes must never read the operator's keystrokes (Enter)
        self.proc = subprocess.Popen(shlex.split(cmd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        with open(self.log_path, "a") as log:
            for line in self.proc.stdout:
                now = time.time()
                log.write(f"{now:.6f} {line}")
                log.flush()
                for key, text in self.watch.items():
                    if text in line and key not in self.events:
                        self.events[key] = now
                        self.event_lines[key] = line.strip()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc is not None:
            stop_group(self.proc)
        self.proc = None
        self.mode = None
        time.sleep(2.0)  # let the device release its link before a restart

    def wait_for(self, key: str, timeout: float) -> float | None:
        t_end = time.time() + timeout
        while time.time() < t_end:
            if key in self.events:
                return self.events[key]
            if not self.alive():
                raise RuntimeError(f"driver exited; see {self.log_path}")
            time.sleep(0.05)
        return None

    def save_map(self, timeout: float | None = None) -> Path | None:
        """Save the map built in Mode.MAPPING; returns the file, or None on failure/timeout."""
        raise RuntimeError("this sensor's driver cannot save maps")

    def device_serial(self) -> str:
        return "unknown"


class Sensor:
    """The sensor under test. `cfg` is the campaign's resolved `sensor:` block."""

    name = "sensor"  # shown in prompts and the report; `name:` in the config overrides it
    driver_class = Driver

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.type = cfg.get("type", "")
        self.name = cfg.get("name") or self.name
        self.topics: dict = dict(cfg.get("topics") or {})
        self.warmup_min = float(cfg.get("warmup_min", 0))
        self.connect_timeout_s = float(cfg.get("connect_timeout_s", 60))
        self.qos_overrides = expand(cfg.get("qos_overrides"))

    # ------------------------------------------------------------------ analysis (bags, no ROS)
    def map_pose(self, bag, use_arrival: bool = False):
        """The sensor's estimate T_MS in its map frame at every output sample.
        Returns (t, T_MS, has_map, map_age): has_map is False where it had no fix in the prior
        map yet, map_age is the age (s) of the registration in use (inf without one)."""
        raise NotImplementedError

    def odometry(self, bag, use_arrival: bool = False) -> PoseSeries:
        """The smooth local estimate T_OS (calibration hand-eye; gaps mark driver restarts)."""
        raise NotImplementedError

    def registration(self, bag, use_arrival: bool = False) -> PoseSeries:
        """The odometry-to-map registration stream; each message is one map update. Raises
        KeyError when the bag has none (no fix, or a sensor without a separate registration)."""
        raise KeyError(f"{self.name} publishes no registration stream")

    # ------------------------------------------------------------------ live campaign (ROS)
    @property
    def heartbeat(self) -> tuple[str, str]:
        """(topic, message type) whose arrival means the driver is up and streaming."""
        return self.topics["imu"], "sensor_msgs/msg/Imu"

    fix_tf: tuple[str, str] | None = None  # (parent, child) on /tf that appears once it has a fix
    odometry_tf: tuple[str, str] | None = None  # checked by the `sensor` bring-up step

    def record_topics(self, with_cloud: bool) -> list[str]:
        """The sensor's topics for `ros2 bag record` (/tf and the mocap are added by the campaign)."""
        tp = self.topics
        names = [tp[k] for k in ("imu", "odometry") if tp.get(k)] + list(tp.get("extra") or [])
        if with_cloud and tp.get("cloud"):
            names.append(tp["cloud"])
        return names

    def owns(self, topic: str = "", frame: str = "") -> bool:
        """True for the sensor's own topics/frames (`find-mocap` never suggests them)."""
        ns = self.cfg.get("namespace")
        return bool((ns and topic.startswith(ns)) or (frame and frame in self.own_frames))

    @property
    def own_frames(self) -> set[str]:
        return set(self.cfg.get("other_frames") or [])

    def log_issue(self, line: str) -> tuple[str, bool] | None:
        """(what, serious) for a driver log line reporting a problem (scripts/check_runs.py)."""
        return None

    def make_driver(self) -> Driver:
        return self.driver_class(dict(self.cfg.get("driver") or {}))


class TfSensor(Sensor):
    """A sensor that publishes odom -> body and odom -> map on /tf (frames in `sensor.frames`).

    `odom_map_is_ros_convention: false` means the odom -> map stream is T_OM, the pose of the map
    frame in odom (parent `odom`, child `map`), the inverse of ROS's usual map -> odom; hence
    T_MS = T_OM^-1 @ T_OS."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        fr = cfg["frames"]
        self.frames = fr
        self.fix_tf = (fr["odom"], fr["map"])
        self.odometry_tf = (fr["odom"], fr["body"])

    @property
    def own_frames(self) -> set[str]:
        fr = self.frames
        return super().own_frames | {fr["odom"], fr["body"], fr["map"]}

    def odometry(self, bag, use_arrival: bool = False) -> PoseSeries:
        return bag.tf_series(self.frames["odom"], self.frames["body"], use_arrival)

    def registration(self, bag, use_arrival: bool = False) -> PoseSeries:
        return bag.tf_series(self.frames["odom"], self.frames["map"], use_arrival)

    def map_pose(self, bag, use_arrival: bool = False):
        """T_MS at every odom -> body sample, holding the latest odom -> map (zero-order hold)."""
        odom = self.odometry(bag, use_arrival)
        try:
            reg = self.registration(bag, use_arrival)
        except KeyError:
            n = len(odom)
            return odom.t, odom.pose, np.zeros(n, bool), np.full(n, np.inf)
        T_OM, has_map, age = reg.zoh(odom.t)
        if self.frames.get("odom_map_is_ros_convention", False):
            # driver publishes map->odom the ROS way: the stream is T_MO, so T_OM is its inverse
            T_OM = T_OM.inv()
        T_MS: Pose = T_OM.inv() @ odom.pose
        return odom.t, T_MS, has_map, np.where(has_map, age, np.inf)
