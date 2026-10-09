"""`ros2 bag record` (the sensor driver is in odin_char/sensors)."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import time
from pathlib import Path

from ..sensors import load_sensor
from ..sensors.base import stop_group


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class BagRecorder:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sensor = load_sensor(cfg)
        self.proc: subprocess.Popen | None = None
        self.path: Path | None = None

    def topics(self, with_cloud: bool) -> list[str]:
        tp = self.cfg["topics"]
        names = [tp["tf"], tp["tf_static"], *self.sensor.record_topics(with_cloud),
                 tp.get("rosout", "/rosout"), "/odin_char/event", "/odin_char/cmd"]
        if tp.get("markers"):
            names.append(tp["markers"])
        mocap_topic = self.cfg["mocap"]["topic"]
        if mocap_topic not in names:
            names.append(mocap_topic)
        return names

    def start(self, path: Path, with_cloud: bool = False):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(path.parent).free / 1e9
        min_gb = float(self.cfg.get("recording", {}).get("min_free_gb", 2.0))
        if free_gb < min_gb:
            raise RuntimeError(f"only {free_gb:.1f} GB free on {path.parent} (minimum {min_gb} GB): "
                               "move old bags off this machine before recording")
        cmd = ["ros2", "bag", "record", "-o", str(path)]
        qos = self.sensor.qos_overrides
        if qos and qos.exists():
            cmd += ["--qos-profile-overrides-path", str(qos)]
        cmd += self.topics(with_cloud)
        self.path = path
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, start_new_session=True)
        time.sleep(2.0)  # subscriptions come up before anything interesting happens
        if self.proc.poll() is not None:
            raise RuntimeError("ros2 bag record exited immediately")

    def stop(self):
        if self.proc is not None:
            stop_group(self.proc, timeout=30.0)
        self.proc = None
