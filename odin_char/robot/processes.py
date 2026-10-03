"""External processes: the Odin driver and `ros2 bag record`."""
from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

import yaml


def expand(p: str | None) -> Path | None:
    return None if p is None else Path(os.path.expanduser(str(p)))


def _stop_group(proc: subprocess.Popen, timeout: float = 15.0):
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class OdinDriver:
    """Starts/stops odin_ros_driver with a per-run copy of control_command.yaml."""

    def __init__(self, cfg: dict):
        self.cfg = cfg["driver"]
        self.proc: subprocess.Popen | None = None
        self.log_path: Path | None = None
        self.started_at: float | None = None
        self.mode: int | None = None
        self.events: dict[str, float] = {}
        self.event_lines: dict[str, str] = {}
        self._watch = {
            "map_save_target": "Map save triggered, will transfer to",
            "relocalized": self.cfg["relocalization_success_text"],
            "map_saved": self.cfg["map_saved_text"],
            "map_failed": "Map transfer failed",
        }

    @staticmethod
    def already_running() -> bool:
        return subprocess.run(["pgrep", "-f", "host_sdk_sample"], capture_output=True).returncode == 0

    # odin_ros_driver (0.14.4) ignores the `config_file` parameter: it always loads
    # <package source>/config/control_command.yaml (host_sdk_sample.cpp, main()). So each run's config
    # is written INTO that installed file. The user's original is backed up once and restored by
    # restore_config() (called when the campaign exits); runs are always built from the original.
    @property
    def installed_config(self) -> Path:
        return expand(self.cfg["base_config"])

    @property
    def original_backup(self) -> Path:
        p = self.installed_config
        return p.with_name(p.name + ".odin_char_original")

    def write_config(self, out: Path, mode: int, reloc_map: Path | None = None,
                     map_dest: Path | None = None, map_name: str | None = None) -> Path:
        installed, backup = self.installed_config, self.original_backup
        if not backup.exists():
            shutil.copy2(installed, backup)  # the user's own config, restored on exit
        base = yaml.safe_load(backup.read_text())
        keys = base.setdefault("register_keys", {})
        keys.update(self.cfg.get("overrides", {}))
        keys["custom_map_mode"] = int(mode)
        if reloc_map is not None:
            keys["relocalization_map_abs_path"] = str(reloc_map)
        # Let the driver save maps to its default <ws>/src/odin_ros_driver/map/<start time>/ folder;
        # save_map() reads the real path from the driver log and the campaign copies the file.
        keys["mapping_result_dest_dir"] = ""
        keys["mapping_result_file_name"] = ""
        text = yaml.safe_dump(base, sort_keys=False)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)  # archived with the run
        installed.write_text(text)  # what the driver actually reads
        return out

    def restore_config(self):
        """Put the user's original control_command.yaml back (no-op if nothing was changed)."""
        backup = self.original_backup
        if backup.exists():
            shutil.copy2(backup, self.installed_config)
            backup.unlink()

    def start(self, config_path: Path, log_path: Path, mode: int):
        if self.proc is not None and self.proc.poll() is None:
            raise RuntimeError("driver already running")
        if self.already_running():
            raise RuntimeError("another odin driver (host_sdk_sample) is running; stop it first")
        cmd = self.cfg["run_cmd"].format(config=str(config_path))
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.events = {}
        self.event_lines = {}
        self.mode = mode
        self.started_at = time.time()
        self.proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, start_new_session=True)
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        with open(self.log_path, "a") as log:
            for line in self.proc.stdout:
                now = time.time()
                log.write(f"{now:.6f} {line}")
                log.flush()
                for key, text in self._watch.items():
                    if text in line and key not in self.events:
                        self.events[key] = now
                        self.event_lines[key] = line.strip()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc is not None:
            _stop_group(self.proc)
        self.proc = None
        self.mode = None
        time.sleep(2.0)  # let the device release the USB link before a restart

    def wait_for(self, key: str, timeout: float) -> float | None:
        t_end = time.time() + timeout
        while time.time() < t_end:
            if key in self.events:
                return self.events[key]
            if not self.alive():
                raise RuntimeError(f"odin driver exited; see {self.log_path}")
            time.sleep(0.05)
        return None

    def save_map(self, timeout: float | None = None) -> Path | None:
        """Ask the device to generate and transfer its SLAM map. Returns the saved file (taken
        from the driver's own "Map transfer completed: <path>" line), or None on failure/timeout."""
        svc = self.cfg["save_map_service"]
        r = subprocess.run(["ros2", "service", "call", svc, "odin_ros_driver/srv/SaveMap", "{value: 1}"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0 or "success=True" not in r.stdout.replace(" ", ""):
            raise RuntimeError(f"save_map service call failed: {r.stdout} {r.stderr}")
        t_end = time.time() + (timeout or float(self.cfg.get("save_map_timeout_s", 600)))
        while time.time() < t_end:
            if "map_saved" in self.events:
                line = self.event_lines.get("map_saved", "")
                path = line.split(self.cfg["map_saved_text"], 1)[-1].strip(" :")
                return Path(path) if path else None
            if "map_failed" in self.events:
                return None
            time.sleep(0.5)
        return None

    def device_serial(self) -> str:
        p = expand(self.cfg.get("calib_yaml"))
        try:
            return p.read_text().splitlines()[0].strip() if p else "unknown"
        except (OSError, IndexError):
            return "unknown"


class BagRecorder:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None
        self.path: Path | None = None

    def topics(self, with_cloud: bool) -> list[str]:
        tp = self.cfg["topics"]
        rc = self.cfg.get("recording", {})
        names = [tp["tf"], tp["tf_static"], tp["odometry"], tp["imu"],
                 tp.get("rosout", "/rosout"), "/odin_char/event", "/odin_char/cmd"]
        if rc.get("odometry_highfreq", False):
            names.append(tp["odometry_highfreq"])
        if with_cloud and tp.get("cloud"):
            names.append(tp["cloud"])
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
        qos = expand(self.cfg["driver"].get("qos_overrides"))
        if qos and qos.exists():
            cmd += ["--qos-profile-overrides-path", str(qos)]
        cmd += self.topics(with_cloud)
        self.path = path
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     start_new_session=True)
        time.sleep(2.0)  # subscriptions come up before anything interesting happens
        if self.proc.poll() is not None:
            raise RuntimeError("ros2 bag record exited immediately")

    def stop(self):
        if self.proc is not None:
            _stop_group(self.proc, timeout=30.0)
        self.proc = None
