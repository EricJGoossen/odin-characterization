"""LD19 data collection (ROS 2): driver, live SLAM map, bag recording. No mocap.

Steps (see __main__.py): check, static, map. Every recording goes to <dir>/bags and is listed in
<dir>/manifest.json, which analyze.py reads.
"""
from __future__ import annotations

import json
import select
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

from ..base import stop_group

HERE = Path(__file__).parent


class Procs:
    """Named external processes, each in its own process group with its output in logs/<name>.log."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.procs: dict[str, subprocess.Popen] = {}

    def start(self, name: str, cmd: str, params: dict | None = None, **fmt):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if params is not None:
            pf = self.log_dir / f"{name}_params.yaml"
            pf.write_text(yaml.safe_dump({"/**": {"ros__parameters": params}}, sort_keys=False))
            fmt["params"] = str(pf)
        log = open(self.log_dir / f"{name}.log", "a")
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {cmd.format(**fmt)}\n")
        log.flush()
        # stdin=DEVNULL: child processes must never read the operator's keystrokes (Enter)
        self.procs[name] = subprocess.Popen(shlex.split(cmd.format(**fmt)), stdin=subprocess.DEVNULL, stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)

    def dead(self) -> list[str]:
        return [n for n, p in self.procs.items() if p.poll() is not None]

    def stop(self):
        for name in reversed(list(self.procs)):
            stop_group(self.procs.pop(name), timeout=10.0)


class Live:
    """rclpy node watching the scan, the scan-matching odometry and the SLAM map."""

    def __init__(self, cfg: dict):
        import rclpy
        from nav_msgs.msg import OccupancyGrid, Odometry
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import LaserScan

        rclpy.init()
        self.node = rclpy.create_node("ld19_characterization")
        best = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.lock = threading.Lock()
        self.scan_t: list[float] = []  # arrival times (host)
        self.last_points = (0, 0)  # (valid, total) in the last scan
        self.dist = 0.0
        self.turned = 0.0
        self._last_odom = None
        self.map_known_m2 = 0.0
        self.map_occupied_m2 = 0.0
        self.node.create_subscription(LaserScan, cfg["scan_topic"], self._on_scan, best)
        self.node.create_subscription(Odometry, cfg["odom_topic"], self._on_odom, best)
        self.node.create_subscription(OccupancyGrid, cfg["map_topic"], self._on_map, latched)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        threading.Thread(target=self.executor.spin, daemon=True).start()

    def _on_scan(self, msg):
        r = np.asarray(msg.ranges, float)
        valid = int((np.isfinite(r) & (r > msg.range_min) & (r < msg.range_max)).sum())
        with self.lock:
            self.scan_t.append(time.time())
            self.last_points = (valid, len(r))

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        if self._last_odom is not None:
            x0, y0, yaw0 = self._last_odom
            self.dist += float(np.hypot(p.x - x0, p.y - y0))
            self.turned += float(abs(np.remainder(yaw - yaw0 + np.pi, 2 * np.pi) - np.pi))
        self._last_odom = (p.x, p.y, yaw)

    def _on_map(self, msg):
        d = np.asarray(msg.data)
        cell = msg.info.resolution ** 2
        self.map_known_m2 = float((d >= 0).sum() * cell)
        self.map_occupied_m2 = float((d > 50).sum() * cell)

    def scan_rate(self, window: float = 2.0) -> float:
        now = time.time()
        with self.lock:
            n = sum(1 for t in self.scan_t[-200:] if now - t < window)
        return n / window

    def scans_since(self, t0: float) -> int:
        with self.lock:
            return sum(1 for t in self.scan_t if t > t0)

    def shutdown(self):
        import rclpy

        self.executor.shutdown()
        self.node.destroy_node()
        rclpy.try_shutdown()


class Collector:
    def __init__(self, cfg: dict, root: Path, viewer: bool = True):
        self.cfg = cfg
        self.root = root
        self.viewer = viewer
        root.mkdir(parents=True, exist_ok=True)
        (root / "ld19.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))  # what analyze.py reads
        mp = root / "manifest.json"
        self.manifest = json.loads(mp.read_text()) if mp.exists() else {"runs": []}
        self.procs = Procs(root / "logs")
        self.live = Live(cfg)
        self.bag: subprocess.Popen | None = None

    # ------------------------------------------------------------------ helpers
    def save(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest, indent=1))

    def start_driver(self):
        name = self.cfg.get("driver_process")
        if name and subprocess.run(["pgrep", "-f", name], capture_output=True).returncode == 0:
            raise SystemExit(f"an LD19 driver ({name}) is already running; stop it (this tool starts its own)")
        t0 = time.time()
        self.procs.start("driver", self.cfg["driver_cmd"], self.cfg["driver_params"])
        while time.time() - t0 < self.cfg["connect_timeout_s"]:
            if self.live.scans_since(t0) >= 5:
                return
            if "driver" in self.procs.dead():
                raise SystemExit(f"LD19 driver exited; see {self.root / 'logs' / 'driver.log'}")
            time.sleep(0.2)
        raise SystemExit(f"no scans on {self.cfg['scan_topic']} within {self.cfg['connect_timeout_s']} s: "
                         f"check driver_params.port_name ({self.cfg['driver_params']['port_name']}) and the cable")

    def new_run(self, kind: str) -> dict:
        n = sum(1 for r in self.manifest["runs"] if r["kind"] == kind) + 1
        name = f"{kind}_{n:02d}"
        while (self.root / "bags" / name).exists():
            n += 1
            name = f"{kind}_{n:02d}"
        run = {"name": name, "kind": kind, "bag": f"bags/{name}", "started": time.time()}
        self.manifest["runs"].append(run)
        return run

    def record(self, run: dict, topics: list[str]):
        path = self.root / run["bag"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(path.parent).free < 1e9:
            raise SystemExit(f"less than 1 GB free on {path.parent}")
        self.bag = subprocess.Popen(["ros2", "bag", "record", "-o", str(path), *topics], stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        time.sleep(2.0)  # subscriptions come up first
        if self.bag.poll() is not None:
            raise SystemExit("ros2 bag record exited immediately")

    def stop_recording(self, run: dict):
        if self.bag is not None:
            stop_group(self.bag, timeout=30.0)
            self.bag = None
        run["ended"] = time.time()
        self.save()

    @staticmethod
    def _enter_pressed() -> bool:
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            return True
        return False

    def status(self, text: str):
        width = max(20, shutil.get_terminal_size((120, 20)).columns - 5)
        sys.stdout.write("\r    " + text[:width].ljust(width))
        sys.stdout.flush()

    def scan_status(self) -> str:
        v, n = self.live.last_points
        return f"scan {self.live.scan_rate():4.1f} Hz, {n} pts ({100 * v / max(n, 1):3.0f}% valid)"

    # ------------------------------------------------------------------ steps
    def step_check(self, seconds: float = 10.0):
        """Driver up, scan rate and points; nothing recorded."""
        self.start_driver()
        t0 = time.time()
        while time.time() - t0 < seconds:
            self.status(self.scan_status())
            time.sleep(0.5)
        print(f"\n{self.live.scans_since(t0) / seconds:.1f} scans/s over {seconds:.0f} s "
              "(LD19 nominal 10 Hz). Driver log: logs/driver.log")

    def step_static(self):
        """Hands-off recording with the lidar still: noise, dropouts, ghosts, timing."""
        secs = float(self.cfg["static_s"])
        self.start_driver()
        time.sleep(3.0)  # motor at speed
        run = self.new_run("static")
        print(f"recording {run['name']} for {secs:.0f} s: keep the lidar and the scene still ...")
        self.record(run, [self.cfg["scan_topic"], "/tf_static"])
        try:
            t0 = time.time()
            while time.time() - t0 < secs:
                self.status(f"{secs - (time.time() - t0):4.0f} s left | {self.scan_status()}")
                time.sleep(0.5)
        finally:
            self.stop_recording(run)
        print(f"\nsaved {run['bag']}")

    def step_map(self):
        """Live SLAM map while the robot is pushed around; Enter (or Ctrl-C) finishes."""
        c = self.cfg
        self.start_driver()
        self.procs.start("odometry", c["odometry_cmd"], c["odometry_params"])
        self.procs.start("slam", c["slam_cmd"], c["slam_params"])
        if self.viewer:
            self.procs.start("viewer", c["viewer_cmd"], rviz=str(HERE / "ld19.rviz"))
        run = self.new_run("map")
        self.record(run, [c["scan_topic"], c["odom_topic"], c["map_topic"], "/tf", "/tf_static"])
        print(f"\n>>> Push the robot slowly around the whole area; come back past where you started to close\n"
              f"    the loop. Watch the map grow in rviz{'' if self.viewer else ' (start it on your laptop)'}.\n"
              "    Press Enter when done.")
        try:
            while not self._enter_pressed():
                dead = [n for n in self.procs.dead() if n != "viewer"]
                if dead:
                    raise SystemExit(f"{', '.join(dead)} exited; see {self.root / 'logs'}")
                self.status(f"{self.scan_status()} | moved {self.live.dist:5.1f} m, turned "
                            f"{np.degrees(self.live.turned):6.0f} deg | map {self.live.map_known_m2:5.1f} m2 seen, "
                            f"{self.live.map_occupied_m2:4.1f} m2 obstacles")
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        print()
        out = self.root / "maps" / run["name"]
        out.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["ros2", "service", "call", c["save_map_service"], "slam_toolbox/srv/SerializePoseGraph",
                            f"{{filename: '{out}'}}"], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           timeout=60)
        print(f"slam_toolbox map {'saved to ' + str(out) + '.posegraph/.data' if r.returncode == 0 else 'NOT saved: ' + r.stderr.strip()}")
        time.sleep(2.0)  # the last /map in the bag
        self.stop_recording(run)
        print(f"saved {run['bag']}")

    def close(self):
        try:
            if self.bag is not None:
                stop_group(self.bag, timeout=30.0)
        finally:
            try:
                self.procs.stop()
            finally:
                self.live.shutdown()
