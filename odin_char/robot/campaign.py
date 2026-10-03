"""Autonomous Odin1 Tier 0 data collection on the ARUW sentry.

    python -m odin_char.robot.campaign --config config/campaign.yaml --dir campaigns/odin1 <step>

Bring-up / debugging (nothing recorded; run these first on a new setup):
    mocap      print the live mocap stream for 10 s (rate, pose, clock offset, fence)
    odin       start the Odin driver for 20 s and report its topics/tf rates
    jog        small open-loop pulses; prints how the robot actually moved for each command

Steps (run in this order; `day1` and `day2` chain them):
    check      mocap live, robot inside the fence, bridge moves the robot (command-gain ID)
    preflight  Phase 0 IR interference (prompts to toggle mocap strobes and Odin power)
    calib      Phase 1a/1b: rotation-rich motion in odometry mode
    map        Phase 1c: SLAM-mode coverage + loop closures, map saved via /odin1/save_map
    session    Phase 2: one session (fresh relocalization start, 10 nominal laps, speed sweep)
    holds      Phase 3: 10 min stationary holds
    latency    Phase 4: sinusoidal translation and yaw
    init       Phase 5: 24 cold starts at offsets from the mapping path
    outage     Phase 5: covered-sensor outages (needs an operator or a shutter command)
    day1       check, preflight, calib, map, session, holds, latency, init, outage
    day2/day3  check, session        (the protocol wants sessions on different days)

Every run is recorded with `ros2 bag record` into <dir>/bags and listed in <dir>/manifest.json,
which `python -m odin_char.pipeline <dir>` analyses. Steps are resumable: finished steps are
skipped unless --redo is given.

Safety: the robot only moves while the remote is in left-UP/right-MID (ExternalVelocityCommand).
The script stops it on stale mocap (arrival > robot.mocap_timeout_s), a fence breach (including the
stopping point), or a large tracking error; the bridge and the MCB each stop it if commands stop.
Ctrl-C stops the robot and the driver.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from .motion import (
    Controller,
    ControllerConfig,
    Fence,
    Trajectory,
    calib_motion,
    figure8,
    fit_command_gains,
    follow_path,
    goto,
    hold,
    lawnmower_path,
    line_back_and_forth,
    sinusoid_axis,
    sysid_plan,
    wrap,
    yaw_sinusoid,
)
from .processes import BagRecorder, OdinDriver, sha256


class Abort(RuntimeError):
    pass


def beep(n: int = 1):
    for _ in range(n):
        sys.stdout.write("\a")
        sys.stdout.flush()
        time.sleep(0.15)


class Campaign:
    def __init__(self, cfg: dict, root: Path, prompts: bool, skip_warmup: bool, io=None):
        self.cfg = cfg
        self.root = root
        self.prompts = prompts
        self.skip_warmup = skip_warmup
        root.mkdir(parents=True, exist_ok=True)
        (root / "campaign.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        self.state_path = root / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"done": []}
        mp = root / "manifest.json"
        self.manifest = json.loads(mp.read_text()) if mp.exists() else {"runs": [], "init_trials": [], "outages": []}

        rb, room = cfg["robot"], cfg["room"]
        self.fence = Fence(np.array(room["fence_min"], float), np.array(room["fence_max"], float), room["margin"])
        self.ctl = Controller(ControllerConfig(
            kp_pos=rb["kp_pos"], ki_pos=rb["ki_pos"], kp_yaw=rb["kp_yaw"], max_speed=rb["max_speed"],
            max_accel=rb["max_accel"], max_yaw_rate=rb["max_yaw_rate"], max_yaw_accel=rb["max_yaw_accel"]))
        if "G" in self.state:
            self.ctl.G = np.array(self.state["G"])
        self.dt = 1.0 / rb["control_rate_hz"]

        if io is None:  # real robot
            from .ros_io import CommandSender, RosIO

            io = (RosIO(cfg), CommandSender(rb["bridge_host"], rb["bridge_port"]), OdinDriver(cfg), BagRecorder(cfg))
        self.ros, self.cmd, self.driver, self.rec = io  # injectable for the dry-run test
        self.gains_checked = False
        self.warm_since: float | None = None  # host time Odin started running in this invocation

    # ------------------------------------------------------------------ bookkeeping
    def save(self):
        self.state_path.write_text(json.dumps(self.state, indent=1))
        (self.root / "manifest.json").write_text(json.dumps(self.manifest, indent=1))

    def prompt(self, text: str) -> bool:
        """Ask the operator to do something physical. Returns False in --no-prompt mode."""
        if not self.prompts:
            print(f"[skipped, --no-prompt] {text}")
            return False
        beep(2)
        input(f"\n>>> {text}\n    press Enter when done... ")
        return True

    def new_run(self, name: str, phase: str, **meta) -> dict:
        rid = len(self.manifest["runs"])
        run = {"id": rid, "name": f"{rid:03d}_{name}", "phase": phase, "bag": f"bags/{rid:03d}_{name}",
               "started": time.time(), **meta}
        self.manifest["runs"].append(run)
        return run

    def metadata_row(self, run: dict, speed=None):
        path = self.root / "metadata.csv"
        new = not path.exists()
        drv = self.cfg["driver"]
        map_path = self.state.get("map_path")
        row = {
            "date": datetime.now().isoformat(timespec="seconds"),
            "run": run["name"],
            "phase": run["phase"],
            "session": run.get("session"),
            "device_serial": self.driver.device_serial(),
            "firmware": self.cfg.get("metadata", {}).get("firmware", ""),
            "driver_version": self.cfg.get("metadata", {}).get("driver_version", ""),
            "map_sha256": self.state.get("map_sha256", "") if map_path else "",
            "speed_mps": speed,
            "people_present": self.cfg.get("metadata", {}).get("people_present", ""),
            "operator_spot": self.cfg["phases"]["phase2"].get("operator_spot", ""),
            "warmup_min": round((run["started"] - self.warm_since) / 60, 1) if self.warm_since else 0,
            "config_copy": run.get("driver_config", ""),
        }
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    # ------------------------------------------------------------------ motion
    def pose(self) -> np.ndarray:
        m = self.ros.mocap()
        if m is None:
            raise Abort("no mocap data: is the mocap -> base_link transform being published?")
        pose, age = m
        if age > self.cfg["robot"]["mocap_timeout_s"]:
            raise Abort(f"mocap stale ({age * 1000:.0f} ms)")
        return pose

    def stop_robot(self):
        self.cmd.stop()

    def execute(self, tr: Trajectory, label: str = ""):
        """Track a reference on mocap feedback. Raises Abort (robot stopped) on any safety trip."""
        if not self.fence.trajectory_inside(tr):
            raise Abort(f"trajectory '{label or tr.label}' leaves the fence; adjust config")
        if tr.max_speed() > self.cfg["robot"]["max_speed"] + 1e-6:
            raise Abort(f"trajectory '{label}' exceeds robot.max_speed")
        pose = self.pose()
        if np.linalg.norm(pose[:2] - tr.start[:2]) > 0.15 or abs(wrap(pose[2] - tr.start[2])) > 0.2:
            self.execute_raw(self.goto(tr.start), "approach")
        self.execute_raw(tr, label)

    def execute_raw(self, tr: Trajectory, label: str):
        rb = self.cfg["robot"]
        self.ros.event(f"motion_start {label} {tr.duration:.1f}s")
        self.ctl.reset()
        t0 = time.monotonic()
        next_tick = t0
        try:
            while True:
                t = time.monotonic() - t0
                if t > tr.duration + 0.5:
                    break
                pose = self.pose()
                if not self.fence.inside(pose):
                    raise Abort(f"fence breach at {pose[:2].round(2)}")
                v = self.ros.mocap_velocity()
                if v is not None and not self.fence.stop_point_inside(pose, v[:2], rb["max_accel"]):
                    raise Abort(f"stopping point outside fence at {pose[:2].round(2)}")
                ref = tr.sample(t)
                cmd, abort = self.ctl.update(t, pose, ref, self.dt)
                if abort:
                    raise Abort(abort)
                self.cmd.send(*cmd)
                self.ros.publish_cmd(cmd)
                next_tick += self.dt
                time.sleep(max(0.0, next_tick - time.monotonic()))
        finally:
            self.stop_robot()
        self.ros.event(f"motion_end {label}")
        time.sleep(0.5)

    def goto(self, goal) -> Trajectory:
        rb = self.cfg["robot"]
        return goto(self.pose(), goal, min(0.6, rb["max_speed"]), rb["max_accel"], 1.0, rb["max_yaw_accel"])

    def wait_still(self, seconds: float):
        t_end = time.time() + seconds
        while time.time() < t_end:
            self.pose()  # keeps checking mocap health
            self.stop_robot()
            time.sleep(0.2)

    def home(self):
        self.execute(self.goto(self.cfg["room"]["home"]), "home")

    # ------------------------------------------------------------------ driver
    def start_driver(self, mode: int, run_dir: Path, tag: str):
        if self.driver.alive():
            self.driver.stop()
        conf = run_dir / f"control_command_{tag}.yaml"
        map_path = Path(self.state["map_path"]) if mode == 2 else None
        if mode == 2 and (map_path is None or not map_path.exists()):
            raise Abort("relocalization needs a map: run the 'map' step first")
        map_dest = self.root / "maps" if mode == 1 else None
        self.driver.write_config(conf, mode, reloc_map=map_path, map_dest=map_dest, map_name="map.bin")
        self.ros.reset_map_watch()
        self.driver.start(conf, run_dir / f"driver_{tag}.log", mode)
        t_end = time.time() + self.cfg["driver"]["connect_timeout_s"]
        while time.time() < t_end:
            if self.ros.last_imu_arrival and self.ros.last_imu_arrival > self.driver.started_at:
                return conf
            if not self.driver.alive():
                raise Abort(f"odin driver exited during start; see {self.driver.log_path}")
            time.sleep(0.2)
        raise Abort("odin driver did not publish IMU data; is Odin connected?")

    def warmup(self, mode: int):
        """Protocol: >= warmup_min of Odin running before every collection, identically each time.

        Warm time is counted from the first driver start of this invocation (the device stays
        powered across driver restarts and mode changes); powering Odin off resets it."""
        need = self.cfg["driver"]["warmup_min"] * 60
        if not self.driver.alive() or self.driver.mode != mode:
            self.start_driver(mode, self.root / "logs", f"warmup_{int(time.time())}")
        if self.warm_since is None:
            self.warm_since = self.driver.started_at
        remaining = need - (time.time() - self.warm_since)
        if remaining <= 0:
            return
        if self.skip_warmup:
            print("[warm-up skipped by --skip-warmup]")
            return
        print(f"warming Odin up: {remaining / 60:.1f} min to go ...")
        self.wait_still(remaining)

    def relocalize(self, timeout: float) -> float | None:
        """Wait for the first odom->map (or the driver's success line). Returns host time."""
        t_end = time.time() + timeout
        while time.time() < t_end:
            if self.ros.first_map_tf_arrival is not None:
                return self.ros.first_map_tf_arrival
            if "relocalized" in self.driver.events:
                return self.driver.events["relocalized"]
            if not self.driver.alive():
                raise Abort("odin driver exited")
            self.pose()
            time.sleep(0.05)
        return None

    def fresh_relocalized_start(self, run_dir: Path, tag: str) -> tuple[float, float | None]:
        """Fresh driver start in mode 2. If no fix while still, do the standard motion once."""
        self.start_driver(2, run_dir, tag)
        t_launch = self.driver.started_at
        tout = self.cfg["phases"]["phase5"]["timeout_s"]
        fix = self.relocalize(tout)
        if fix is None:
            self.ros.event("no fix while stationary: standard motion")
            self.standard_motion()
            fix = self.relocalize(tout)
        return t_launch, fix

    def save_map_into_campaign(self) -> Path:
        """Save Odin's SLAM map and keep a copy at <campaign>/maps/map.bin (the driver logs where it
        really wrote the file; it may differ from the configured folder)."""
        print("saving map (the device generates it first; this can take a few minutes) ...")
        saved = self.driver.save_map()
        target = self.driver.event_lines.get("map_save_target", "")
        if saved is None or not saved.exists():
            raise Abort(f"map save failed ({target or 'no save started'}); see {self.driver.log_path}")
        mp = self.root / "maps" / "map.bin"
        mp.parent.mkdir(parents=True, exist_ok=True)
        if saved.resolve() != mp.resolve():
            shutil.copy2(saved, mp)
        print(f"map saved: {saved}" + ("" if saved.resolve() == mp.resolve() else f" (copied to {mp})"))
        return mp

    def standard_motion(self):
        p = self.pose()
        fwd = [p[0] + np.cos(p[2]), p[1] + np.sin(p[2]), p[2]]
        if not self.fence.inside(fwd):
            fwd = [p[0] - np.cos(p[2]), p[1] - np.sin(p[2]), p[2]]
        self.execute(self.goto(fwd), "standard_motion_1m")
        self.execute(self.goto([fwd[0], fwd[1], fwd[2] + np.pi / 2]), "standard_motion_rot90")

    # ------------------------------------------------------------------ recording helper
    def record(self, run: dict, body, with_cloud: bool = False, speed=None):
        bag = self.root / run["bag"]
        try:
            self.rec.start(bag, with_cloud)
        except (RuntimeError, FileExistsError) as e:
            self.manifest["runs"].remove(run)
            self.save()
            raise Abort(f"could not start recording: {e}") from e
        self.ros.event(f"run_start {run['name']}")
        try:
            body()
        finally:
            self.ros.event(f"run_end {run['name']}")
            time.sleep(1.0)
            self.rec.stop()
            self.stop_robot()
            run["ended"] = time.time()
            self.save()
        self.metadata_row(run, speed)

    def done(self, step: str) -> bool:
        return step in self.state["done"]

    def mark(self, step: str):
        if step not in self.state["done"]:
            self.state["done"].append(step)
        self.save()

    # ============================================================== bring-up / debugging steps
    def wait_mocap(self, timeout: float = 5.0):
        t_end = time.time() + timeout
        while time.time() < t_end:
            if self.ros.mocap() is not None:
                return
            time.sleep(0.1)
        m = self.cfg["mocap"]
        raise Abort(f"no mocap samples within {timeout:.0f} s on {m['topic']} "
                    f"({m['source']}{', ' + str(m.get('parent_frame')) + ' -> ' + str(m.get('child_frame')) if m['source'] == 'tf' else ''}). "
                    "Run `scripts/run_campaign.sh find-mocap` and fix the mocap: block in the config.")

    def step_mocap(self, seconds: float = 10.0):
        """Print the live mocap stream: rate, pose, age, clock offset, fence status. Nothing moves."""
        self.wait_mocap()
        c0 = self.ros.mocap_count
        t0 = time.time()
        while time.time() - t0 < seconds:
            time.sleep(1.0)
            pose, age = self.ros.mocap()
            with self.ros._lock:
                ta, ts = self.ros.mocap_hist[-1][:2]
            rate = (self.ros.mocap_count - c0) / (time.time() - t0)
            print(f"rate {rate:6.1f} Hz | x {pose[0]:7.3f} y {pose[1]:7.3f} yaw {np.degrees(pose[2]):7.1f} deg | "
                  f"age {age * 1000:5.1f} ms | arrival(host)-stamp {time.time() - (time.monotonic() - ta) - ts:9.3f} s | "
                  f"{'inside' if self.fence.inside(pose) else 'OUTSIDE'} fence")
        if rate < 50:
            print("WARNING: mocap rate below 50 Hz (the protocol wants >= 100 Hz)")

    def step_odin(self, seconds: float = 20.0):
        """Start the Odin driver in odometry mode, report what it publishes, stop it. Nothing moves."""
        run_dir = self.root / "logs"
        self.start_driver(0, run_dir, f"odin_check_{int(time.time())}")
        print(f"driver up (log: {self.driver.log_path}); listening {seconds:.0f} s ...")
        i0, p0 = self.ros.imu_count, dict(self.ros.tf_pair_counts)
        time.sleep(seconds)
        print(f"  /odin1/imu: {(self.ros.imu_count - i0) / seconds:.0f} Hz")
        for pair, n in sorted(self.ros.tf_pair_counts.items()):
            print(f"  tf {pair[0]} -> {pair[1]}: {(n - p0.get(pair, 0)) / seconds:.0f} Hz")
        need = (self.cfg["frames"]["odin_odom"], self.cfg["frames"]["odin_imu"])
        if need not in self.ros.tf_pair_counts:
            print(f"WARNING: no {need[0]} -> {need[1]} tf: check send_odom_baselink_tf / frames in the config")
        print(f"  device serial (calib.yaml): {self.driver.device_serial()}")
        self.driver.stop()

    def step_jog(self, speed: float = 0.15, rate: float = 0.5, seconds: float = 0.6):
        """Open-loop pulses to check command directions: +x should move forward, +y left, +wz CCW.
        Mocap is still watched (stale / fence stop the robot)."""
        self.wait_mocap()
        self.prompt("Remote: left switch UP, right switch MID. Clear ~0.5 m around the robot.")
        rows = []
        for name, cmd in (("+vx (forward)", [speed, 0, 0]), ("-vx", [-speed, 0, 0]), ("+vy (left)", [0, speed, 0]),
                          ("-vy", [0, -speed, 0]), ("+wz (CCW)", [0, 0, rate]), ("-wz", [0, 0, -rate])):
            start = self.pose()
            t_end = time.monotonic() + seconds
            try:
                while time.monotonic() < t_end:
                    if not self.fence.inside(self.pose()):
                        raise Abort("fence")
                    self.cmd.send(*cmd)
                    time.sleep(self.dt)
            finally:
                self.stop_robot()
            time.sleep(0.8)
            end = self.pose()
            c, s = np.cos(start[2]), np.sin(start[2])
            d = end[:2] - start[:2]
            rows.append((name, c * d[0] + s * d[1], -s * d[0] + c * d[1], np.degrees(wrap(end[2] - start[2]))))
        print(f"\n{'command':16s} {'moved fwd m':>11s} {'moved left m':>12s} {'turned deg':>10s}")
        for r in rows:
            print(f"{r[0]:16s} {r[1]:11.3f} {r[2]:12.3f} {r[3]:10.1f}")
        print("expected: +vx -> +fwd, +vy -> +left, +wz -> +deg (CCW). Signs are also learned by `check`, "
              "but a robot that does not move at all means the bridge/MCB/remote are not set up.")

    # ============================================================== steps
    def step_check(self):
        """Mocap alive, inside fence, no stray driver, and identify command gains with mocap."""
        self.wait_mocap()
        pose = self.pose()
        print(f"mocap OK: robot at {pose.round(3)}")
        if not self.fence.inside(pose, self.fence.margin + 0.6):
            raise Abort("put the robot at least 1 m inside the fence before starting")
        if self.driver.already_running() and not self.driver.alive():
            raise Abort("a separate odin driver is running; stop it (the campaign starts its own)")
        # clock sanity: arrival vs server stamps (informative only)
        with self.ros._lock:
            h = list(self.ros.mocap_hist)
        if len(h) > 50:
            d = np.array([r[0] for r in h]) - np.array([r[1] for r in h])
            print(f"mocap server clock: arrival-stamp spread {np.ptp(d) * 1000:.1f} ms over {len(h)} msgs")
        if self.cfg["robot"].get("sysid", True):
            self.identify_gains()

    def identify_gains(self):
        print("identifying command -> velocity gains (small back-and-forth pulses) ...")
        self.prompt("Remote: left switch UP, right switch MID (external velocity mode). Keep clear of the robot.")
        ts, cmds, meas = [], [], []
        t0 = time.monotonic()
        try:
            for cmd, dur in sysid_plan():
                t_end = time.monotonic() + dur
                while time.monotonic() < t_end:
                    pose = self.pose()
                    if not self.fence.inside(pose):
                        raise Abort("fence breach during gain identification")
                    self.cmd.send(*cmd)
                    v = self.ros.mocap_velocity(0.06)
                    if v is not None:
                        c, s = np.cos(pose[2]), np.sin(pose[2])
                        ts.append(time.monotonic() - t0)
                        cmds.append(cmd)
                        meas.append([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]])
                    time.sleep(self.dt)
        finally:
            self.stop_robot()
        res = fit_command_gains(np.array(ts), np.array(cmds), np.array(meas))
        print(f"gains:\n{np.round(res.G, 3)}\ndelay {res.delay_s * 1000:.0f} ms, residual {res.residual:.3f}")
        if not res.ok:
            raise Abort(f"command-gain identification failed: {res.reason}")
        self.ctl.G = res.G
        self.state["G"] = res.G.tolist()
        self.state["G_delay_s"] = res.delay_s
        self.gains_checked = True
        self.save()

    def step_preflight(self):
        dur = self.cfg["phases"]["preflight"]["duration_s"]
        run_dir = self.root / "logs"
        self.warmup(0)
        r = self.new_run("preflight_mocap_on", "preflight", condition="mocap_on")
        self.record(r, lambda: self.wait_still(dur), with_cloud=True)
        # mocap side, Odin on (same scene) is the run above; reuse it as odin_on
        self.manifest["runs"].append({**r, "id": len(self.manifest["runs"]), "condition": "odin_on"})
        if self.prompt("Turn the mocap STROBES OFF (keep the robot still)."):
            r = self.new_run("preflight_mocap_off", "preflight", condition="mocap_off")
            self.rec.start(self.root / r["bag"], True)
            time.sleep(dur)  # no mocap: nothing to watch
            self.rec.stop()
            self.save()
            self.prompt("Turn the mocap strobes back ON.")
        self.driver.stop()
        if self.prompt("Power Odin OFF (unplug it) for the mocap-side check."):
            r = self.new_run("preflight_odin_off", "preflight", condition="odin_off")
            self.record(r, lambda: self.wait_still(dur))
            self.prompt("Power Odin back ON.")
            self.warm_since = None  # device was off: warm up again
        self.start_driver(0, run_dir, f"after_preflight_{int(time.time())}")
        self.mark("preflight")

    def step_calib(self):
        ph = self.cfg["phases"]["calib"]
        self.warmup(0)
        center = self.cfg["room"]["home"]
        tr = calib_motion(center, ph["duration_s"], radius=0.8)
        r = self.new_run("calib", "calib")
        self.execute(self.goto(tr.start), "to_calib")
        self.record(r, lambda: self.execute(tr, "calib"))
        self.mark("calib")

    def step_map(self):
        ph = self.cfg["phases"]["mapping"]
        room = self.cfg["room"]
        lo = np.array(room["fence_min"]) + room["margin"] + 0.3
        hi = np.array(room["fence_max"]) - room["margin"] - 0.3
        path = lawnmower_path(lo, hi, ph["lane_spacing"], ph["loops"])
        tr = follow_path(path, ph["speed"], self.cfg["robot"]["max_accel"])
        self.execute(self.goto(tr.start), "to_map_start")
        run_dir = self.root / "logs"
        self.warmup(1)
        self.start_driver(1, run_dir, "mapping")  # fresh SLAM map, started at the path's start pose
        time.sleep(3.0)
        start_pose = self.pose()
        r = self.new_run("mapping", "mapping")
        self.record(r, lambda: self.execute(tr, "mapping"))
        mp = self.save_map_into_campaign()
        self.state.update(map_path=str(mp), map_sha256=sha256(mp), mapping_start_pose=start_pose.tolist(),
                          mapping_trajectory=np.column_stack([tr.xy, tr.yaw])[::20].tolist())
        self.driver.stop()
        self.mark("map")

    def step_session(self):
        p2 = self.cfg["phases"]["phase2"]
        sessions = sorted({r["session"] for r in self.manifest["runs"] if r["phase"] == "phase2"})
        sid = f"s{len(sessions) + 1}"
        print(f"=== Phase 2 session {sid} ===")
        if self.prompts:
            self.prompt(f"Operator: stand at the fixed spot ({p2['operator_spot']}) and stay there.")
        rot = np.radians(p2.get("rotation_deg", 0.0))
        a_max = self.cfg["robot"]["max_accel"]
        mk = lambda speed, laps: figure8(p2["center"], p2["half_length"], p2["half_width"], rot, laps, speed, a_max)
        nominal = mk(p2["nominal_speed"], p2["nominal_laps"])
        self.warmup(2)
        self.execute(self.goto(nominal.start), "to_figure8")
        run_dir = self.root / "logs"
        # fresh start inside the first bag: it is also a Phase 5 INIT trial
        r = self.new_run(f"{sid}_1x", "phase2", session=sid, speed_class="1x")
        r["driver_config"] = str(run_dir / f"control_command_{sid}.yaml")

        def first():
            t_launch, fix = self.fresh_relocalized_start(run_dir, sid)
            self.manifest["init_trials"].append({"run": r["id"], "offset_m": 0.0, "heading_deg": 0.0, "moving": False,
                                                 "t_launch": t_launch, "t_log_success": self.driver.events.get("relocalized"),
                                                 "timeout": self.cfg["phases"]["phase5"]["timeout_s"], "source": "session_start"})
            if fix is None:
                raise Abort("no relocalization fix at session start")
            self.wait_still(3.0)
            self.execute(nominal, "figure8_1x")

        self.record(r, first, speed=p2["nominal_speed"])
        for f in p2["sweep_factors"]:
            tr = mk(p2["nominal_speed"] * f, p2["sweep_laps"])
            r = self.new_run(f"{sid}_{f:g}x", "phase2", session=sid, speed_class=f"{f:g}x")
            self.execute(self.goto(tr.start), "to_figure8")
            self.record(r, lambda tr=tr, f=f: self.execute(tr, f"figure8_{f:g}x"), speed=p2["nominal_speed"] * f)
        self.home()
        self.mark(f"session_{sid}")

    def ensure_tracking(self):
        """Driver running in mode 2 with a fix."""
        if self.driver.alive() and self.driver.mode == 2 and self.ros.first_map_tf_arrival is not None:
            return
        self.warmup(2)
        if self.ros.first_map_tf_arrival is None or self.driver.mode != 2:
            _, fix = self.fresh_relocalized_start(self.root / "logs", f"reloc_{int(time.time())}")
            if fix is None:
                raise Abort("could not relocalize")

    def step_holds(self):
        ph = self.cfg["phases"]["phase3"]
        self.ensure_tracking()
        for i, loc in enumerate(ph["locations"]):
            self.execute(self.goto(loc), f"to_hold{i}")
            r = self.new_run(f"hold{i}", "phase3", location=f"L{i}")
            self.record(r, lambda: self.wait_still(ph["hold_s"]))
        self.home()
        self.mark("holds")

    def step_latency(self):
        ph = self.cfg["phases"]["phase4"]
        self.ensure_tracking()
        home = self.cfg["room"]["home"]
        tr = sinusoid_axis(home, home[2], ph["frequencies_hz"], ph["peak_speed"], ph["seconds_per_frequency"]) + \
            yaw_sinusoid(home, ph["frequencies_hz"], ph["peak_yaw_rate"], ph["seconds_per_frequency"])
        self.execute(self.goto(tr.start), "to_latency")
        r = self.new_run("latency", "phase4")
        self.record(r, lambda: self.execute(tr, "latency"))
        self.mark("latency")

    def init_start_poses(self) -> list[dict]:
        """Start poses whose *actual* distance to the mapping trajectory (nearest point) and heading
        difference to the mapping heading there match the requested grid. The achieved values are
        what gets logged, so the analysis regresses on what really happened."""
        ph = self.cfg["phases"]["phase5"]
        traj = np.array(self.state["mapping_trajectory"])  # (N, 3): x, y, yaw
        lo = self.fence.lo + self.fence.margin + 0.3
        hi = self.fence.hi - self.fence.margin - 0.3
        gx, gy = np.meshgrid(np.arange(lo[0], hi[0], 0.1), np.arange(lo[1], hi[1], 0.1))
        cand = np.stack([gx.ravel(), gy.ravel()], 1)
        d2 = ((cand[:, None, :] - traj[None, :, :2]) ** 2).sum(-1)
        near = np.argmin(d2, axis=1)
        dist = np.sqrt(d2[np.arange(len(cand)), near])
        centre = np.array(self.cfg["room"]["home"][:2])
        out = []
        for off in ph["offsets_m"]:
            ok = np.abs(dist - off) < 0.15
            if not ok.any():
                print(f"skipping INIT offset {off} m: no free space that far from the mapping path inside the fence")
                continue
            i = np.where(ok)[0][np.argmin(np.linalg.norm(cand[ok] - centre, axis=1))]
            for hd in ph["heading_offsets_deg"]:
                for beh in ph["start_behaviours"]:
                    yaw = traj[near[i], 2] + np.radians(hd)
                    out.append({"pose": [cand[i, 0], cand[i, 1], float(yaw)], "offset_m": round(float(dist[i]), 3),
                                "heading_deg": hd, "moving": beh == "moving", "requested_offset_m": off})
        return out

    def step_init(self):
        if "mapping_start_pose" not in self.state:
            raise Abort("run the 'map' step first (INIT offsets are relative to the mapping start)")
        ph = self.cfg["phases"]["phase5"]
        self.warmup(2)
        run_dir = self.root / "logs"
        trials = self.init_start_poses()
        done = {t.get("key") for t in self.manifest["init_trials"]}
        for k, tr in enumerate(trials):
            key = f"init_{tr['requested_offset_m']}m_{tr['heading_deg']}deg_{'moving' if tr['moving'] else 'still'}"
            if key in done:
                continue
            print(f"INIT trial {k + 1}/{len(trials)}: {key}")
            self.driver.stop()
            self.execute(self.goto(tr["pose"]), "to_init_pose")
            r = self.new_run(key, "init")

            def body(tr=tr, key=key, r=r):
                self.start_driver(2, run_dir, key)
                t_launch = self.driver.started_at
                if tr["moving"]:
                    time.sleep(1.0)
                    self.standard_motion()
                fix = self.relocalize(ph["timeout_s"] - (time.time() - t_launch))
                self.manifest["init_trials"].append({
                    "run": r["id"], "key": key, "offset_m": tr["offset_m"], "heading_deg": tr["heading_deg"],
                    "moving": tr["moving"], "t_launch": t_launch, "t_log_success": self.driver.events.get("relocalized"),
                    "timeout": ph["timeout_s"], "t_fix_tf": fix})
                if fix is not None:
                    self.wait_still(3.0)  # samples right after the fix: is it correct?

            self.record(r, body)
        self.driver.stop()
        self.home()
        self.mark("init")

    def step_outage(self):
        ph = self.cfg["phases"]["phase5"]
        auto = ph.get("shutter_cmd_cover") and ph.get("shutter_cmd_uncover")
        if not auto and not self.prompts:
            print("[outage step skipped: needs an operator (or set phases.phase5.shutter_cmd_*)]")
            return
        self.ensure_tracking()
        home = self.cfg["room"]["home"]
        done = {o.get("key") for o in self.manifest["outages"]}
        if not auto:
            self.prompt("Outage trials: stand next to the robot, ready to cover Odin's window with your hand. "
                        "Cover on 'COVER NOW', uncover on 'UNCOVER NOW'.")
        for dur in ph["cover_durations_s"]:
            for rep in range(ph["cover_repeats"]):
                key = f"outage_{dur}s_{rep}"
                if key in done:
                    continue
                tr = line_back_and_forth(home, 1.5, ph["cover_speed"], 0.5, dur + 15)
                self.execute(self.goto(tr.start), "to_outage")
                r = self.new_run(key, "outage")
                times = {}

                def cover_cycle():
                    time.sleep(5.0)
                    self._announce("COVER NOW", ph.get("shutter_cmd_cover"))
                    times["t_cover"] = time.time()
                    time.sleep(dur)
                    self._announce("UNCOVER NOW", ph.get("shutter_cmd_uncover"))
                    times["t_uncover"] = time.time()

                def body():
                    import threading

                    th = threading.Thread(target=cover_cycle, daemon=True)
                    th.start()
                    self.execute(tr, key)
                    th.join()
                    self.wait_still(5.0)

                self.record(r, body, with_cloud=bool(self.cfg.get("recording", {}).get("cloud_in_outages", False)))
                self.manifest["outages"].append({"run": r["id"], "key": key, "cover_s": dur, **times})
                self.save()
        self.home()
        self.mark("outage")

    def _announce(self, text: str, cmd: str | None):
        self.ros.event(text)
        if cmd:
            subprocess.run(cmd, shell=True, check=False)
        else:
            beep(3 if "UNCOVER" in text else 1)
            print(f"\n\n          ######## {text} ########\n", flush=True)

    # ------------------------------------------------------------------ main
    BRINGUP = ["survey", "mocap", "odin", "jog"]  # set-up / debugging aids, always run, never recorded

    def step_survey(self):
        raise Abort("survey is driven by hand: run it with robot.control: human")
    STEPS = BRINGUP + ["check", "preflight", "calib", "map", "session", "holds", "latency", "init", "outage"]
    CHAINS = {
        "day1": ["check", "preflight", "calib", "map", "session", "holds", "latency", "init", "outage"],
        "day2": ["check", "session"],
        "day3": ["check", "session"],
    }

    def run(self, step: str, redo: bool):
        steps = self.CHAINS.get(step, [step])
        for s in steps:
            if s == "check" and self.gains_checked:
                continue
            if s not in self.BRINGUP + ["check", "session"] and self.done(s) and not redo:
                print(f"[{s} already done; --redo to repeat]")
                continue
            if s == "session" and step in ("day2", "day3") and self.done(f"session_{step}"):
                print(f"[{step} session already recorded]")
                continue
            print(f"\n================ {s} ================")
            try:
                getattr(self, f"step_{s}")()
            except Abort as e:
                with open(self.root / "aborts.log", "a") as f:
                    pose = self.ros.mocap()
                    f.write(f"{datetime.now().isoformat(timespec='seconds')} step={s} chain={step} "
                            f"pose={None if pose is None else pose[0].round(3).tolist()} reason={e}\n")
                raise
            if s == "session" and step in self.CHAINS:
                self.mark(f"session_{step}")
            if s not in self.BRINGUP + ["check"] and not self.gains_checked and self.cfg["robot"].get("sysid", True) and "G" not in self.state:
                self.identify_gains()

    def close(self):
        try:
            self.stop_robot()
        finally:
            try:
                self.rec.stop()
            finally:
                self.driver.stop()
                self.ros.shutdown()


def load_config(path: Path) -> dict:
    """campaign.yaml, overlaid with room.yaml next to it (written by the `survey` step)."""
    cfg = yaml.safe_load(path.read_text())
    room = path.parent / "room.yaml"
    if room.exists():
        over = yaml.safe_load(room.read_text()) or {}
        cfg["room"].update(over.get("room", {}))
        for ph, vals in (over.get("phases") or {}).items():
            cfg["phases"].setdefault(ph, {}).update(vals)
        print(f"room/figure-8/hold spots from {room}")
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=Campaign.STEPS + list(Campaign.CHAINS))
    ap.add_argument("--config", type=Path, default=Path(__file__).parents[2] / "config" / "campaign.yaml")
    ap.add_argument("--dir", type=Path, required=True, help="campaign directory (created if missing)")
    ap.add_argument("--no-prompt", action="store_true", help="skip steps that need a person")
    ap.add_argument("--skip-warmup", action="store_true", help="skip the warm-up even if driver.warmup_min > 0")
    ap.add_argument("--redo", action="store_true", help="repeat steps already marked done")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if (args.dir / "campaign.yaml").exists() and args.config.resolve() != (args.dir / "campaign.yaml").resolve():
        old = yaml.safe_load((args.dir / "campaign.yaml").read_text())
        if old.get("frames") != cfg.get("frames") or old.get("mocap") != cfg.get("mocap"):
            sys.exit("frames/mocap differ from the campaign's saved config; use the same config for a campaign")
        shutil.copy(args.dir / "campaign.yaml", args.dir / f"campaign.yaml.{int(time.time())}.bak")
    control = cfg["robot"].get("control", "human")
    if control == "human":
        from .human import HumanCampaign as cls
    elif control == "autonomous":
        cls = Campaign
    else:
        sys.exit(f"robot.control must be 'human' or 'autonomous', got {control!r}")
    print(f"control: {control}")
    c = cls(cfg, args.dir, prompts=not args.no_prompt, skip_warmup=args.skip_warmup)
    c.config_dir = args.config.parent  # where `survey` writes room.yaml
    try:
        c.run(args.step, args.redo)
        print("\nstep finished. Analyse with: python -m odin_char.pipeline", args.dir)
    except Abort as e:
        print(f"\n!!! ABORTED: {e}\nThe robot was stopped. Fix the cause and rerun the same step.")
        sys.exit(2)
    except KeyboardInterrupt:
        print("\ninterrupted: robot stopped")
        sys.exit(130)
    finally:
        c.close()


if __name__ == "__main__":
    main()
