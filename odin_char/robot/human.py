"""Human-driven campaign: a person drives the sentry with the remote; the script does the rest.

The script starts/stops the Odin driver, records every run, tells the driver what to do and shows
live mocap feedback (speed, distance, laps, map coverage, distance/heading to a target). It never
sends velocity commands, so neither otto's odin-characterization mode nor the MCB branch is needed.

Select it with `robot.control: human` in campaign.yaml (the default). The recorded data and the
manifest are the same as the autonomous campaign's, so `python -m odin_char.pipeline` is unchanged.
"""
from __future__ import annotations

import select
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .campaign import Abort, Campaign, beep
from .motion import figure8_path, wrap
from .processes import BagRecorder, OdinDriver


class NullCommand:
    """Stands in for the UDP command sender: in human mode nothing is ever commanded."""

    def send(self, vx, vy, wz):
        pass

    def stop(self, repeats=5):
        pass


class HumanCampaign(Campaign):
    TARGET_POS_TOL = 0.15  # m
    TARGET_YAW_TOL = np.radians(8)
    STILL_SPEED = 0.03  # m/s

    def __init__(self, cfg: dict, root: Path, prompts: bool, skip_warmup: bool, io=None):
        if not prompts:
            raise SystemExit("human mode needs a person at the keyboard: drop --no-prompt")
        if io is None:
            from .ros_io import RosIO

            io = (RosIO(cfg), NullCommand(), OdinDriver(cfg), BagRecorder(cfg))
        super().__init__(cfg, root, prompts, skip_warmup, io)
        self._stdin_ready = self._stdin_ready_default
        self._readline = sys.stdin.readline

    # ------------------------------------------------------------------ input / feedback helpers
    @staticmethod
    def _stdin_ready_default() -> bool:
        return bool(select.select([sys.stdin], [], [], 0)[0])

    @staticmethod
    def _flush_typeahead():
        """Drop keys typed before a prompt (an extra Enter must not confirm the NEXT prompt)."""
        try:
            import termios

            if sys.stdin.isatty():
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except (ImportError, OSError, ValueError):
            pass

    def ask(self, text: str):
        """Instruction that needs a confirmation before anything is recorded."""
        self._flush_typeahead()
        beep(2)
        print(f"\n>>> {text}\n    press Enter when ready... ", end="", flush=True)
        self._readline()

    def _sample(self) -> dict:
        m = self.ros.mocap()
        v = self.ros.mocap_velocity()
        if m is None:
            return {"pose": None, "age": np.inf, "v": None}
        return {"pose": m[0], "age": m[1], "v": v}

    def monitor(self, instructions: str, status, min_s: float = 0.0, target_s: float | None = None,
                done=None, track=None) -> float:
        """Show `instructions`, then a live status line until the operator presses Enter (after
        min_s) or `done(state)` becomes true. Returns the elapsed seconds.

        status(state) -> str; state carries pose, speed, distance, elapsed, and whatever `track`
        (called every tick with the state) accumulates."""
        self._flush_typeahead()
        beep(1)
        print(f"\n>>> {instructions}")
        print("    press Enter when finished" + (f" (at least {min_s:.0f} s)" if min_s else ""), flush=True)
        t0 = time.time()
        st = {"elapsed": 0.0, "dist": 0.0, "last_p": None, "target_beeped": False}
        while True:
            s = self._sample()
            st.update(s)
            st["elapsed"] = time.time() - t0
            p = s["pose"]
            if p is not None and s["age"] < 0.5:
                if st["last_p"] is not None:
                    st["dist"] += float(np.linalg.norm(p[:2] - st["last_p"][:2]))
                st["last_p"] = p
            st["speed"] = float(np.linalg.norm(s["v"][:2])) if s["v"] is not None else 0.0
            st["yaw_rate"] = float(s["v"][2]) if s["v"] is not None else 0.0
            if track:
                track(st)
            line = "mocap STALE" if p is None or s["age"] > 0.2 else status(st)
            if p is not None and not self.fence.inside(p, 0.0):
                line += "  !! OUTSIDE FENCE"
            if min_s and st["elapsed"] < min_s:
                line += f"  [Enter accepted in {min_s - st['elapsed']:.0f} s]"
            elif done is None:
                line += "  [press Enter to finish]"
            if target_s and st["elapsed"] >= target_s and not st["target_beeped"]:
                beep(2)
                st["target_beeped"] = True
                line += "  [target reached: press Enter]"
            width = max(20, shutil.get_terminal_size((120, 20)).columns - 5)  # never wrap
            sys.stdout.write("\r    " + line[:width].ljust(width))
            sys.stdout.flush()
            if done is not None and done(st):
                print()
                return st["elapsed"]
            if self._stdin_ready():
                self._readline()
                if st["elapsed"] >= min_s:
                    print()
                    return st["elapsed"]
                print(f"\n    keep going: at least {min_s - st['elapsed']:.0f} s more")
            time.sleep(0.1)

    def guide_to(self, target, what: str = "the target pose"):
        """Live distance/heading to `target` (x, y, yaw); returns once parked within tolerance."""
        target = np.asarray(target, float)
        still_since = {"t": None}

        def status(st):
            p = st["pose"]
            d = target[:2] - p[:2]
            c, s = np.cos(p[2]), np.sin(p[2])
            fwd, left = c * d[0] + s * d[1], -s * d[0] + c * d[1]
            dyaw = np.degrees(wrap(target[2] - p[2]))
            return (f"to target: {np.linalg.norm(d):5.2f} m  (ahead {fwd:+5.2f}, left {left:+5.2f})  "
                    f"turn {dyaw:+6.1f} deg {'(CCW)' if dyaw > 0 else '(CW)'}  speed {st['speed']:4.2f}")

        def done(st):
            p = st["pose"]
            if p is None or st["age"] > 0.2:
                return False
            ok = (np.linalg.norm(target[:2] - p[:2]) < self.TARGET_POS_TOL
                  and abs(wrap(target[2] - p[2])) < self.TARGET_YAW_TOL and st["speed"] < self.STILL_SPEED)
            if not ok:
                still_since["t"] = None
                return False
            still_since["t"] = still_since["t"] or time.time()
            return time.time() - still_since["t"] > 1.0

        self.monitor(f"Drive to {what}: x {target[0]:.2f}, y {target[1]:.2f}, yaw {np.degrees(target[2]):.0f} deg "
                     f"(stops automatically once parked within {self.TARGET_POS_TOL * 100:.0f} cm / "
                     f"{np.degrees(self.TARGET_YAW_TOL):.0f} deg)", status, done=done)

    def wait_still(self, seconds: float):
        """Hands off: countdown with a warning if the robot moves (mocap staleness only warned)."""
        t_end = time.time() + seconds
        while time.time() < t_end:
            s = self._sample()
            sp = float(np.linalg.norm(s["v"][:2])) if s["v"] is not None else 0.0
            msg = f"hold still: {t_end - time.time():5.0f} s left"
            if s["pose"] is None or s["age"] > 0.2:
                msg += "  (mocap stale)"
            elif sp > 0.05:
                msg += "  !! ROBOT MOVING"
            sys.stdout.write("\r    " + msg.ljust(60))
            sys.stdout.flush()
            time.sleep(0.2)
        print()

    def standard_motion(self):
        self.monitor("Standard start motion: drive forward ~1 m, then turn ~90 deg in place, then stop.",
                     lambda st: f"moved {st['dist']:4.2f} m  speed {st['speed']:4.2f}", min_s=2.0)

    def identify_gains(self):
        pass  # nothing is commanded in human mode

    # ------------------------------------------------------------------ steps
    def step_survey(self):
        """Drive around the edge of the usable floor; derive the fence, home, figure-8 and hold
        spots from where the robot actually went, and write them to config/room.yaml."""
        import yaml

        self.wait_mocap()
        path: list = []

        def track(st):
            p = st["pose"]
            if p is not None and st["age"] < 0.2:
                path.append(p.copy())

        def status(st):
            if not path:
                return "waiting for mocap"
            xy = np.array(path)[:, :2]
            lo, hi = xy.min(0), xy.max(0)
            return (f"x {lo[0]:+5.2f} .. {hi[0]:+5.2f}   y {lo[1]:+5.2f} .. {hi[1]:+5.2f}   "
                    f"area {hi[0] - lo[0]:4.1f} x {hi[1] - lo[1]:4.1f} m")

        self.monitor("Room survey: drive slowly all the way around the EDGE of the floor the robot may use "
                     "(as close to walls/obstacles as you'd let it go during the tests).",
                     status, min_s=15, track=track)
        xy = np.array(path)[:, :2]
        if len(xy) < 50:
            raise Abort("survey recorded too little mocap data")
        lo, hi = xy.min(0), xy.max(0)
        size, centre = hi - lo, (lo + hi) / 2
        if np.any(size < 2.5):
            print(f"WARNING: the surveyed area is small ({size[0]:.1f} x {size[1]:.1f} m); figure-8 and holds "
                  "will be cramped")
        rot = 0.0 if size[0] >= size[1] else 90.0
        L, S = size.max(), size.min()
        half_length = float(max(0.8, L / 2 - 0.7))
        half_width = float(max(0.4, min(S / 2 - 0.7, 0.6 * half_length)))
        q = 0.55 * size / 2
        holds = [[float(centre[0] + sx * q[0]), float(centre[1] + sy * q[1]), float(np.radians(yaw))]
                 for sx, sy, yaw in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, -90))]
        room = {
            "room": {"fence_min": [float(lo[0]), float(lo[1])], "fence_max": [float(hi[0]), float(hi[1])],
                     "margin": 0.2, "home": [float(centre[0]), float(centre[1]), 0.0]},
            "phases": {"phase2": {"center": [float(centre[0]), float(centre[1])], "half_length": half_length,
                                  "half_width": half_width, "rotation_deg": rot},
                       "phase3": {"locations": holds}},
        }
        out = Path(getattr(self, "config_dir", Path(__file__).parents[2] / "config")) / "room.yaml"
        out.write_text("# Written by `run_campaign.sh survey` "
                       f"({time.strftime('%Y-%m-%d %H:%M')}): room bounds from the driven edge,\n"
                       "# figure-8 and hold spots derived from them. Overrides campaign.yaml. Edit freely.\n"
                       + yaml.safe_dump(room, sort_keys=False))
        print(f"\nroom: x {lo[0]:+.2f}..{hi[0]:+.2f}, y {lo[1]:+.2f}..{hi[1]:+.2f} ({size[0]:.1f} x {size[1]:.1f} m)")
        print(f"figure-8: centre ({centre[0]:+.2f}, {centre[1]:+.2f}), {2 * half_length:.1f} x {2 * half_width:.1f} m, "
              f"rotated {rot:.0f} deg, one lap ~{self._lap_length_for(centre, half_length, half_width, rot):.1f} m")
        for i, h in enumerate(holds):
            print(f"hold {i + 1}: ({h[0]:+.2f}, {h[1]:+.2f}) facing {np.degrees(h[2]):.0f} deg")
        print(f"written to {out}. Mark the figure-8 with tape: it passes through the centre and reaches "
              f"{half_length:.1f} m either side along the {'x' if rot == 0 else 'y'} axis.")

    @staticmethod
    def _lap_length_for(centre, half_length, half_width, rot_deg) -> float:
        xy = figure8_path(centre, half_length, half_width, np.radians(rot_deg), 1)
        return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))

    def step_check(self):
        self.wait_mocap()
        pose = self.ros.mocap()[0]
        print(f"mocap OK: robot at x {pose[0]:.2f}, y {pose[1]:.2f}, yaw {np.degrees(pose[2]):.0f} deg "
              f"({'inside' if self.fence.inside(pose) else 'OUTSIDE'} the fence)")
        if self.driver.already_running() and not self.driver.alive():
            raise Abort("a separate odin driver is running; stop it (the campaign starts its own)")
        print("human mode: the robot is never commanded; drive it with the remote when asked.")

    def step_jog(self, *a, **k):
        print("jog is for autonomous mode (it sends velocity commands); nothing to do in human mode.")

    def step_calib(self):
        ph = self.cfg["phases"]["calib"]
        self.warmup(0)
        r = self.new_run("calib", "calib")
        self.record(r, lambda: self.monitor(
            "Calibration: drive around with LOTS of rotation for the next "
            f"{ph['duration_s'] / 60:.1f} min: spin in place both ways (fast and slow), swerve, change speed. "
            "Stay inside the fence.",
            lambda st: f"{st['elapsed']:5.0f}/{ph['duration_s']:.0f} s  yaw rate {st['yaw_rate']:+5.2f} rad/s  "
                       f"speed {st['speed']:4.2f} m/s",
            min_s=0.8 * ph["duration_s"], target_s=ph["duration_s"]))
        self.mark("calib")

    def step_map(self):
        ph = self.cfg["phases"]["mapping"]
        lo, hi, res = self.fence.lo + self.fence.margin, self.fence.hi - self.fence.margin, 0.5
        shape = np.maximum(np.ceil((hi - lo) / res).astype(int), 1)
        visited = np.zeros(shape, bool)
        path: list = []

        def track(st):
            p = st["pose"]
            if p is None or st["age"] > 0.2:
                return
            ij = np.floor((p[:2] - lo) / res).astype(int)
            if np.all(ij >= 0) and np.all(ij < shape):
                visited[ij[0], ij[1]] = True
            # keep a point every 5 cm or 5 deg (the INIT start poses are chosen from this path)
            if not path or np.hypot(p[0] - path[-1][0], p[1] - path[-1][1]) > 0.05 \
                    or abs(wrap(p[2] - path[-1][2])) > np.radians(5):
                path.append([float(p[0]), float(p[1]), float(p[2])])

        run_dir = self.root / "logs"
        self.warmup(1)
        self.ask("Mapping: park at a central spot, facing into the room. Odin will start a fresh SLAM map.")
        self.start_driver(1, run_dir, "mapping")
        time.sleep(3.0)
        start_pose = self.ros.mocap()[0]
        r = self.new_run("mapping", "mapping")
        self.record(r, lambda: self.monitor(
            f"Mapping: drive slowly (~{ph['speed']} m/s) covering the whole area in lanes ~{ph['lane_spacing']} m apart, "
            f"then {ph['loops']} loops around the edge. Aim for > 90% coverage.",
            lambda st: f"coverage {100 * visited.mean():5.1f}%  driven {st['dist']:6.1f} m  speed {st['speed']:4.2f} m/s",
            min_s=30, track=track))
        self.ask("Stop the robot and keep it still while the map is generated and saved.")
        mp = self.save_map_into_campaign()
        from .processes import sha256

        if len(path) < 10:
            raise Abort("no mocap path recorded during mapping (mocap stale?)")
        self.state.update(map_path=str(mp), map_sha256=sha256(mp), mapping_start_pose=start_pose.tolist(),
                          mapping_trajectory=path, mapping_coverage=float(visited.mean()))
        self.driver.stop()
        self.mark("map")

    def _lap_length(self) -> float:
        p2 = self.cfg["phases"]["phase2"]
        xy = figure8_path(p2["center"], p2["half_length"], p2["half_width"], np.radians(p2.get("rotation_deg", 0)), 1)
        return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))

    def _drive_laps(self, laps: int, speed: float, label: str) -> float:
        lap = self._lap_length()
        speeds: list = []

        def track(st):
            if st["speed"] > 0.05:
                speeds.append(st["speed"])

        self.monitor(
            f"{label}: drive {laps} figure-8 lap(s) through the tape marks at a STEADY ~{speed:.2f} m/s "
            f"(one lap is ~{lap:.1f} m).",
            lambda st: f"laps ~{st['dist'] / lap:4.1f}/{laps}  speed {st['speed']:4.2f} m/s (aim {speed:.2f})  "
                       f"mean {np.mean(speeds) if speeds else 0:4.2f}",
            min_s=0.8 * laps * lap / max(speed, 0.05), done=lambda st: st["dist"] >= laps * lap, track=track)
        return float(np.mean(speeds)) if speeds else 0.0

    def step_session(self):
        p2 = self.cfg["phases"]["phase2"]
        sessions = sorted({r["session"] for r in self.manifest["runs"] if r["phase"] == "phase2"})
        sid = f"s{len(sessions) + 1}"
        print(f"=== Phase 2 session {sid} ===")
        self.ask(f"Operator: stand at the fixed spot ({p2['operator_spot']}) except for the person driving; "
                 "keep everyone else still and out of Odin's view.")
        self.warmup(2)
        self.ask("Drive to the start of the figure-8 (tape) and stop.")
        run_dir = self.root / "logs"
        r = self.new_run(f"{sid}_1x", "phase2", session=sid, speed_class="1x")
        r["driver_config"] = str(run_dir / f"control_command_{sid}.yaml")
        measured = {}

        def first():
            t_launch, fix = self.fresh_relocalized_start(run_dir, sid)
            self.manifest["init_trials"].append({"run": r["id"], "offset_m": 0.0, "heading_deg": 0.0, "moving": False,
                                                 "t_launch": t_launch, "t_log_success": self.driver.events.get("relocalized"),
                                                 "timeout": self.cfg["phases"]["phase5"]["timeout_s"], "source": "session_start"})
            if fix is None:
                raise Abort("no relocalization fix at session start")
            measured["1x"] = self._drive_laps(p2["nominal_laps"], p2["nominal_speed"], "Nominal laps")

        self.record(r, first, speed=p2["nominal_speed"])
        r["measured_speed_mps"] = measured.get("1x")
        for f in p2["sweep_factors"]:
            rr = self.new_run(f"{sid}_{f:g}x", "phase2", session=sid, speed_class=f"{f:g}x")
            self.record(rr, lambda f=f: measured.__setitem__(
                f, self._drive_laps(p2["sweep_laps"], p2["nominal_speed"] * f, f"{'Slow' if f < 1 else 'Fast'} laps")),
                speed=p2["nominal_speed"] * f)
            rr["measured_speed_mps"] = measured.get(f)
        self.save()
        self.mark(f"session_{sid}")

    def step_validation(self):
        """Short held-out figure-8 run: a fresh relocalization start, then validation_laps steady laps.
        Recorded as the last Phase 2 session, which the analysis holds out for Phase 6 validation."""
        p2 = self.cfg["phases"]["phase2"]
        laps = int(p2.get("validation_laps", 5))
        sessions = sorted({r["session"] for r in self.manifest["runs"] if r["phase"] == "phase2"})
        sid = f"s{len(sessions) + 1}"
        print(f"=== validation run ({sid}, held out from fitting): {laps} laps ===")
        self.ask("Everyone except the driver back at the fixed spot, out of Odin's view.")
        self.warmup(2)
        self.ask("Drive to the start of the figure-8 (tape) and stop.")
        run_dir = self.root / "logs"
        r = self.new_run(f"{sid}_validation", "phase2", session=sid, speed_class="1x", validation=True)
        measured = {}

        def body():
            t_launch, fix = self.fresh_relocalized_start(run_dir, sid)
            self.manifest["init_trials"].append({"run": r["id"], "offset_m": 0.0, "heading_deg": 0.0, "moving": False,
                                                 "t_launch": t_launch, "t_log_success": self.driver.events.get("relocalized"),
                                                 "timeout": self.cfg["phases"]["phase5"]["timeout_s"], "source": "validation_start"})
            if fix is None:
                raise Abort("no relocalization fix at the validation start")
            measured["v"] = self._drive_laps(laps, p2["nominal_speed"], "Validation laps")

        self.record(r, body, speed=p2["nominal_speed"])
        r["measured_speed_mps"] = measured.get("v")
        self.save()
        self.mark("validation")

    def ensure_tracking(self):
        if self.driver.alive() and self.driver.mode == 2 and self.ros.first_map_tf_arrival is not None:
            return
        self.warmup(2)
        if self.ros.first_map_tf_arrival is None or self.driver.mode != 2:
            self.ask("Odin will relocalize: park somewhere you drove during mapping.")
            _, fix = self.fresh_relocalized_start(self.root / "logs", f"reloc_{int(time.time())}")
            if fix is None:
                raise Abort("could not relocalize")

    def step_holds(self):
        ph = self.cfg["phases"]["phase3"]
        self.ensure_tracking()
        held = {r.get("location") for r in self.manifest["runs"]
                if r["phase"] == "phase3" and r.get("ended") and r["ended"] - r["started"] >= 0.9 * ph["hold_s"]}
        for i, loc in enumerate(ph["locations"]):
            if f"L{i}" in held:
                print(f"[hold spot {i + 1} already recorded]")
                continue
            self.guide_to(loc, f"hold spot {i + 1}/{len(ph['locations'])}")
            print(f"    parked. Hands off for {ph['hold_s'] / 60:.0f} min (everyone stay still).")
            r = self.new_run(f"hold{i}", "phase3", location=f"L{i}")
            self.record(r, lambda: self.wait_still(ph["hold_s"]))
        self.mark("holds")

    def step_latency(self):
        """Two parts (translate, rotate), resumable per part: a run records which parts it completed
        (`latency_parts`); a restart only records the parts still missing."""
        ph = self.cfg["phases"]["phase4"]
        seg = len(ph["frequencies_hz"]) * ph["seconds_per_frequency"]
        done = set()
        for run in self.manifest["runs"]:
            if run["phase"] == "phase4" and not run.get("excluded"):
                done |= set(run.get("latency_parts", []))
        todo = [p for p in ("translate", "rotate") if p not in done]
        if not todo:
            self.mark("latency")
            return
        if done:
            print(f"[latency: {', '.join(sorted(done))} already recorded; doing {', '.join(todo)}]")
        self.ensure_tracking()
        self.ask("Latency test: park in an open area.")
        r = self.new_run("latency_" + "_".join(todo), "phase4", latency_parts=[])
        parts = {
            "translate": (f"Translate back and forth along the robot's forward axis (~20-40 cm, quick: 0.5 to 2 Hz, "
                          f"vary it) for {seg:.0f} s. No turning.",
                          lambda st: f"{st['elapsed']:4.0f}/{seg:.0f} s  speed {st['speed']:4.2f} m/s"),
            "rotate": (f"Rotate back and forth in place (+-30-60 deg, 0.5 to 2 Hz, vary it) for {seg:.0f} s. "
                       "No driving.",
                       lambda st: f"{st['elapsed']:4.0f}/{seg:.0f} s  yaw rate {st['yaw_rate']:+5.2f} rad/s"),
        }

        def body():
            for p in todo:
                text, status = parts[p]
                self.monitor(text, status, min_s=0.8 * seg, target_s=seg)
                r["latency_parts"].append(p)
                self.save()

        self.record(r, body)
        self.mark("latency")

    def _offset_from_map_path(self, pose) -> tuple[float, float]:
        traj = np.array(self.state["mapping_trajectory"])
        d = np.linalg.norm(traj[:, :2] - pose[:2], axis=1)
        k = int(np.argmin(d))
        return float(d[k]), float(np.degrees(abs(wrap(pose[2] - traj[k, 2]))))

    def step_init(self):
        if "mapping_trajectory" not in self.state:
            raise Abort("run the 'map' step first (INIT offsets are relative to the mapping path)")
        ph = self.cfg["phases"]["phase5"]
        self.warmup(2)
        run_dir = self.root / "logs"
        trials = self.init_start_poses()
        done = {t.get("key") for t in self.manifest["init_trials"]}
        for k, tr in enumerate(trials):
            key = f"init_{tr['requested_offset_m']}m_{tr['heading_deg']}deg_{'moving' if tr['moving'] else 'still'}"
            if key in done:
                continue
            print(f"\n=== INIT trial {k + 1}/{len(trials)}: {key} ===")
            self.driver.stop()
            self.guide_to(tr["pose"], "the INIT start pose")
            actual_off, actual_hd = self._offset_from_map_path(self.ros.mocap()[0])
            r = self.new_run(key, "init")

            def body(tr=tr, key=key, r=r, actual_off=actual_off, actual_hd=actual_hd):
                self.start_driver(2, run_dir, key)
                t_launch = self.driver.started_at
                if tr["moving"]:
                    self.standard_motion()
                else:
                    print("    hands off: waiting for the relocalization fix ...")
                fix = self.relocalize(max(1.0, ph["timeout_s"] - (time.time() - t_launch)))
                print("    fix after %.1f s" % (fix - t_launch) if fix else "    no fix before the timeout")
                self.manifest["init_trials"].append({
                    "run": r["id"], "key": key, "offset_m": actual_off, "heading_deg": actual_hd,
                    "requested_offset_m": tr["requested_offset_m"], "requested_heading_deg": tr["heading_deg"],
                    "moving": tr["moving"], "t_launch": t_launch, "t_log_success": self.driver.events.get("relocalized"),
                    "timeout": ph["timeout_s"], "t_fix_tf": fix})
                if fix is not None:
                    self.wait_still(3.0)

            self.record(r, body)
        self.driver.stop()
        self.mark("init")

    def relocalize(self, timeout: float) -> float | None:
        t_end = time.time() + timeout
        while time.time() < t_end:
            if self.ros.first_map_tf_arrival is not None:
                return self.ros.first_map_tf_arrival
            if "relocalized" in self.driver.events:
                return self.driver.events["relocalized"]
            if not self.driver.alive():
                raise Abort("odin driver exited")
            time.sleep(0.05)
        return None

    def step_outage(self):
        ph = self.cfg["phases"]["phase5"]
        auto = ph.get("shutter_cmd_cover") and ph.get("shutter_cmd_uncover")
        self.ensure_tracking()
        done = {o.get("key") for o in self.manifest["outages"]}
        self.ask("Outage trials: one person drives slowly back and forth; a second person covers Odin's window "
                 "on 'COVER NOW' and uncovers it on 'UNCOVER NOW'." if not auto else
                 "Outage trials: drive slowly back and forth when asked; the shutter is automatic.")
        for dur in ph["cover_durations_s"]:
            for rep in range(ph["cover_repeats"]):
                key = f"outage_{dur}s_{rep}"
                if key in done:
                    continue
                r = self.new_run(key, "outage")
                times = {}

                def cover_cycle(dur=dur):
                    time.sleep(5.0)
                    self._announce("COVER NOW", ph.get("shutter_cmd_cover"))
                    times["t_cover"] = time.time()
                    time.sleep(dur)
                    self._announce("UNCOVER NOW", ph.get("shutter_cmd_uncover"))
                    times["t_uncover"] = time.time()

                def body(dur=dur):
                    th = threading.Thread(target=cover_cycle, daemon=True)
                    th.start()
                    total = dur + 15
                    self.monitor(f"Trial {key}: drive slowly (~{ph['cover_speed']} m/s) back and forth for ~{total:.0f} s.",
                                 lambda st: f"{st['elapsed']:4.0f}/{total:.0f} s  speed {st['speed']:4.2f} m/s",
                                 done=lambda st: st["elapsed"] >= total and not th.is_alive())
                    th.join()
                    self.wait_still(5.0)

                self.record(r, body, with_cloud=bool(self.cfg.get("recording", {}).get("cloud_in_outages", False)))
                self.manifest["outages"].append({"run": r["id"], "key": key, "cover_s": dur, **times})
                self.save()
        self.mark("outage")
