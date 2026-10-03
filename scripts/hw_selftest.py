"""Hardware self-test on the robot computer, without mocap (nothing is commanded).

    .venv/bin/python scripts/hw_selftest.py [--dir /tmp/odin_itest]
    (run with ROS 2 and the odin_ros_driver workspace sourced; scripts/run_campaign.sh does that)

Uses the campaign's own code to: start Odin in SLAM mode, record a bag (with the point cloud),
save the map via /odin1/save_map, restart Odin in relocalization mode on that map, wait for the
fix, record again, and read both bags back with the analysis reader.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("/tmp/odin_itest"))
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "campaign.yaml")
    args = ap.parse_args()
    import sys

    sys.path.insert(0, str(ROOT))
    from odin_char.bagio import read_bag
    from odin_char.robot.human import HumanCampaign

    cfg = yaml.safe_load(args.config.read_text())
    if args.dir.exists():
        shutil.rmtree(args.dir)  # only this self-test's own scratch folder
    results = []

    def check(name, ok, detail=""):
        results.append((name, ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)

    c = HumanCampaign(cfg, args.dir, prompts=True, skip_warmup=True)
    logs = args.dir / "logs"
    try:
        # SLAM mode + recording + map save
        c.start_driver(1, logs, "itest_slam")
        check("driver starts in SLAM mode (IMU flowing)", True, f"log {c.driver.log_path}")
        bag1 = args.dir / "bags" / "slam"
        c.rec.start(bag1, with_cloud=True)
        time.sleep(8)
        c.rec.stop()
        check("bag recorded (SLAM)", (bag1 / "metadata.yaml").exists())
        ok = c.driver.save_map(timeout=300)
        mp = args.dir / "maps" / "map.bin"
        check("map saved via /odin1/save_map", ok and mp.exists(), f"{mp} {mp.stat().st_size if mp.exists() else 0} B")
        c.driver.stop()
        if not mp.exists():
            raise SystemExit("cannot continue without a map")
        c.state["map_path"] = str(mp)

        # relocalization mode on that map
        c.start_driver(2, logs, "itest_reloc")
        t0 = c.driver.started_at
        bag2 = args.dir / "bags" / "reloc"
        c.rec.start(bag2, with_cloud=False)
        fix = c.relocalize(60)
        time.sleep(3)
        c.rec.stop()
        check("relocalization fix detected", fix is not None,
              f"after {fix - t0:.1f} s (tf: {c.ros.first_map_tf_arrival is not None}, "
              f"log line: {'relocalized' in c.driver.events})" if fix else "within 60 s")
        c.driver.stop()

        # read back with the analysis reader
        b1, b2 = read_bag(bag1, cfg), read_bag(bag2, cfg)
        fr = cfg["frames"]
        n = lambda b, p: len(b.tf.get(p).t_stamp) if p in b.tf else 0
        check("bag: IMU", len(b1.imu_t) > 1000, f"{len(b1.imu_t)} msgs in 8 s")
        check("bag: odom->imu tf", n(b1, (fr["odin_odom"], fr["odin_imu"])) > 30, f"{n(b1, (fr['odin_odom'], fr['odin_imu']))}")
        check("bag: point cloud (preflight path)", len(b1.cloud_n) > 20, f"{len(b1.cloud_n)} clouds, ~{sum(b1.cloud_n) // max(len(b1.cloud_n), 1)} pts")
        check("bag: odom->map tf in relocalization", n(b2, (fr["odin_odom"], fr["odin_map"])) > 0,
              f"{n(b2, (fr['odin_odom'], fr['odin_map']))}")
        check("bag: mocap", len(b1.mocap.t_stamp) > 0, "(expected FAIL until the mocap is configured)")
        print("topics in bag:", ", ".join(sorted(b1.topics)))
    finally:
        c.close()
    bad = [n for n, ok in results if not ok and "mocap" not in n]
    print("\nself-test", "OK" if not bad else f"FAILED: {bad}")
    du = sum(f.stat().st_size for f in args.dir.rglob("*") if f.is_file()) / 1e6
    print(f"scratch output: {args.dir} ({du:.0f} MB)")


if __name__ == "__main__":
    main()
