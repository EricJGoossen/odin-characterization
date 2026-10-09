"""Find the mocap stream on the live ROS 2 graph.

    python -m odin_char.robot.discover [--seconds 5] [--config config/campaign.yaml]

Lists every topic whose type can carry a mocap pose (TFMessage, PoseStamped, TransformStamped,
Odometry), its rate, frame ids and the arrival-minus-stamp offset (a big offset means the stamps
come from another clock, i.e. mocap.clock: server), then prints a `mocap:` block to paste into
config/campaign.yaml for the best candidate. The sensor's own frames/topics are listed but not suggested.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import yaml

from ..mocap_config import MSGTYPES
from ..sensors import load_sensor


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--config", type=Path, default=Path(__file__).parents[2] / "config" / "campaign.yaml")
    args = ap.parse_args()
    sensor = load_sensor(yaml.safe_load(args.config.read_text()))

    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    from .ros_io import _msg_class

    rclpy.init()
    node = rclpy.create_node("odin_char_find_mocap")
    time.sleep(1.5)  # graph discovery
    by_type = {v: k for k, v in MSGTYPES.items()}  # ROS 2 reports e.g. "tf2_msgs/msg/TFMessage"
    topics = [(n, t[0]) for n, t in node.get_topic_names_and_types() if t and t[0] in by_type]
    if not topics:
        print("no TFMessage / PoseStamped / TransformStamped / Odometry topics visible. "
              "Is the mocap running, ROS_DOMAIN_ID the same, and the network reachable?")
        return

    stats: dict = defaultdict(lambda: {"n": 0, "first": None, "last": None, "offset": [], "frame": "", "child": ""})
    qos = QoSProfile(depth=500, reliability=ReliabilityPolicy.BEST_EFFORT)

    def cb_factory(topic, kind):
        def cb(msg):
            now = time.time()
            if kind == "tf":
                items = [((topic, t.header.frame_id.lstrip("/"), t.child_frame_id.lstrip("/")), t.header) for t in msg.transforms]
            else:
                child = getattr(msg, "child_frame_id", "")
                items = [((topic, msg.header.frame_id, child), msg.header)]
            for key, h in items:
                s = stats[key]
                s["n"] += 1
                s["first"] = s["first"] or now
                s["last"] = now
                s["offset"].append(now - (h.stamp.sec + h.stamp.nanosec * 1e-9))
        return cb

    subs = [node.create_subscription(_msg_class(MSGTYPES[by_type[t]]), n, cb_factory(n, by_type[t]), qos)
            for n, t in topics]
    print(f"listening to {len(subs)} candidate topics for {args.seconds:.0f} s ...")
    t_end = time.time() + args.seconds
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.05)

    rows = []
    for (topic, parent, child), s in stats.items():
        dur = max(args.seconds, 1e-3)
        off = sorted(s["offset"])
        rows.append((topic, parent, child, s["n"] / dur, off[len(off) // 2] if off else float("nan")))
    rows.sort(key=lambda r: -r[3])
    print(f"\n{'topic':36s} {'parent/frame_id':20s} {'child':20s} {'rate Hz':>8s} {'arrival-stamp':>14s}")
    for topic, parent, child, rate, off in rows:
        print(f"{topic:36s} {parent:20s} {child:20s} {rate:8.1f} {off:13.3f}s")

    kinds = {n: by_type[t] for n, t in topics}
    cand = [r for r in rows if not sensor.owns(frame=r[1]) and not sensor.owns(frame=r[2]) and r[3] > 20
            and not sensor.owns(topic=r[0]) and r[0] != "/tf_static"]
    if not cand:
        print(f"\nno obvious mocap candidate (> 20 Hz, not {sensor.name}). Pick from the table above.")
        return
    topic, parent, child, rate, off = cand[0]
    kind = kinds[topic]
    print("\nsuggested config (check the body is the robot!):\n\nmocap:")
    print(f"  source: {kind}")
    print(f"  topic: {topic}")
    if kind == "tf":
        print(f"  parent_frame: {parent}")
        print(f"  child_frame: {child}")
    print(f"  clock: {'server' if abs(off) > 0.05 else 'host'}   # median arrival - stamp = {off:.3f} s")
    print("  qos: best_effort")
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
