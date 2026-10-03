"""ROS2 side of the campaign: mocap listener, command sender, bag markers.

Mocap staleness is judged on *arrival* time (host monotonic clock): the mocap stamps come from
the mocap server's clock and cannot be compared with the host clock directly.
"""
from __future__ import annotations

import math
import socket
import struct
import threading
import time
from collections import deque

import numpy as np

UDP_PACKET_FORMAT = "<Ifff"  # must match project-otto tools/chassis_velocity_bridge.py


def quat_yaw(x, y, z, w) -> float:
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class CommandSender:
    """UDP client for the chassis velocity bridge."""

    def __init__(self, host: str, port: int):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0

    def send(self, vx: float, vy: float, wz: float):
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        self.sock.sendto(struct.pack(UDP_PACKET_FORMAT, self.seq, vx, vy, wz), self.addr)

    def stop(self, repeats: int = 5):
        for _ in range(repeats):
            self.send(0.0, 0.0, 0.0)
            time.sleep(0.01)


def _msg_class(msgtype: str):
    """'geometry_msgs/msg/PoseStamped' -> the rclpy message class."""
    import importlib

    pkg, _, name = msgtype.split("/")
    return getattr(importlib.import_module(f"{pkg}.msg"), name)


class RosIO:
    """Owns the rclpy node (spun in a background thread)."""

    def __init__(self, cfg: dict):
        import rclpy
        from geometry_msgs.msg import TwistStamped
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Imu
        from std_msgs.msg import String
        from tf2_msgs.msg import TFMessage

        from ..mocap_config import MocapConfig

        self._TwistStamped, self._String = TwistStamped, String
        rclpy.init()
        self.node = rclpy.create_node("odin_characterization")
        fr, tp = cfg["frames"], cfg["topics"]
        self.mocap_cfg = MocapConfig.from_cfg(cfg)
        self.map_pair = (fr["odin_odom"], fr["odin_map"])
        self._lock = threading.Lock()
        self.mocap_hist: deque = deque(maxlen=2000)  # (t_arrival_monotonic, stamp_server, x, y, yaw)
        self.mocap_count = 0
        self.last_map_tf_arrival: float | None = None  # time.time()
        self.first_map_tf_arrival: float | None = None
        self.last_imu_arrival: float | None = None
        self.imu_count = 0
        self.tf_pair_counts: dict[tuple[str, str], int] = {}  # for the `odin` bring-up step

        rel = {"best_effort": ReliabilityPolicy.BEST_EFFORT, "reliable": ReliabilityPolicy.RELIABLE}
        mocap_qos = QoSProfile(depth=200, reliability=rel[self.mocap_cfg.qos])
        tf_topic = tp.get("tf", "/tf")
        shared = self.mocap_cfg.topic == tf_topic
        if shared and self.mocap_cfg.source != "tf":
            raise ValueError(f"mocap.topic is {tf_topic} but mocap.source is {self.mocap_cfg.source!r}")
        self._mocap_on_tf = shared
        # Odin's tf (odom->map) arrives on /tf; when the mocap also uses /tf, one subscription serves both
        self.node.create_subscription(TFMessage, tf_topic, self._on_tf,
                                      mocap_qos if shared else QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE))
        if not shared:
            self.node.create_subscription(_msg_class(self.mocap_cfg.msgtype), self.mocap_cfg.topic,
                                          self._on_mocap, mocap_qos)
        self.node.create_subscription(Imu, tp["imu"], self._on_imu,
                                      QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.event_pub = self.node.create_publisher(String, "/odin_char/event", 10)
        self.cmd_pub = self.node.create_publisher(TwistStamped, "/odin_char/cmd", 10)

        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.node)
        self._spin = threading.Thread(target=self.executor.spin, daemon=True)
        self._spin.start()

    # ------------------------------------------------------------------ callbacks
    def _on_mocap(self, msg):
        now = time.monotonic()
        for stamp, p, q in self.mocap_cfg.extract(msg):
            with self._lock:
                self.mocap_hist.append((now, stamp, p[0], p[1], quat_yaw(*q)))
                self.mocap_count += 1

    def _on_tf(self, msg):
        if self._mocap_on_tf:
            self._on_mocap(msg)
        for tr in msg.transforms:
            key = (tr.header.frame_id.lstrip("/"), tr.child_frame_id.lstrip("/"))
            self.tf_pair_counts[key] = self.tf_pair_counts.get(key, 0) + 1
            if key == self.map_pair:
                now = time.time()
                with self._lock:
                    self.last_map_tf_arrival = now
                    if self.first_map_tf_arrival is None:
                        self.first_map_tf_arrival = now

    def _on_imu(self, _msg):
        self.last_imu_arrival = time.time()
        self.imu_count += 1

    # ------------------------------------------------------------------ queries
    def mocap(self) -> tuple[np.ndarray, float] | None:
        """Latest (x, y, yaw) and its age in seconds (by arrival)."""
        with self._lock:
            if not self.mocap_hist:
                return None
            ta, _, x, y, yaw = self.mocap_hist[-1]
        return np.array([x, y, yaw]), time.monotonic() - ta

    def mocap_velocity(self, window: float = 0.1) -> np.ndarray | None:
        """World-frame (vx, vy, wz) from mocap, differenced on the mocap server's own stamps."""
        with self._lock:
            h = list(self.mocap_hist)
        if len(h) < 3:
            return None
        t1 = h[-1][1]
        old = [r for r in h if t1 - r[1] >= window]
        if not old:
            return None
        a, b = old[-1], h[-1]
        dt = b[1] - a[1]
        if dt <= 0:
            return None
        return np.array([(b[2] - a[2]) / dt, (b[3] - a[3]) / dt,
                         math.remainder(b[4] - a[4], 2 * math.pi) / dt])

    def reset_map_watch(self):
        with self._lock:
            self.first_map_tf_arrival = None
            self.last_map_tf_arrival = None

    def event(self, text: str):
        m = self._String()
        m.data = text
        self.event_pub.publish(m)
        self.node.get_logger().info(text)

    def publish_cmd(self, cmd):
        m = self._TwistStamped()
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.header.frame_id = "base_link"
        m.twist.linear.x, m.twist.linear.y, m.twist.angular.z = (float(c) for c in cmd)
        self.cmd_pub.publish(m)

    def shutdown(self):
        import rclpy

        self.executor.shutdown()
        self.node.destroy_node()
        rclpy.try_shutdown()
