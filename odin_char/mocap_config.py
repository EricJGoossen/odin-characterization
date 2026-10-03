"""Where the mocap ground truth comes from (configurable: names differ between mocap setups).

campaign.yaml:

    mocap:
      source: tf            # tf | pose | transform | odometry
      topic: /tf            # tf topic, or the PoseStamped / TransformStamped / Odometry topic
      parent_frame: mocap   # tf only: the mocap world frame
      child_frame: base_link  # tf only: the tracked rigid body
      clock: server         # server: stamps from the mocap server's clock; host: stamped on arrival
      qos: best_effort      # best_effort also receives from reliable publishers

`python -m odin_char.cli find-mocap` lists candidate topics / tf pairs on the live system.
"""
from __future__ import annotations

from dataclasses import dataclass

MSGTYPES = {
    "tf": "tf2_msgs/msg/TFMessage",
    "pose": "geometry_msgs/msg/PoseStamped",
    "transform": "geometry_msgs/msg/TransformStamped",
    "odometry": "nav_msgs/msg/Odometry",
}


@dataclass(frozen=True)
class MocapConfig:
    source: str = "tf"
    topic: str = "/tf"
    parent_frame: str | None = None
    child_frame: str | None = None
    clock: str = "server"
    qos: str = "best_effort"

    @staticmethod
    def from_cfg(cfg: dict) -> "MocapConfig":
        m = dict(cfg.get("mocap") or {})
        mc = MocapConfig(**{k: v for k, v in m.items() if k in MocapConfig.__dataclass_fields__})
        if mc.source not in MSGTYPES:
            raise ValueError(f"mocap.source must be one of {list(MSGTYPES)}, got {mc.source!r}")
        if mc.source == "tf" and not (mc.parent_frame and mc.child_frame):
            raise ValueError("mocap.source tf needs mocap.parent_frame and mocap.child_frame")
        return mc

    @property
    def msgtype(self) -> str:
        return MSGTYPES[self.source]

    def extract(self, msg) -> list[tuple[float, list, list]]:
        """(stamp seconds, [x, y, z], [qx, qy, qz, qw]) samples of the tracked body in `msg`.
        Works for both rosbags-deserialised messages and live rclpy messages."""
        def stamp(h):
            return h.stamp.sec + h.stamp.nanosec * 1e-9

        def xyz(v):
            return [v.x, v.y, v.z]

        def xyzw(q):
            return [q.x, q.y, q.z, q.w]

        if self.source == "tf":
            out = []
            for tr in msg.transforms:
                if (tr.header.frame_id.lstrip("/"), tr.child_frame_id.lstrip("/")) == (
                    self.parent_frame.lstrip("/"), self.child_frame.lstrip("/")):
                    out.append((stamp(tr.header), xyz(tr.transform.translation), xyzw(tr.transform.rotation)))
            return out
        if self.source == "pose":
            return [(stamp(msg.header), xyz(msg.pose.position), xyzw(msg.pose.orientation))]
        if self.source == "transform":
            return [(stamp(msg.header), xyz(msg.transform.translation), xyzw(msg.transform.rotation))]
        return [(stamp(msg.header), xyz(msg.pose.pose.position), xyzw(msg.pose.pose.orientation))]
