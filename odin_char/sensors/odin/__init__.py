"""Odin1 (manifoldsdk/odin_ros_driver v0.14.4): tf odom -> imu and odom -> map, IMU on /odin1/imu,
raw cloud on /odin1/cloud_raw, maps saved via /odin1/save_map. Settings in defaults.yaml."""
from __future__ import annotations

import re

from ..base import TfSensor
from .driver import OdinDriver

# driver queues whose overflow corrupts what the analysis uses (fake latency and gaps)
_SERIOUS_QUEUES = {"ODOMETRY", "TF_ODOM", "IMU"}


class OdinSensor(TfSensor):
    name = "Odin"
    driver_class = OdinDriver

    def log_issue(self, line: str) -> tuple[str, bool] | None:
        m = re.search(r"Queue (\w+) is full", line)
        return (f"queue {m.group(1)} full", m.group(1) in _SERIOUS_QUEUES) if m else None


SENSOR = OdinSensor
