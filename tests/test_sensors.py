"""Sensor selection: config resolution (incl. pre-split campaign configs) and the Odin sensor."""
from pathlib import Path

import pytest
import yaml

from odin_char.sensors import load_sensor, resolve_config
from odin_char.sensors.odin import OdinSensor

CFG = Path(__file__).parents[1] / "config" / "campaign.yaml"

# the frames/topics/driver blocks of a campaign.yaml written before the sensor split
LEGACY = {
    "frames": {"odin_odom": "odom", "odin_imu": "imu", "odin_map": "map", "odom_map_is_ros_convention": False},
    "topics": {"tf": "/tf", "tf_static": "/tf_static", "odometry": "/odin1/odometry",
               "odometry_highfreq": "/odin1/odometry_highfreq", "imu": "/odin1/imu", "cloud": "/odin1/cloud_raw",
               "rosout": "/rosout", "markers": None},
    "recording": {"min_free_gb": 2.0, "odometry_highfreq": True},
    "driver": {"base_config": "~/odin_ws/src/odin_ros_driver/config/control_command.yaml", "warmup_min": 20,
               "connect_timeout_s": 60},
}


def test_campaign_config_resolves_to_odin():
    cfg = resolve_config(yaml.safe_load(CFG.read_text()))
    assert resolve_config(cfg) == cfg  # idempotent
    s = load_sensor(cfg)
    assert isinstance(s, OdinSensor) and s.name == "Odin"
    assert s.fix_tf == ("odom", "map") and s.odometry_tf == ("odom", "imu")
    assert s.heartbeat == ("/odin1/imu", "sensor_msgs/msg/Imu")
    assert "/odin1/cloud_raw" in s.record_topics(with_cloud=True)
    assert "/odin1/cloud_raw" not in s.record_topics(with_cloud=False)
    assert s.owns(topic="/odin1/odometry") and s.owns(frame="lidar") and not s.owns(topic="/vrpn/x/pose", frame="world")


def test_legacy_config_is_converted():
    cfg = resolve_config(LEGACY)
    assert "frames" not in cfg and "driver" not in cfg and "imu" not in cfg["topics"]
    s = cfg["sensor"]
    assert s["type"] == "odin" and s["warmup_min"] == 20
    assert s["frames"]["body"] == "imu"
    assert s["topics"]["extra"] == ["/odin1/odometry_highfreq"]  # recording.odometry_highfreq: true
    assert s["driver"]["process_name"] == "host_sdk_sample"  # from the Odin defaults


def test_unknown_sensor_type():
    with pytest.raises(ValueError, match="no package"):
        load_sensor({"sensor": {"type": "nonexistent"}})
    with pytest.raises(ValueError, match="sensor.type"):
        resolve_config({"sensor": {}})


def test_odin_log_issue():
    s = load_sensor(yaml.safe_load(CFG.read_text()))
    assert s.log_issue("[WARN] Queue TF_ODOM is full, dropping") == ("queue TF_ODOM full", True)
    assert s.log_issue("[WARN] Queue RGB is full, dropping") == ("queue RGB full", False)
    assert s.log_issue("relocalization success!") is None
