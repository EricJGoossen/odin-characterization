"""Every supported mocap source type is read from a bag, and only the configured body is used."""
import numpy as np
import pytest

from odin_char.bagio import read_bag
from odin_char.mocap_config import MocapConfig

from .synth import TS, T, _time, write_bag


def _header(t, frame="world"):
    return T["std_msgs/msg/Header"](stamp=_time(t), frame_id=frame)


def _pose(x):
    return T["geometry_msgs/msg/Pose"](position=T["geometry_msgs/msg/Point"](x=x, y=2.0, z=0.1),
                                      orientation=T["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0))


def _tf(t, parent, child, x):
    return T["geometry_msgs/msg/TransformStamped"](
        header=_header(t, parent), child_frame_id=child,
        transform=T["geometry_msgs/msg/Transform"](
            translation=T["geometry_msgs/msg/Vector3"](x=x, y=2.0, z=0.1),
            rotation=T["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0)))


def _cfg(mocap):
    return {"mocap": mocap, "topics": {"tf": "/tf", "tf_static": "/tf_static", "odometry": "/odin1/odometry",
                                       "imu": "/odin1/imu", "cloud": None, "rosout": "/rosout", "markers": None}}


CASES = {
    "tf": ({"source": "tf", "topic": "/tf", "parent_frame": "world", "child_frame": "sentry"}, "/tf"),
    "tf_other_topic": ({"source": "tf", "topic": "/mocap/tf", "parent_frame": "world", "child_frame": "sentry"}, "/mocap/tf"),
    "pose": ({"source": "pose", "topic": "/vrpn_mocap/sentry/pose"}, "/vrpn_mocap/sentry/pose"),
    "transform": ({"source": "transform", "topic": "/vicon/sentry/sentry"}, "/vicon/sentry/sentry"),
    "odometry": ({"source": "odometry", "topic": "/mocap/sentry/odom"}, "/mocap/sentry/odom"),
}


@pytest.mark.parametrize("case", list(CASES))
def test_source_types(tmp_path, case):
    mocap, topic = CASES[case]
    streams = []
    for k in range(50):
        t = 100.0 + k / 100
        if mocap["source"] == "tf":
            msg = T["tf2_msgs/msg/TFMessage"](transforms=[_tf(t, "world", "other_body", -9.0), _tf(t, "world", "sentry", float(k))])
            streams.append((t + 0.001, 0, topic, msg, "tf2_msgs/msg/TFMessage"))
        elif mocap["source"] == "pose":
            streams.append((t + 0.001, 0, topic, T["geometry_msgs/msg/PoseStamped"](header=_header(t), pose=_pose(float(k))),
                            "geometry_msgs/msg/PoseStamped"))
        elif mocap["source"] == "transform":
            streams.append((t + 0.001, 0, topic, _tf(t, "world", "sentry", float(k)), "geometry_msgs/msg/TransformStamped"))
        else:
            odo = T["nav_msgs/msg/Odometry"](
                header=_header(t), child_frame_id="sentry",
                pose=T["geometry_msgs/msg/PoseWithCovariance"](pose=_pose(float(k)), covariance=np.zeros(36)),
                twist=T["geometry_msgs/msg/TwistWithCovariance"](
                    twist=T["geometry_msgs/msg/Twist"](linear=T["geometry_msgs/msg/Vector3"](x=0.0, y=0.0, z=0.0),
                                                       angular=T["geometry_msgs/msg/Vector3"](x=0.0, y=0.0, z=0.0)),
                    covariance=np.zeros(36)))
            streams.append((t + 0.001, 0, topic, odo, "nav_msgs/msg/Odometry"))
    bag = tmp_path / "b"
    write_bag(bag, streams)
    b = read_bag(bag, _cfg(mocap))
    assert len(b.mocap.t_stamp) == 50
    assert np.allclose(np.array(b.mocap.p)[:, 0], np.arange(50))  # never the other body


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        MocapConfig.from_cfg({"mocap": {"source": "tf", "topic": "/tf"}})
    with pytest.raises(ValueError):
        MocapConfig.from_cfg({"mocap": {"source": "posestamped", "topic": "/x"}})
