"""Sensors under test. One sub-package per sensor type; campaign.yaml picks one:

    sensor:
      type: odin          # odin_char/sensors/odin/
      driver:             # anything here overrides odin_char/sensors/odin/defaults.yaml
        base_config: ~/odin_ws/src/odin_ros_driver/config/control_command.yaml

Adding a sensor: see sensors/README.md.
"""
from __future__ import annotations

import copy
import importlib
from functools import lru_cache
from pathlib import Path

import yaml

from .base import Driver, Mode, Sensor, TfSensor

__all__ = ["Driver", "Mode", "Sensor", "TfSensor", "load_sensor", "resolve_config"]


@lru_cache(maxsize=None)
def _defaults(sensor_type: str) -> dict:
    p = Path(__file__).parent / sensor_type / "defaults.yaml"
    return (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


_LEGACY_TOPICS = ("imu", "odometry", "odometry_highfreq", "cloud")


def _from_legacy(cfg: dict) -> dict:
    """Campaigns recorded before the sensor split kept Odin's settings in top-level `frames`,
    `topics` and `driver`; turn them into the equivalent `sensor:` block."""
    fr, tp = cfg.get("frames") or {}, cfg.get("topics") or {}
    drv = dict(cfg.get("driver") or {})
    s: dict = {"type": "odin", "driver": drv}
    names = {"odin_odom": "odom", "odin_imu": "body", "odin_map": "map",
             "odom_map_is_ros_convention": "odom_map_is_ros_convention"}
    s["frames"] = {new: fr[old] for old, new in names.items() if old in fr}
    s["topics"] = {k: tp[k] for k in ("imu", "odometry", "cloud") if k in tp}
    if (cfg.get("recording") or {}).get("odometry_highfreq") and tp.get("odometry_highfreq"):
        s["topics"]["extra"] = [tp["odometry_highfreq"]]
    for k in ("warmup_min", "connect_timeout_s", "qos_overrides"):
        if k in drv:
            s[k] = drv.pop(k)
    return s


def resolve_config(cfg: dict) -> dict:
    """campaign.yaml with a complete `sensor:` block (the sensor's defaults.yaml under the
    campaign's own values). Idempotent; old-layout configs are converted."""
    cfg = copy.deepcopy(cfg)
    if "sensor" not in cfg:
        cfg["sensor"] = _from_legacy(cfg)
        cfg.pop("frames", None)
        cfg.pop("driver", None)
        for k in _LEGACY_TOPICS:
            (cfg.get("topics") or {}).pop(k, None)
        (cfg.get("recording") or {}).pop("odometry_highfreq", None)
    st = cfg["sensor"].get("type")
    if not st:
        raise ValueError("sensor.type is missing in the config (e.g. `type: odin`)")
    cfg["sensor"] = _merge(_defaults(st), cfg["sensor"])
    return cfg


def load_sensor(cfg: dict) -> Sensor:
    """The Sensor for a campaign config (raw or resolved)."""
    block = resolve_config(cfg)["sensor"]
    try:
        mod = importlib.import_module(f"{__name__}.{block['type']}")
    except ModuleNotFoundError as e:
        if e.name != f"{__name__}.{block['type']}":
            raise
        raise ValueError(f"unknown sensor.type {block['type']!r}: no package odin_char/sensors/{block['type']}/") from e
    return mod.SENSOR(block)
