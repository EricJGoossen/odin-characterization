"""LD19 2D lidar: quick characterization without mocap.

    scripts/ld19.sh check   [--dir campaigns/ld19]   driver up? scan rate and points (nothing recorded)
    scripts/ld19.sh static  [--dir ...] [--seconds 60]  lidar still, hands off: noise, dropouts, ghosts
    scripts/ld19.sh map     [--dir ...] [--no-viewer]   push the robot around; live map in rviz2; Enter ends
    scripts/ld19.sh analyze [--dir ...]              -> <dir>/results/report.md (no ROS needed)

Suggested session: `check`, then `static` at two or three spots (near a wall, in the open, facing
far walls), then one or two `map` runs, some of them spinning the robot in place, then `analyze`.
--config my.yaml overrides odin_char/sensors/ld19/defaults.yaml (e.g. driver_params.port_name).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .. import _merge

HERE = Path(__file__).parent


def load_config(path: Path | None) -> dict:
    cfg = yaml.safe_load((HERE / "defaults.yaml").read_text())
    return _merge(cfg, yaml.safe_load(path.read_text()) or {}) if path else cfg


def main():
    ap = argparse.ArgumentParser(prog="python -m odin_char.sensors.ld19", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["check", "static", "map", "analyze"])
    ap.add_argument("--dir", type=Path, default=Path("campaigns/ld19"))
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--seconds", type=float, default=None, help="static: recording length (default static_s)")
    ap.add_argument("--no-viewer", action="store_true", help="map: don't start rviz2 here")
    args = ap.parse_args()

    if args.step == "analyze":
        from .analyze import analyze

        saved = args.dir / "ld19.yaml"  # the config the data was recorded with
        cfg = load_config(args.config) if args.config or not saved.exists() else yaml.safe_load(saved.read_text())
        analyze(args.dir, cfg)
        print(f"report: {args.dir / 'results' / 'report.md'}")
        return

    from .collect import Collector

    cfg = load_config(args.config)
    if args.seconds:
        cfg["static_s"] = args.seconds
    c = Collector(cfg, args.dir, viewer=not args.no_viewer)
    try:
        getattr(c, f"step_{args.step}")()
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        c.close()
    if args.step != "check":
        print(f"analyse with: scripts/ld19.sh analyze --dir {args.dir}")


if __name__ == "__main__":
    main()
