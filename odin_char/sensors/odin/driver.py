"""odin_ros_driver: per-run control_command.yaml, map save over its service, device serial."""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import yaml

from ..base import Driver, expand


class OdinDriver(Driver):
    """Starts/stops odin_ros_driver with a per-run copy of control_command.yaml."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.watch.update({
            "map_save_target": "Map save triggered, will transfer to",
            "relocalized": self.cfg["relocalization_success_text"],
            "map_saved": self.cfg["map_saved_text"],
            "map_failed": "Map transfer failed",
        })

    # odin_ros_driver (0.14.4) ignores the `config_file` parameter: it always loads
    # <package source>/config/control_command.yaml (host_sdk_sample.cpp, main()). So each run's config
    # is written INTO that installed file. The user's original is backed up once and restored by
    # restore_config() (called when the campaign exits); runs are always built from the original.
    @property
    def installed_config(self) -> Path:
        return expand(self.cfg["base_config"])

    @property
    def original_backup(self) -> Path:
        p = self.installed_config
        return p.with_name(p.name + ".odin_char_original")

    def prepare(self, run_dir: Path, tag: str, mode: int, reloc_map: Path | None = None) -> Path:
        installed, backup = self.installed_config, self.original_backup
        if not backup.exists():
            shutil.copy2(installed, backup)  # the user's own config, restored on exit
        base = yaml.safe_load(backup.read_text())
        keys = base.setdefault("register_keys", {})
        keys.update(self.cfg.get("overrides", {}))
        keys["custom_map_mode"] = int(mode)  # Odin's modes are the campaign's: 0 odometry, 1 SLAM, 2 relocalization
        if reloc_map is not None:
            keys["relocalization_map_abs_path"] = str(reloc_map)
        # Let the driver save maps to its default <ws>/src/odin_ros_driver/map/<start time>/ folder;
        # save_map() reads the real path from the driver log and the campaign copies the file.
        keys["mapping_result_dest_dir"] = ""
        keys["mapping_result_file_name"] = ""
        text = yaml.safe_dump(base, sort_keys=False)
        out = run_dir / f"control_command_{tag}.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)  # archived with the run
        installed.write_text(text)  # what the driver actually reads
        self.config_path = out
        return out

    def restore_config(self):
        """Put the user's original control_command.yaml back (no-op if nothing was changed)."""
        backup = self.original_backup
        if backup.exists():
            shutil.copy2(backup, self.installed_config)
            backup.unlink()

    def save_map(self, timeout: float | None = None) -> Path | None:
        """Ask the device to generate and transfer its SLAM map. Returns the saved file (taken
        from the driver's own "Map transfer completed: <path>" line), or None on failure/timeout."""
        svc = self.cfg["save_map_service"]
        r = subprocess.run(["ros2", "service", "call", svc, "odin_ros_driver/srv/SaveMap", "{value: 1}"],
                           stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
        if r.returncode != 0 or "success=True" not in r.stdout.replace(" ", ""):
            raise RuntimeError(f"save_map service call failed: {r.stdout} {r.stderr}")
        t_end = time.time() + (timeout or float(self.cfg.get("save_map_timeout_s", 600)))
        while time.time() < t_end:
            if "map_saved" in self.events:
                line = self.event_lines.get("map_saved", "")
                path = line.split(self.cfg["map_saved_text"], 1)[-1].strip(" :")
                return Path(path) if path else None
            if "map_failed" in self.events:
                return None
            time.sleep(0.5)
        return None

    def device_serial(self) -> str:
        p = expand(self.cfg.get("calib_yaml"))
        try:
            return p.read_text().splitlines()[0].strip() if p else "unknown"
        except (OSError, IndexError):
            return "unknown"
