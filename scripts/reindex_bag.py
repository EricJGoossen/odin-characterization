"""Rebuild metadata.yaml for rosbag2 sqlite bags whose recorder was killed (power cut, SIGKILL).

ros2 bag writes metadata.yaml only on a clean shutdown; the .db3 file itself is usually intact.
This is the `ros2 bag reindex` equivalent for machines without ROS.

    .venv/bin/python scripts/reindex_bag.py campaigns/odin1/bags/012_hold0 [more bags ...]
"""
import sqlite3
import sys
from pathlib import Path

import yaml


def needs_reindex(bag: Path) -> bool:
    meta = bag / "metadata.yaml"
    if not meta.exists():
        return True
    try:
        return not (yaml.safe_load(meta.read_text()) or {}).get("rosbag2_bagfile_information")
    except yaml.YAMLError:
        return True


def reindex(bag: Path) -> dict:
    files = sorted(bag.glob("*.db3"))
    if not files:
        raise SystemExit(f"{bag}: no .db3 files")
    topics: dict[int, dict] = {}
    counts: dict[str, int] = {}
    t_min, t_max, total = None, None, 0
    per_file = []
    for f in files:
        con = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
        cols = [r[1] for r in con.execute("PRAGMA table_info(topics)")]
        q = "SELECT id, name, type, serialization_format" + (", offered_qos_profiles" if "offered_qos_profiles" in cols else "")
        for row in con.execute(q + " FROM topics"):
            topics[row[0]] = {"name": row[1], "type": row[2], "serialization_format": row[3],
                              "offered_qos_profiles": row[4] if len(row) > 4 else ""}
        lo, hi, n = con.execute("SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM messages").fetchone()
        for tid, c in con.execute("SELECT topic_id, COUNT(*) FROM messages GROUP BY topic_id"):
            name = topics[tid]["name"]
            counts[name] = counts.get(name, 0) + c
        con.close()
        if n:
            t_min = lo if t_min is None else min(t_min, lo)
            t_max = hi if t_max is None else max(t_max, hi)
            total += n
        per_file.append({"path": f.name, "starting_time": {"nanoseconds_since_epoch": int(lo or 0)},
                         "duration": {"nanoseconds": int((hi or 0) - (lo or 0))}, "message_count": int(n)})
    info = {
        "version": 5,
        "storage_identifier": "sqlite3",
        "duration": {"nanoseconds": int((t_max or 0) - (t_min or 0))},
        "starting_time": {"nanoseconds_since_epoch": int(t_min or 0)},
        "message_count": int(total),
        "topics_with_message_count": [
            {"topic_metadata": t, "message_count": int(counts.get(t["name"], 0))} for t in topics.values()],
        "compression_format": "",
        "compression_mode": "",
        "relative_file_paths": [f.name for f in files],
        "files": per_file,
    }
    meta = bag / "metadata.yaml"
    if meta.exists():
        meta.rename(bag / "metadata.yaml.broken")
    meta.write_text(yaml.safe_dump({"rosbag2_bagfile_information": info}, sort_keys=False))
    return {"messages": total, "duration_s": ((t_max or 0) - (t_min or 0)) * 1e-9, "topics": counts}


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        b = Path(arg)
        if not needs_reindex(b):
            print(f"{b.name}: metadata OK")
            continue
        try:
            s = reindex(b)
        except sqlite3.DatabaseError as e:
            print(f"{b.name}: cannot reindex ({e}); the .db3 needs `sqlite3 <db3> .recover` first")
            continue
        print(f"{b.name}: rebuilt metadata, {s['messages']} messages over {s['duration_s']:.1f} s")
        for k, v in sorted(s["topics"].items()):
            print(f"    {k}: {v}")
