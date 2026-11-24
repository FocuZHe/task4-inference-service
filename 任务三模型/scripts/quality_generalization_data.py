from __future__ import annotations

import json
import math
import re
from pathlib import Path

from train_quality_candidates import FRAME_STATISTICS_COLUMNS, _videos

NATIVE_CONFIG = {"max_long_side": 0, "sample_fps": 2.0, "window_s": 10.0, "step_s": 5.0}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_train(manifest_path, features_path, *, expected_count=840):
    official = {}
    source_splits = {}
    for row in read_jsonl(manifest_path):
        vid, split, digest = row["video_id"], row["split"], row["sha256"]
        if vid in official or split not in {"train", "val", "test"}:
            raise ValueError("duplicate official id or invalid split")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("invalid source hash")
        if digest in source_splits and source_splits[digest] != split:
            raise ValueError("source hash crosses split")
        source_splits[digest] = split
        official[vid] = row
    train_ids = {vid for vid, row in official.items() if row["split"] == "train"}
    if len(train_ids) != expected_count:
        raise ValueError("unexpected official train count")
    windows, seen = [], set()
    for row in read_jsonl(features_path):
        vid = row["video_id"]
        source = official.get(vid)
        if source is None or row["split"] != source["split"]:
            raise ValueError("feature/manifest split mismatch")
        if row["split"] != "train":
            continue
        if source.get("source") != "konvid-1k" or row.get("source") != "konvid-1k":
            raise ValueError("unapproved source")
        if source.get("parent_video_id") is not None or source.get("isSynthetic", False):
            raise ValueError("only official original videos are allowed")
        if row.get("feature_profile") != "quality-features-v1-native-resolution":
            raise ValueError("native profile required")
        if row.get("extraction_config") != NATIVE_CONFIG:
            raise ValueError("native extraction configuration mismatch")
        mos = float(row["mos"])
        if not math.isfinite(mos) or not math.isclose(mos, float(source["mos"]), abs_tol=1e-12, rel_tol=0):
            raise ValueError("official MOS mismatch")
        start, end = float(row["window_start_s"]), float(row["window_end_s"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError("invalid window boundary")
        key = (vid, start, end)
        if key in seen:
            raise ValueError("duplicate window")
        seen.add(key)
        windows.append(row)
    if {row["video_id"] for row in windows} != train_ids:
        raise ValueError("train feature coverage mismatch")
    videos = _videos(windows, FRAME_STATISTICS_COLUMNS, frame_statistics=True)
    for row in videos:
        row["source_group"] = official[row["video_id"]]["sha256"]
    return sorted(videos, key=lambda row: row["video_id"])

