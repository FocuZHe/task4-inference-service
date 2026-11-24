"""Train-source admission and exclusive caches for the fixed quality study."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np

from extract_features import extract_video_features
from train_quality_candidates import FRAME_STATISTICS_COLUMNS, _videos

FEATURE_COLUMNS = tuple(FRAME_STATISTICS_COLUMNS)
FIXED_CONFIGS = {
    f"C{index}": {"max_long_side": long_side, "sample_fps": fps,
                   "window_s": window, "step_s": step}
    for index, (long_side, fps, window, step) in enumerate((
        (0, 2., 10., 5.), (640, 2., 10., 5.), (0, 1., 10., 5.), (640, 1., 10., 5.),
        (0, 2., 5., 2.5), (640, 2., 5., 2.5), (0, 1., 5., 2.5), (640, 1., 5., 2.5),
    ))
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, value: object) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def write_jsonl_row(handle: Any, value: object) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{number}: JSON object required")
                yield value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite numeric")
    return number


def load_official_train(manifest: Path, video_root: Path, *, expected_count: int = 840) -> list[dict[str, Any]]:
    """Inspect split structure globally, but open source bytes/MOS only for train."""
    video_root = Path(video_root).resolve(strict=True)
    ids: set[str] = set()
    source_splits: dict[str, str] = {}
    admitted = []
    for record in read_jsonl(manifest):
        video_id, split, digest = record.get("video_id"), record.get("split"), record.get("sha256")
        if not isinstance(video_id, str) or not video_id.strip() or video_id in ids:
            raise ValueError("official manifest requires unique nonempty video IDs")
        if not isinstance(split, str) or split not in {"train", "val", "test"}:
            raise ValueError("invalid official split")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid source SHA-256")
        if digest in source_splits and source_splits[digest] != split:
            raise ValueError("source hash crosses split")
        ids.add(video_id)
        source_splits[digest] = split
        if split != "train":
            continue
        if (record.get("source") != "konvid-1k" or record.get("parent_video_id") is not None
                or record.get("isSynthetic", False) is not False):
            raise ValueError("only official train original videos are admitted")
        mos = _finite(record.get("mos"), "finite MOS")
        path_value = record.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError("train source path is required")
        path = Path(path_value)
        path = (video_root / path).resolve() if not path.is_absolute() else path.resolve()
        if path == video_root or not path.is_relative_to(video_root):
            raise ValueError("train source path escapes video root")
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"source SHA-256 mismatch: {video_id}")
        admitted.append({**record, "mos": mos, "path": str(path), "source_group": digest})
    if len(admitted) != expected_count:
        raise ValueError(f"expected {expected_count} official train originals, found {len(admitted)}")
    by_source: dict[str, float] = {}
    for row in admitted:
        if row["source_group"] in by_source and by_source[row["source_group"]] != row["mos"]:
            raise ValueError("same source SHA has conflicting official MOS")
        by_source[row["source_group"]] = row["mos"]
    return sorted(admitted, key=lambda row: row["video_id"])


def verify_sources(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if sha256(Path(row["path"])) != row["source_group"]:
            raise ValueError(f"source SHA-256 changed: {row['video_id']}")


def extraction_profile(config: dict[str, Any]) -> str:
    return "quality-features-v1-native-resolution" if config["max_long_side"] == 0 else "quality-features-v1"


def validate_extraction(windows: list[dict[str, Any]], metadata: dict[str, Any], config: dict[str, Any]) -> None:
    if metadata.get("extraction_config") != config or metadata.get("max_long_side") != config["max_long_side"]:
        raise ValueError("extraction configuration mismatch")
    if metadata.get("feature_profile") != extraction_profile(config):
        raise ValueError("extraction profile mismatch")
    fps = _finite(metadata.get("fps"), "decode metadata fps")
    count = _finite(metadata.get("frame_count"), "decode metadata frame_count")
    duration = _finite(metadata.get("duration_s"), "decode metadata duration")
    effective = _finite(metadata.get("sample_fps"), "decode metadata sample fps")
    if fps <= 0 or count < 1 or int(count) != count or duration <= 0 or effective <= 0:
        raise ValueError("invalid decode metadata")
    if (not math.isclose(duration, count / fps, rel_tol=1e-9, abs_tol=1e-9)
            or not math.isclose(effective, fps / max(1, round(fps / config["sample_fps"])), rel_tol=1e-9, abs_tol=1e-9)):
        raise ValueError("inconsistent decode metadata")
    if not windows:
        raise ValueError("extraction produced no windows")
    seen = set()
    for window in windows:
        start = _finite(window.get("window_start_s"), "window boundary")
        end = _finite(window.get("window_end_s"), "window boundary")
        if not 0 <= start < end <= duration + 1e-6 or end - start > config["window_s"] + 1e-6:
            raise ValueError("invalid window boundary")
        if (start, end) in seen or not math.isclose(start / config["step_s"], round(start / config["step_s"]), abs_tol=1e-6):
            raise ValueError("duplicate or off-grid window boundary")
        seen.add((start, end))
        samples = _finite(window.get("sample_count"), "window sample_count")
        if samples < 1 or int(samples) != samples:
            raise ValueError("invalid window sample_count")
        for column in FEATURE_COLUMNS:
            _finite(window.get(column), f"finite feature {column}")
    timestamps = np.arange(0, int(count), max(1, round(fps / config["sample_fps"]))) / fps
    starts = [0.] if duration <= config["window_s"] else np.arange(0., duration, config["step_s"])
    expected = []
    for start in starts:
        end = min(duration, start + config["window_s"])
        samples = int(np.sum((timestamps >= start) & (timestamps < end)))
        if samples:
            expected.append((round(float(start), 6), round(float(end), 6), samples))
    actual = [(float(row["window_start_s"]), float(row["window_end_s"]), int(row["sample_count"])) for row in windows]
    if actual != expected:
        raise ValueError("window coverage/count differs from fixed extractor sampling semantics")


def extract_configurations(
    rows: list[dict[str, Any]], output: Path, *,
    extractor: Callable[..., Any] = extract_video_features,
    progress: Callable[[str], None] = print,
    extraction_stats: dict[str, Any] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Decode sequentially into brand-new per-config window/metadata/aggregate files."""
    if not rows or any(row.get("split") != "train" or row.get("source") != "konvid-1k"
                       or row.get("parent_video_id") is not None or row.get("isSynthetic", False) is not False for row in rows):
        raise ValueError("only official train original videos may be decoded")
    stats = extraction_stats if extraction_stats is not None else {}
    stats.update(configId=None, videoId=None, completedVideoDecodes=0, attemptedVideoDecodes=0)
    features_dir = Path(output) / "features"
    features_dir.mkdir(exist_ok=False)
    configurations: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    for config_id, config in FIXED_CONFIGS.items():
        directory = features_dir / config_id
        directory.mkdir(exist_ok=False)
        progress(f"extract {config_id}: 0/{len(rows)} videos")
        aggregate_rows = []
        rates, window_counts = [], []
        with (directory / "windows.jsonl").open("x", encoding="utf-8", newline="\n") as window_file, (directory / "metadata.jsonl").open("x", encoding="utf-8", newline="\n") as metadata_file:
            for number, record in enumerate(rows, 1):
                stats.update(configId=config_id, videoId=record["video_id"])
                path = Path(record["path"])
                if sha256(path) != record["source_group"]:
                    raise ValueError(f"source SHA-256 mismatch before decode: {record['video_id']}")
                stats["attemptedVideoDecodes"] += 1
                windows, metadata = extractor(path, **config)
                if sha256(path) != record["source_group"]:
                    raise ValueError(f"source SHA-256 changed during decode: {record['video_id']}")
                validate_extraction(windows, metadata, config)
                stats["completedVideoDecodes"] += 1
                bound_windows = [{**metadata, **window, "video_id": record["video_id"],
                                  "split": "train", "source": "konvid-1k", "mos": record["mos"],
                                  "sourceVideoSha256": record["source_group"], "path": str(path)} for window in windows]
                aggregate = _videos(bound_windows, FEATURE_COLUMNS, frame_statistics=True)[0]
                aggregate.update(source_group=record["source_group"], source="konvid-1k")
                aggregate_rows.append(aggregate)
                for window in bound_windows:
                    write_jsonl_row(window_file, window)
                write_jsonl_row(metadata_file, {"videoId": record["video_id"], "sourceVideoSha256": record["source_group"],
                                               "windowCount": len(windows), **metadata})
                rates.append(metadata["sample_fps"])
                window_counts.append(len(windows))
                if number % 25 == 0 or number == len(rows):
                    progress(f"extract {config_id}: {number}/{len(rows)} videos")
        with (directory / "videos.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for row in aggregate_rows:
                write_jsonl_row(handle, row)
        summaries[config_id] = {
            "extractionConfig": dict(config), "inputFeatureProfile": extraction_profile(config),
            "modelFeatureProfile": "quality-frame-statistics-v2", "featureColumns": list(FEATURE_COLUMNS),
            "successVideos": len(aggregate_rows), "failedVideos": 0,
            "effectiveSampleFps": {"min": float(min(rates)), "max": float(max(rates)), "mean": float(np.mean(rates)),
                                   "p05": float(np.quantile(rates, .05)), "p50": float(np.quantile(rates, .5)), "p95": float(np.quantile(rates, .95))},
            "windowCount": sum(window_counts), "windowsPerVideoMin": min(window_counts), "windowsPerVideoMax": max(window_counts),
            "cacheSha256": {path.name: sha256(path) for path in sorted(directory.iterdir())},
        }
        write_new(directory / "summary.json", summaries[config_id])
        configurations[config_id] = aggregate_rows
    return configurations, summaries
