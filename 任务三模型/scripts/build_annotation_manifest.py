"""Build a provenance-first inventory for real-video human annotation.

This script never downloads videos and never creates labels. A low-motion flag
is only a deterministic sampling aid for the later human review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
DEFAULT_SAMPLE_PERIOD_S = 0.5
DEFAULT_LOW_MOTION_THRESHOLD = 2.0


def validate_inventory_record(record: dict[str, Any]) -> None:
    required = ("videoId", "path", "sourceUrl", "license")
    missing = [field for field in required if not str(record.get(field) or "").strip()]
    if missing:
        raise ValueError("inventory record missing: " + ", ".join(missing))
    video_id = str(record["videoId"]).strip()
    if any(character in video_id for character in "\\/\r\n\t"):
        raise ValueError("videoId must not contain path separators or control characters")
    url = str(record["sourceUrl"]).strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("sourceUrl must be an http(s) URL")
    path = Path(str(record["path"]).strip())
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"unsupported video extension: {path.suffix}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_low_motion(frame_differences: Iterable[float], *, threshold: float) -> bool:
    values = [float(value) for value in frame_differences]
    if not values or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("frame differences must be non-empty and threshold must be positive")
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("frame differences must be finite and non-negative")
    return float(np.mean(values)) <= threshold


def inspect_video(path: Path, *, sample_period_s: float, low_motion_threshold: float) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"cannot decode video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError(f"video has unusable timing metadata: {path}")
    sample_every = max(1, round(fps * sample_period_s))
    differences: list[float] = []
    previous: np.ndarray | None = None
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % sample_every == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
                if previous is not None:
                    differences.append(float(cv2.absdiff(small, previous).mean()))
                previous = small
            index += 1
    finally:
        capture.release()
    duration = frame_count / fps
    return {
        "durationS": float(duration),
        "fps": fps,
        "frameCount": frame_count,
        "samplePeriodS": sample_period_s,
        "meanFrameDiff": float(np.mean(differences)) if differences else 0.0,
        "p10FrameDiff": float(np.quantile(differences, 0.10)) if differences else 0.0,
        "lowMotionCandidate": classify_low_motion(differences or [0.0], threshold=low_motion_threshold),
        "lowMotionThreshold": low_motion_threshold,
        "motionSamples": len(differences),
    }


def read_inventory(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError("inventory must be .csv or .jsonl")
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("inventory must contain a non-empty list of objects")
    return rows


def build_manifest(
    records: Iterable[dict[str, Any]],
    *,
    data_root: Path,
    sample_period_s: float = DEFAULT_SAMPLE_PERIOD_S,
    low_motion_threshold: float = DEFAULT_LOW_MOTION_THRESHOLD,
) -> list[dict[str, Any]]:
    if sample_period_s <= 0 or not math.isfinite(sample_period_s):
        raise ValueError("sample_period_s must be finite and positive")
    if low_motion_threshold <= 0 or not math.isfinite(low_motion_threshold):
        raise ValueError("low_motion_threshold must be finite and positive")
    output = []
    seen: set[str] = set()
    data_root = data_root.resolve(strict=True)
    for record in records:
        validate_inventory_record(record)
        video_id = str(record["videoId"]).strip()
        if video_id in seen:
            raise ValueError(f"duplicate videoId: {video_id}")
        seen.add(video_id)
        candidate = Path(str(record["path"]).strip())
        path = (data_root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if data_root not in path.parents:
            raise ValueError(f"video path escapes data_root: {video_id}")
        info = inspect_video(path, sample_period_s=sample_period_s, low_motion_threshold=low_motion_threshold)
        output.append({
            "videoId": video_id,
            "path": str(path),
            "sourceUrl": str(record["sourceUrl"]).strip(),
            "license": str(record["license"]).strip(),
            "videoSha256": sha256_file(path),
            "clipStartS": 0.0,
            "clipEndS": min(10.0, info["durationS"]),
            **info,
            "annotationStatus": "not_started",
        })
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-period-s", type=float, default=DEFAULT_SAMPLE_PERIOD_S)
    parser.add_argument("--low-motion-threshold", type=float, default=DEFAULT_LOW_MOTION_THRESHOLD)
    args = parser.parse_args(argv)
    rows = build_manifest(
        read_inventory(args.inventory),
        data_root=args.data_root,
        sample_period_s=args.sample_period_s,
        low_motion_threshold=args.low_motion_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(f"wrote {len(rows)} real-video annotation inventory records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
