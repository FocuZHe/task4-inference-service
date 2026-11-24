"""Extract Task 3 quality and event features in one video decode pass."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scripts.extract_anomaly_features import (
    event_ground_truth,
    frame_sharpness_features,
    label_window,
)
from scripts.extract_features import FrameSample, _frame_metrics, _resize_for_quality, aggregate_samples


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resize_for_event(gray: np.ndarray) -> np.ndarray:
    height, width = gray.shape[:2]
    long_side = max(height, width)
    if long_side <= 640:
        return gray
    scale = 640 / long_side
    return cv2.resize(
        gray,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def extract_quality_and_event_features(
    record: dict[str, Any],
    sample_period_s: float = 0.5,
    quality_sample_fps: float = 2.0,
    quality_window_s: float = 10.0,
    quality_step_s: float = 5.0,
    quality_max_long_side: int = 640,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the existing quality and event feature rows from one decode.

    The independently maintained feature definitions are intentionally kept
    byte-for-byte compatible in their calculations: quality frame difference
    uses adjacent decoded frames, while event frame difference uses adjacent
    sampled frames.
    """
    if sample_period_s <= 0 or not math.isfinite(sample_period_s):
        raise ValueError("sample_period_s must be finite and positive")
    if not isinstance(quality_max_long_side, int) or isinstance(quality_max_long_side, bool) or quality_max_long_side < 0:
        raise ValueError("quality_max_long_side must be a non-negative integer")
    path = Path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if _sha256(path) != record.get("videoSha256"):
        raise ValueError(f"video SHA-256 mismatch: {record.get('videoId')}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"cannot decode video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError(f"video has missing timestamp or frame-count metadata: {path}")

    duration_s = frame_count / fps
    quality_sample_every = max(1, round(fps / quality_sample_fps))
    event_sample_every = max(1, round(fps * sample_period_s))
    quality_samples: list[FrameSample] = []
    event_rows: list[dict[str, Any]] = []
    previous_quality_gray: np.ndarray | None = None
    previous_event_gray: np.ndarray | None = None
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            small = cv2.resize(gray, (160, 90))
            quality_frame_diff = (
                0.0
                if previous_quality_gray is None
                else float(np.mean(cv2.absdiff(small, previous_quality_gray)))
            )
            previous_quality_gray = small

            if index % quality_sample_every == 0:
                quality_frame = _resize_for_quality(frame, quality_max_long_side)
                luma_mean, luma_std, dark_ratio, bright_ratio, contrast, lap_var = _frame_metrics(
                    quality_frame
                )
                quality_samples.append(
                    FrameSample(
                        timestamp_s=index / fps,
                        luma_mean=luma_mean,
                        luma_std=luma_std,
                        dark_ratio=dark_ratio,
                        bright_ratio=bright_ratio,
                        contrast=contrast,
                        lap_var=lap_var,
                        frame_diff=quality_frame_diff,
                    )
                )

            if index % event_sample_every == 0:
                event_small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
                sharpness_small = cv2.resize(event_small, (64, 36), interpolation=cv2.INTER_AREA)
                quality_gray = _resize_for_event(gray)
                gray_float = quality_gray.astype(np.float32)
                start_s = index / fps
                end_s = min(duration_s, start_s + event_sample_every / fps)
                event = record.get("anomaly")
                event_rows.append(
                    {
                        "recordId": record["recordId"],
                        "videoId": record["videoId"],
                        "sourceVideoId": record["sourceVideoId"],
                        "sourceVideoSha256": record["sourceVideoSha256"],
                        "split": record["split"],
                        "isSynthetic": bool(record["isSynthetic"]),
                        "groundTruthEvent": event_ground_truth(record),
                        "windowStartS": round(start_s, 6),
                        "windowEndS": round(end_s, 6),
                        "label": label_window(start_s, end_s, event),
                        "luma_mean": float(gray_float.mean()),
                        "luma_std": float(gray_float.std()),
                        "contrast": float(gray_float.std()),
                        "dark_ratio": float(np.mean(quality_gray <= 16)),
                        "bright_ratio": float(np.mean(quality_gray >= 239)),
                        "lap_var": float(cv2.Laplacian(quality_gray, cv2.CV_64F).var()),
                        "frame_diff": (
                            0.0
                            if previous_event_gray is None
                            else float(cv2.absdiff(event_small, previous_event_gray).mean())
                        ),
                        "sharpnessFeatureSize": [64, 36],
                        **frame_sharpness_features(sharpness_small),
                    }
                )
                previous_event_gray = event_small
            index += 1
    finally:
        capture.release()

    if not quality_samples:
        raise ValueError(f"video decoded no frames: {path}")
    if not event_rows:
        raise ValueError(f"video decoded no feature windows: {path}")
    quality_duration = min(duration_s, index / fps)
    quality_windows = aggregate_samples(
        quality_samples, quality_duration, quality_window_s, quality_step_s
    )
    return quality_windows, event_rows
