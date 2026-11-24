"""Extract windowed video-quality and frame-difference features."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

try:
    from scripts.quality_experiment import refuse_existing_output
except ModuleNotFoundError:
    from quality_experiment import refuse_existing_output


@dataclass(frozen=True)
class FrameSample:
    timestamp_s: float
    luma_mean: float
    luma_std: float
    dark_ratio: float
    bright_ratio: float
    contrast: float
    lap_var: float
    frame_diff: float


FEATURE_NAMES = (
    "lap_var",
    "luma_mean",
    "luma_std",
    "dark_ratio",
    "bright_ratio",
    "contrast",
    "frame_diff",
)
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
BASE_FEATURE_PROFILE = "quality-features-v1"
MANIFEST_PROVENANCE_FIELDS = (
    "dataset",
    "source",
    "split",
    "mos",
    "recordId",
    "filename",
    "videoSha256",
    "packageSha256",
    "officialUrl",
    "license",
    "externalTestOnly",
    "trainingAllowed",
    "selectionAllowed",
)


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _aggregate(values: list[float], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_p05": _percentile(values, 5),
        f"{prefix}_p50": _percentile(values, 50),
        f"{prefix}_p95": _percentile(values, 95),
    }


def _output_prefix(name: str) -> str:
    # Keep the public feature names aligned with the training contract:
    # luma_mean, dark_ratio_mean, lap_var_mean, and so on.
    return "luma" if name == "luma_mean" else name


def aggregate_samples(
    samples: list[FrameSample],
    duration_s: float,
    window_s: float = 10.0,
    step_s: float = 5.0,
) -> list[dict[str, Any]]:
    """Aggregate samples into overlapping windows using media timestamps."""
    if not samples:
        raise ValueError("cannot aggregate an empty sample list")
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if window_s <= 0 or step_s <= 0:
        raise ValueError("window_s and step_s must be positive")

    starts = [0.0] if duration_s <= window_s else list(np.arange(0.0, duration_s, step_s))
    windows: list[dict[str, Any]] = []
    for start in starts:
        end = min(duration_s, start + window_s)
        selected = [sample for sample in samples if start <= sample.timestamp_s < end]
        if not selected:
            continue
        row: dict[str, Any] = {
            "window_start_s": round(float(start), 6),
            "window_end_s": round(float(end), 6),
            "sample_count": len(selected),
        }
        for name in FEATURE_NAMES:
            row.update(_aggregate([float(getattr(sample, name)) for sample in selected], _output_prefix(name)))
        windows.append(row)
    return windows


def _resize_for_quality(frame: np.ndarray, max_long_side: int = 640) -> np.ndarray:
    if max_long_side < 0:
        raise ValueError("max_long_side must be non-negative")
    if max_long_side == 0:
        return frame
    height, width = frame.shape[:2]
    long_side = max(height, width)
    if long_side <= max_long_side:
        return frame
    scale = max_long_side / long_side
    return cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))


def _feature_profile(max_long_side: int) -> str:
    """Return a stable profile name so resolution variants cannot be mixed."""
    if max_long_side == 640:
        return BASE_FEATURE_PROFILE
    if max_long_side == 0:
        return f"{BASE_FEATURE_PROFILE}-native-resolution"
    return f"{BASE_FEATURE_PROFILE}-max-long-side-{max_long_side}"


def _frame_metrics(frame: np.ndarray) -> tuple[float, float, float, float, float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    gray_float = gray.astype(np.float32)
    return (
        float(gray_float.mean()),
        float(gray_float.std()),
        float(np.mean(gray <= 16)),
        float(np.mean(gray >= 239)),
        float(gray_float.std()),
        float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    )


def extract_video_features(
    path: Path,
    sample_fps: float = 2.0,
    window_s: float = 10.0,
    step_s: float = 5.0,
    max_long_side: int = 640,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Decode one video and return window rows plus decode metadata."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError(f"video has missing timestamp metadata: {path}")
    duration_s = frame_count / fps
    sample_every = max(1, round(fps / sample_fps))
    previous_gray: np.ndarray | None = None
    samples: list[FrameSample] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
        frame_diff = 0.0 if previous_gray is None else float(np.mean(cv2.absdiff(small, previous_gray)))
        previous_gray = small
        if index % sample_every == 0:
            quality_frame = _resize_for_quality(frame, max_long_side)
            luma_mean, luma_std, dark_ratio, bright_ratio, contrast, lap_var = _frame_metrics(quality_frame)
            samples.append(
                FrameSample(
                    timestamp_s=index / fps,
                    luma_mean=luma_mean,
                    luma_std=luma_std,
                    dark_ratio=dark_ratio,
                    bright_ratio=bright_ratio,
                    contrast=contrast,
                    lap_var=lap_var,
                    frame_diff=frame_diff,
                )
            )
        index += 1
    capture.release()
    if not samples:
        raise ValueError(f"video decoded no frames: {path}")
    metadata = {
        "fps": fps,
        "frame_count": index,
        "duration_s": index / fps,
        "sample_fps": fps / sample_every,
        "max_long_side": max_long_side,
        "feature_profile": _feature_profile(max_long_side),
        "extraction_config": {
            "sample_fps": sample_fps,
            "window_s": window_s,
            "step_s": step_s,
            "max_long_side": max_long_side,
        },
    }
    return aggregate_samples(samples, min(duration_s, index / fps), window_s, step_s), metadata


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def propagate_manifest_provenance(record: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Carry dataset authorization metadata into every extracted window row."""
    propagated = dict(row)
    if "video_id" not in propagated:
        video_id = record.get("video_id", record.get("videoId"))
        if video_id is not None and str(video_id).strip():
            propagated["video_id"] = str(video_id).strip()
    for field in MANIFEST_PROVENANCE_FIELDS:
        if field in record:
            propagated[field] = record[field]
    return propagated


def _write_rows(rows: list[dict[str, Any]], output: Path) -> None:
    output = refuse_existing_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".jsonl":
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return
    if output.suffix.lower() != ".csv":
        raise ValueError("output must use .jsonl or .csv")
    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL manifest from prepare_manifest.py")
    parser.add_argument("--output", type=Path, required=True, help=".jsonl or .csv feature output")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--window-s", type=float, default=10.0)
    parser.add_argument("--step-s", type=float, default=5.0)
    parser.add_argument(
        "--max-long-side",
        type=int,
        default=640,
        help="maximum quality-feature frame dimension; 0 keeps decoded resolution",
    )
    args = parser.parse_args(argv)
    if args.sample_fps <= 0:
        parser.error("--sample-fps must be positive")
    if args.max_long_side < 0:
        parser.error("--max-long-side must be non-negative")

    rows: list[dict[str, Any]] = []
    for record in _read_jsonl(args.manifest):
        path = Path(record["path"])
        video_id = record.get("video_id", record.get("videoId"))
        if video_id is None or not str(video_id).strip():
            raise ValueError("manifest record requires video_id or videoId")
        windows, decode = extract_video_features(
            path,
            args.sample_fps,
            args.window_s,
            args.step_s,
            args.max_long_side,
        )
        for window in windows:
            row = propagate_manifest_provenance(record, {
                "video_id": str(video_id).strip(),
                "path": record["path"],
                **decode,
                **window,
            })
            rows.append(row)
    _write_rows(rows, args.output)
    print(f"wrote {len(rows)} feature windows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
