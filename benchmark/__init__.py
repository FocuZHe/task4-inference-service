from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEO_1080P = ROOT / "artifacts" / "fixtures" / "ten-second-1080p.mp4"


def first_1080p_frame(image: Path | None = None) -> tuple[bytes, int, int]:
    if image is not None:
        encoded = np.fromfile(image, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(image)
    else:
        frame = _first_video_frame(VIDEO_1080P)
    return _jpeg(frame)


def fixture_jpegs(count: int) -> list[tuple[bytes, int, int]]:
    with tempfile.TemporaryDirectory() as temporary:
        local = Path(temporary) / "clip.mp4"
        shutil.copyfile(VIDEO_1080P, local)
        capture = cv2.VideoCapture(str(local))
        try:
            return [_jpeg(frame) for frame in _read_frames(capture, count)]
        finally:
            capture.release()


def _jpeg(frame: np.ndarray) -> tuple[bytes, int, int]:
    ok, encoded_jpeg = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("could not encode the 1080p frame")
    height, width = frame.shape[:2]
    return encoded_jpeg.tobytes(), width, height


def _read_frames(capture: cv2.VideoCapture, count: int) -> Iterator[np.ndarray]:
    for _ in range(count):
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"could not read {count} frames from {VIDEO_1080P}")
        yield frame


def _first_video_frame(path: Path):
    with tempfile.TemporaryDirectory() as temporary:
        local = Path(temporary) / "clip.mp4"
        shutil.copyfile(path, local)
        capture = cv2.VideoCapture(str(local))
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"could not read a frame from {path}")
    return frame
