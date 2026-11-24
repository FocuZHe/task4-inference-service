"""Write the fixed 1080p latency fixture. The pixels come from seed 42."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "artifacts" / "fixtures" / "ten-second-1080p.mp4"
REPORT = ROOT / "artifacts" / "reports" / "fixture-hashes.json"


def main() -> None:
    generator = np.random.default_rng(42)
    base = generator.integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(DESTINATION),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25.0,
        (1920, 1080),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the mp4 writer")
    for index in range(250):
        frame = base.copy()
        x = (index * 48) % 1760
        y = (index * 24) % 990
        frame[y : y + 90, x : x + 160] = (32, 180, 240)
        writer.write(frame)
    writer.release()

    capture = cv2.VideoCapture(str(DESTINATION))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not read the fixture")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    digest = hashlib.sha256(DESTINATION.read_bytes()).hexdigest()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {
                "ten-second-1080p.mp4": {
                    "sha256": digest,
                    "width": width,
                    "height": height,
                    "frames": frames,
                    "fps": fps,
                    "seed": 42,
                    "purpose": "latency-and-interface-only",
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"path": str(DESTINATION), "sha256": digest, "width": width, "height": height, "frames": frames, "fps": fps}))


if __name__ == "__main__":
    main()
