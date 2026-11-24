from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DETECTION_SAMPLES = sorted((ROOT / "任务一模型" / "demo" / "samples").glob("*.jpg"))
QUALITY_SMOKE_VIDEO = ROOT / "任务三模型" / "data" / "raw" / "task3-smoke" / "engineering-smoke-10s.avi"
VIDEO_1080P = ROOT / "artifacts" / "fixtures" / "ten-second-1080p.mp4"


def rng() -> np.random.Generator:
    return np.random.default_rng(42)
