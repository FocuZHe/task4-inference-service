"""Measure the Task 3 CPU and memory gate on a fixed performance fixture."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import inspect
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

import cv2
from threadpoolctl import threadpool_info

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.task3_quality_service import inference
from services.task3_quality_service.inference import load_models
from services.task3_quality_service.registry import RegisteredVideo
from scripts.extract_features import _feature_profile


SOURCE_FILES = (
    "services/task3_quality_service/inference.py",
    "services/task3_quality_service/model_registry.py",
    "scripts/extract_combined_features.py",
    "scripts/extract_features.py",
    "scripts/extract_anomaly_features.py",
    "scripts/diagnose_events.py",
    "scripts/train_quality.py",
    "scripts/task3_performance_benchmark.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _peak_working_set_bytes() -> int:
    """Return the process peak working set on Windows, with a portable fallback."""
    if os.name == "nt":
        size_t = ctypes.c_size_t

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint32),
                ("page_fault_count", ctypes.c_uint32),
                ("peak_working_set_size", size_t),
                ("working_set_size", size_t),
                ("quota_peak_paged_pool_usage", size_t),
                ("quota_paged_pool_usage", size_t),
                ("quota_peak_non_paged_pool_usage", size_t),
                ("quota_non_paged_pool_usage", size_t),
                ("pagefile_usage", size_t),
                ("peak_pagefile_usage", size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), ctypes.c_uint32]
        get_process_memory_info.restype = ctypes.c_int
        if get_process_memory_info(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return int(counters.peak_working_set_size)
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError) as exc:
        raise RuntimeError("peak working-set measurement requires Windows or psutil") from exc


def benchmark_runs(
    loader: Callable[[], Any],
    analyze: Callable[[Any, str], Any],
    peak_rss: Callable[[], int],
    *,
    warmup_runs: int = 1,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    if not isinstance(warmup_runs, int) or isinstance(warmup_runs, bool) or warmup_runs < 0:
        raise ValueError("warmup_runs must be a non-negative integer")
    model_started = clock()
    engine = loader()
    model_load_ms = round((clock() - model_started) * 1000)
    def measure(kind: str) -> dict[str, Any]:
        started = clock()
        result = analyze(engine, kind)
        wall_ms = round((clock() - started) * 1000)
        processing_ms = getattr(result, "processing_time_ms", None)
        if not isinstance(processing_ms, (int, float)) or isinstance(processing_ms, bool):
            processing_ms = wall_ms
        return {"kind": kind, "processingTimeMs": int(processing_ms), "wallTimeMs": wall_ms}

    runs = [measure("cold")]
    for index in range(warmup_runs):
        analyze(engine, f"warmup-{index + 1}")
    runs.append(measure("warmed"))
    return {
        "engine": engine,
        "modelLoadMs": model_load_ms,
        "warmupRunCount": warmup_runs,
        "runs": runs,
        "peakRssBytes": int(peak_rss()),
    }


def _probe_fixture(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("performance fixture could not be opened")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or not fps > 0 or frame_count <= 0:
        raise ValueError("performance fixture has incomplete media metadata")
    duration = frame_count / fps
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "durationSeconds": round(duration, 6),
        "frameCount": frame_count,
    }


def _source_record(manifest_path: Path, source_video_id: str) -> dict[str, Any]:
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("videoId", row.get("video_id")) == source_video_id:
            is_synthetic = row.get("isSynthetic", row.get("is_synthetic", False))
            if row.get("split") != "train" or is_synthetic is not False:
                raise ValueError("performance source must be an official non-synthetic train video")
            return {
                **row,
                "videoId": source_video_id,
                "videoSha256": row.get("videoSha256", row.get("sha256")),
                "isSynthetic": is_synthetic,
            }
    raise ValueError(f"source video not found in manifest: {source_video_id}")


def _runtime_config(engine: Any, input_fps: float) -> dict[str, Any]:
    # Analyze passes only quality_max_long_side explicitly. Read all remaining
    # defaults from the very function bound to inference, rather than copying
    # nominal values or inferring native resolution from candidate opt-in.
    parameters = inspect.signature(inference.extract_quality_and_event_features).parameters
    quality_sample_fps = float(parameters["quality_sample_fps"].default)
    event_sample_period = float(parameters["sample_period_s"].default)
    quality_every = max(1, round(input_fps / quality_sample_fps))
    event_every = max(1, round(input_fps * event_sample_period))
    quality_max_side = engine._quality_max_long_side()
    event_artifact = engine.event_artifact
    event_profile = str(event_artifact.get("featureProfile") or "event-features-v1")
    time_config = event_artifact.get("timePostprocess")
    if time_config is None:
        # This is the effective fallback in diagnose_events.predict_intervals.
        time_config = {
            "maxGapS": float(event_artifact.get("mergeMaxGapS", 0.5)),
            "confirmationWindows": 1,
            "recoveryS": float(event_artifact.get("mergeMaxGapS", 0.5)),
            "minimumEventDurationS": float(event_artifact.get("minimumEventDurationS", 0.5)),
        }
    return {
        "analysisMode": "offline-full-video",
        "configurationSource": "inference-bound extractor defaults and loaded engine artifacts",
        "fixedSemanticsSource": "scripts/extract_combined_features.py and scripts/diagnose_events.py; see sourceSha256",
        "quality": {
            "inputFeatureProfile": _feature_profile(quality_max_side),
            "maxLongSide": quality_max_side,
            "sampleFps": quality_sample_fps,
            "sampleEveryFrames": quality_every,
            "effectiveSampleFps": input_fps / quality_every,
            "windowSeconds": float(parameters["quality_window_s"].default),
            "stepSeconds": float(parameters["quality_step_s"].default),
            "frameDifference": {"reference": "adjacent-decoded-frames", "size": [160, 90]},
        },
        "events": {
            "featureProfile": event_profile,
            "samplePeriodSeconds": event_sample_period,
            "sampleEveryFrames": event_every,
            "effectiveSamplePeriodSeconds": event_every / input_fps,
            "maxLongSide": 640,
            "sharpnessFeatureSize": [64, 36],
            "frameDifference": {"reference": "adjacent-sampled-frames", "size": [160, 90]},
            # All supported non-baseline profiles use whole-video temporal
            # aggregates, including later samples; v13 also uses wide context.
            "usesFullVideoContext": event_profile != "event-features-v1",
            "usesFutureContext": event_profile != "event-features-v1",
            "causalOnline": False,
            "thresholds": dict(event_artifact["thresholds"]),
            "timePostprocess": dict(time_config),
        },
    }


def _estimator_job_settings(model: Any) -> dict[str, Any]:
    get_params = getattr(model, "get_params", None)
    parameters = get_params(deep=True) if callable(get_params) else {}
    return {
        name: value for name, value in parameters.items()
        if name == "n_jobs" or name.endswith("__n_jobs")
    }


def run_benchmark(
    fixture_path: Path,
    source_manifest: Path,
    source_video_id: str,
    quality_model_path: Path,
    event_model_path: Path,
    *,
    allow_candidate: bool,
    warmup_runs: int,
) -> dict[str, Any]:
    fixture_path = fixture_path.resolve(strict=True)
    source_manifest = source_manifest.resolve(strict=True)
    quality_model_path = quality_model_path.resolve(strict=True)
    event_model_path = event_model_path.resolve(strict=True)
    source = _source_record(source_manifest, source_video_id)
    source_hash = str(source.get("videoSha256") or "").lower()
    if len(source_hash) != 64 or _sha256(Path(source["path"])) != source_hash:
        raise ValueError("source manifest SHA-256 does not match the official source file")
    fixture_meta = _probe_fixture(fixture_path)
    if fixture_meta != {**fixture_meta, "width": 1920, "height": 1080, "fps": 25.0, "durationSeconds": 600.0, "frameCount": 15000}:
        raise ValueError("performance fixture must be exactly 1920x1080, 25 fps, 600 seconds and 15000 frames")
    fixture_hash = _sha256(fixture_path)

    def loader() -> Any:
        return load_models(
            quality_model_path, event_model_path,
            allow_candidate=allow_candidate, require_hashes=True,
        )

    def analyze(engine: Any, run_name: str) -> Any:
        registered = RegisteredVideo(
            "task3-performance-1080p25-600s",
            fixture_path,
            "visible",
            True,
            sha256=fixture_hash,
        )
        return engine.analyze(registered, f"performance-{run_name}")

    measured = benchmark_runs(loader, analyze, _peak_working_set_bytes, warmup_runs=warmup_runs)
    engine = measured.pop("engine")
    runtime_config = _runtime_config(engine, fixture_meta["fps"])
    root = Path(__file__).resolve().parents[1]
    return {
        "schemaVersion": "task3.performance-benchmark/1.0",
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "fixture": {
            "scope": "performance-only",
            "trainingAllowed": False,
            "path": str(fixture_path),
            **fixture_meta,
            "sha256": fixture_hash,
            "sourceDataset": "KoNViD-1k",
            "sourceSplit": "train",
            "sourceVideoId": source_video_id,
            "sourceVideoSha256": source_hash,
            "sourceManifestPath": str(source_manifest),
            "sourceManifestSha256": _sha256(source_manifest),
        },
        "hardware": {
            "cpu": platform.processor() or platform.uname().processor or "unknown",
            "logicalThreads": os.cpu_count() or 1,
            "physicalMemoryBytes": _physical_memory_bytes(),
        },
        "software": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "packages": {name: version(name) for name in ("numpy", "scikit-learn", "scipy", "joblib")},
            "opencvThreads": cv2.getNumThreads(),
            "threadPools": threadpool_info(),
            "estimatorJobs": {
                "quality": _estimator_job_settings(engine.quality_artifact["model"]),
                "events": _estimator_job_settings(engine.event_artifact["model"]),
            },
        },
        "model": {
            "profile": "candidate" if allow_candidate else "baseline",
            "version": engine.model_version,
            "qualityVersion": engine.quality_artifact["modelVersion"],
            "eventVersion": engine.event_artifact["modelVersion"],
            "qualityFeatureProfile": engine.quality_artifact.get("featureProfile") or "quality-features-v1",
            "qualityInputFeatureProfile": runtime_config["quality"]["inputFeatureProfile"],
            "qualityInputExtractionConfig": {"maxLongSide": runtime_config["quality"]["maxLongSide"]},
            "eventFeatureProfile": runtime_config["events"]["featureProfile"],
            "qualityPath": str(quality_model_path),
            "qualitySha256": _sha256(quality_model_path),
            "eventPath": str(event_model_path),
            "eventSha256": _sha256(event_model_path),
            "modelLoadMs": measured["modelLoadMs"],
        },
        "runtimeConfig": runtime_config,
        "sourceSha256": {relative: _sha256(root / relative) for relative in SOURCE_FILES},
        "runs": measured["runs"],
        "warmupRunCount": measured["warmupRunCount"],
        "peakRssBytes": measured["peakRssBytes"],
        "methodology": {
            "coldRun": "first full service analysis in a fresh benchmark process after model loading; OS file cache is not cleared",
            "warmedRun": "full service analysis after the cold pass and configured warmup passes",
            "memoryMethod": "Windows process PeakWorkingSetSize high-water counter",
            "inputScope": "synthetic performance fixture only; never training or evaluation data",
            "sampling": "sampleFps and samplePeriodSeconds are nominal requests; effective values follow frame-index rounding at the fixture fps",
        },
        "targets": {"maxProcessingTimeMs": 600_000, "maxPeakRssBytes": 2 * 1024**3},
    }


def _physical_memory_bytes() -> int:
    if os.name == "nt":
        installed_kb = ctypes.c_ulonglong()
        if ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(installed_kb)):
            return int(installed_kb.value) * 1024
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except (ImportError, OSError):
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-video-id", required=True)
    parser.add_argument("--quality-model", type=Path, required=True)
    parser.add_argument("--event-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=1)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; refusing to overwrite")
    try:
        report = run_benchmark(
            args.fixture,
            args.source_manifest,
            args.source_video_id,
            args.quality_model,
            args.event_model,
            allow_candidate=args.candidate,
            warmup_runs=args.warmup_runs,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
