from __future__ import annotations

import asyncio
import base64
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from serving.contracts import ErrorCode, InferenceRequest, InferenceResponse, error_response
from serving.image_decode import decode_bgr as _decode_bgr
from serving.limits import Limits
from serving.model_registry import ROOT, RegistryError

FIXTURES = {
    "ten-second-1080p": ROOT / "artifacts" / "fixtures" / "ten-second-1080p.mp4",
    "engineering-smoke-10s": ROOT / "任务三模型" / "data" / "raw" / "task3-smoke" / "engineering-smoke-10s.avi",
}


class InferenceService:
    def __init__(self, detection, quality, limits: Limits | None = None) -> None:
        self.detection = detection
        self.quality = quality
        self.limits = limits or Limits()
        self._queued = 0
        self._lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(self.limits.max_in_flight)
        self._cancel: dict[str, asyncio.Event] = {}
        self.last_queue_wait_s = 0.0

    @property
    def queued(self) -> int:
        return self._queued

    def cancel(self, request_id: str) -> None:
        event = self._cancel.get(request_id)
        if event is not None:
            event.set()

    async def serve(self, request: InferenceRequest, task: str) -> InferenceResponse:
        started = time.perf_counter()
        async with self._lock:
            if self._queued >= self.limits.queue_capacity:
                return _failure(request, ErrorCode.QUEUE_FULL, started)
            self._queued += 1
            cancel = asyncio.Event()
            self._cancel[request.requestId] = cancel
        wait_started = time.perf_counter()
        acquired = False
        try:
            await self._slots.acquire()
            acquired = True
            self.last_queue_wait_s = time.perf_counter() - wait_started
            async with self._lock:
                self._queued -= 1
            if cancel.is_set():
                return _failure(request, ErrorCode.CANCELLED, started)
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(self._execute, request, task, cancel),
                    timeout=self.limits.timeout_s,
                )
            except TimeoutError:
                cancel.set()
                return _failure(request, ErrorCode.TIMEOUT, started)
            if cancel.is_set():
                return _failure(request, ErrorCode.CANCELLED, started)
            return InferenceResponse(
                requestId=request.requestId,
                modelVersion=request.modelVersion,
                source=request.source,
                receivedAt=request.receivedAt,
                result=result,
                latencyMs=_elapsed_ms(started),
                errorCode=None,
            )
        except MemoryError:
            return _failure(request, ErrorCode.RESOURCE_EXHAUSTED, started)
        except FileNotFoundError:
            return _failure(request, ErrorCode.MODEL_MISSING, started)
        except (ValueError, RegistryError):
            return _failure(request, ErrorCode.INVALID_INPUT, started)
        except Exception:
            return _failure(request, ErrorCode.INTERNAL, started)
        finally:
            self._cancel.pop(request.requestId, None)
            if acquired:
                self._slots.release()

    def _execute(self, request: InferenceRequest, task: str, cancel: asyncio.Event) -> dict:
        encoded_batch = task != "quality" and request.framesBase64 is not None and hasattr(self.detection, "predict_batch_encoded")
        frames = [] if encoded_batch else _frames_for(request)
        if cancel.is_set():
            raise TimeoutError
        if task == "quality":
            return _quality_result(self.quality, request, frames)
        if request.imageBase64 is not None:
            return _frame_boxes(frames[0], self.detection.predict(frames[0], request.modelVersion))
        if encoded_batch:
            grouped = self.detection.predict_batch_encoded(request.framesBase64, request.modelVersion)
            return {"frames": [_sized_boxes(width, height, boxes) for width, height, boxes in grouped]}
        if hasattr(self.detection, "predict_batch"):
            grouped = self.detection.predict_batch(frames, request.modelVersion)
            return {"frames": [_frame_boxes(frame, boxes) for frame, boxes in zip(frames, grouped)]}
        return {"frames": [_frame_boxes(frame, self.detection.predict(frame, request.modelVersion)) for frame in frames]}


def _failure(request: InferenceRequest, code: ErrorCode, started: float) -> InferenceResponse:
    return error_response(
        requestId=request.requestId,
        modelVersion=request.modelVersion,
        receivedAt=request.receivedAt,
        errorCode=code,
        latencyMs=_elapsed_ms(started),
    )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _frames_for(request: InferenceRequest) -> list[np.ndarray]:
    if request.imageBase64 is not None:
        return [_decode_bgr(request.imageBase64)]
    if request.framesBase64 is not None:
        return [_decode_bgr(frame) for frame in request.framesBase64]
    if request.fixtureId is not None:
        path = FIXTURES.get(request.fixtureId)
        if path is None or not path.is_file():
            raise ValueError("unknown fixtureId")
        return _read_video(path)
    if request.videoBase64 is not None:
        raw = base64.b64decode(request.videoBase64)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "upload.mp4"
            path.write_bytes(raw)
            return _read_video(path)
    raise ValueError("request has no decodable input")


def _read_video(path: Path) -> list[np.ndarray]:
    with tempfile.TemporaryDirectory() as temporary:
        local = Path(temporary) / "clip.mp4"
        shutil.copyfile(path, local)
        capture = cv2.VideoCapture(str(local))
        frames: list[np.ndarray] = []
        try:
            if not capture.isOpened():
                raise ValueError("video could not be decoded")
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            capture.release()
    if not frames:
        raise ValueError("video contains no frames")
    return frames


def _frame_boxes(frame: np.ndarray, boxes) -> dict:
    return _sized_boxes(frame.shape[1], frame.shape[0], boxes)


def _sized_boxes(width: int, height: int, boxes) -> dict:
    return {
        "boxes": [
            {
                "bboxXyxy": [float(value) for value in box.bboxXyxy],
                "label": box.label,
                "confidence": float(box.confidence),
                "imageWidth": width,
                "imageHeight": height,
            }
            for box in boxes
        ]
    }


def _quality_result(quality, request: InferenceRequest, frames: list[np.ndarray]) -> dict:
    raw = float(np.asarray(quality.score_features(_quality_row(frames))).reshape(-1)[0])
    artifact = getattr(quality, "quality_artifact", None)
    if isinstance(artifact, dict) and "qualityScoreNormalization" in artifact:
        bounds = artifact["qualityScoreNormalization"]
        low, high = float(bounds["mosMin"]), float(bounds["mosMax"])
        score = float(np.clip((raw - low) / (high - low), 0.0, 1.0))
    else:
        score = float(np.clip(raw, 0.0, 1.0))
    return {
        "schemaVersion": "shared.video-quality/1.0",
        "jobId": request.requestId,
        "videoId": request.fixtureId or request.requestId,
        "provenance": {
            "source": "eo",
            "deviceId": "task4-serving",
            "observedAt": request.receivedAt,
            "receivedAt": request.receivedAt,
            "dataQuality": "ok",
            "timeQuality": "estimated",
            "modelVersion": request.modelVersion,
        },
        "channelType": "visible",
        "isSynthetic": request.fixtureId is not None,
        "limitations": [],
        "scoreStatus": "scored",
        "qualityScore": score,
        "anomalies": [],
        "processingTimeMs": 0,
    }


def _quality_row(frames: list[np.ndarray]) -> np.ndarray:
    laps, lumas, darks, brights, contrasts, diffs = [], [], [], [], [], []
    previous = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 90))
        lumas.append(float(small.mean()))
        contrasts.append(float(small.std()))
        darks.append(float(np.mean(small <= 16)))
        brights.append(float(np.mean(small >= 239)))
        laps.append(float(cv2.Laplacian(small, cv2.CV_64F).var()))
        diffs.append(0.0 if previous is None else float(np.mean(cv2.absdiff(small, previous))))
        previous = small
    return np.asarray([[np.mean(laps), np.mean(lumas), np.mean(darks), np.mean(brights), np.mean(contrasts), np.mean(diffs)]], dtype=np.float64)
