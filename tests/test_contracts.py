import base64
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from serving.contracts import (
    DetectionBox,
    ErrorCode,
    InferenceRequest,
    InferenceResponse,
    QualityResult,
    error_response,
    http_status,
)
from tests.fixtures import DETECTION_SAMPLES, QUALITY_SMOKE_VIDEO, VIDEO_1080P, rng


def test_detection_samples_exist():
    assert len(DETECTION_SAMPLES) == 6
    assert all(path.is_file() for path in DETECTION_SAMPLES)


def test_quality_smoke_video_exists():
    assert QUALITY_SMOKE_VIDEO.is_file()


def test_1080p_fixture_exists():
    assert VIDEO_1080P.is_file()
    assert VIDEO_1080P.stat().st_size > 0


def test_rng_is_seed_42():
    expected = np.random.default_rng(42).random(4)
    assert np.array_equal(rng().random(4), expected)


def test_1080p_fixture_shape_and_hash():
    import hashlib
    import json

    import cv2

    capture = cv2.VideoCapture(str(VIDEO_1080P))
    assert capture.isOpened()
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    assert (width, height, frames) == (1920, 1080, 250)
    assert abs(fps - 25) < 0.1

    digest = hashlib.sha256(VIDEO_1080P.read_bytes()).hexdigest()
    report = json.loads((Path(__file__).resolve().parents[1] / "artifacts" / "reports" / "fixture-hashes.json").read_text(encoding="utf-8"))
    assert report["ten-second-1080p.mp4"]["sha256"] == digest
    assert report["ten-second-1080p.mp4"]["seed"] == 42


def _png(width: int, height: int) -> str:
    ok, encoded = cv2.imencode(".png", np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _request(**overrides: object) -> dict:
    body = {
        "requestId": "req-1",
        "modelVersion": "yolo11n-dut-new",
        "source": "eo",
        "receivedAt": "2026-11-17T00:00:00Z",
        "imageBase64": _png(64, 48),
    }
    body.update(overrides)
    return body


def test_response_contains_required_fields():
    body = InferenceResponse(
        requestId="req-1",
        modelVersion="yolo11n-dut-new",
        source="eo",
        receivedAt=datetime(2026, 11, 17, tzinfo=timezone.utc),
        result={"boxes": []},
        latencyMs=1.25,
        errorCode=None,
    )
    dumped = body.model_dump()
    for key in ("requestId", "modelVersion", "source", "receivedAt", "result", "latencyMs", "errorCode"):
        assert key in dumped


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        InferenceRequest.model_validate({
            "requestId": "req-1",
            "modelVersion": "yolo11n-dut-new",
            "source": "eo",
            "receivedAt": "2026-11-17T00:00:00Z",
            "imageBase64": "YQ==",
            "extra": 1,
        })


def test_received_at_requires_timezone():
    with pytest.raises(ValidationError):
        InferenceRequest.model_validate(_request(receivedAt="2026-11-17T00:00:00"))


def test_exactly_one_input_is_required():
    missing = _request()
    missing.pop("imageBase64")
    with pytest.raises(ValidationError):
        InferenceRequest.model_validate(missing)
    with pytest.raises(ValidationError):
        InferenceRequest.model_validate(_request(framesBase64=[_png(8, 8)]))


def test_batch_length_is_one_to_eight():
    accepted = _request(framesBase64=[_png(8, 8)] * 8)
    accepted.pop("imageBase64")
    InferenceRequest.model_validate(accepted)
    rejected = _request(framesBase64=[_png(8, 8)] * 9)
    rejected.pop("imageBase64")
    with pytest.raises(ValidationError):
        InferenceRequest.model_validate(rejected)


def test_image_long_edge_limit():
    InferenceRequest.model_validate(_request(imageBase64=_png(1920, 1080)))
    with pytest.raises(ValidationError):
        InferenceRequest.model_validate(_request(imageBase64=_png(1921, 8)))


def test_detection_box_uses_shared_labels():
    DetectionBox.model_validate({
        "bboxXyxy": [1, 2, 10, 12],
        "label": "drone",
        "confidence": 0.5,
        "imageWidth": 64,
        "imageHeight": 48,
    })
    with pytest.raises(ValidationError):
        DetectionBox.model_validate({
            "bboxXyxy": [1, 2, 10, 12],
            "label": "uav",
            "confidence": 0.5,
            "imageWidth": 64,
            "imageHeight": 48,
        })


def test_quality_result_rejects_confirmed_freeze():
    payload = {
        "schemaVersion": "shared.video-quality/1.0",
        "jobId": "job-1",
        "videoId": "video-1",
        "provenance": {
            "source": "eo",
            "deviceId": "camera-01",
            "observedAt": "2026-11-17T00:00:00Z",
            "receivedAt": "2026-11-17T00:00:00.200Z",
            "dataQuality": "ok",
            "timeQuality": "estimated",
            "modelVersion": "ridge-v1+task3-event-rf-v1",
        },
        "channelType": "visible",
        "isSynthetic": True,
        "limitations": [],
        "scoreStatus": "scored",
        "qualityScore": 0.5,
        "anomalies": [{
            "type": "freeze_suspected",
            "startTimeS": 0.0,
            "endTimeS": 1.0,
            "status": "confirmed",
        }],
        "processingTimeMs": 10,
    }
    with pytest.raises(ValidationError):
        QualityResult.model_validate(payload)
    payload["anomalies"][0]["status"] = "suspected"
    QualityResult.model_validate(payload)


def test_error_response_keeps_contract_fields():
    body = error_response(
        requestId="req-1",
        modelVersion="yolo11n-dut-new",
        receivedAt=datetime(2026, 11, 17, tzinfo=timezone.utc),
        errorCode=ErrorCode.QUEUE_FULL,
        latencyMs=0.4,
    )
    dumped = body.model_dump()
    for key in ("requestId", "modelVersion", "source", "receivedAt", "result", "latencyMs", "errorCode"):
        assert key in dumped
    assert dumped["errorCode"] == "QUEUE_FULL"
    assert http_status(ErrorCode.INVALID_INPUT) == 400
    assert http_status(ErrorCode.MODEL_MISSING) == 503
    assert http_status(ErrorCode.TIMEOUT) == 408
    assert http_status(ErrorCode.CANCELLED) == 429
    assert http_status(ErrorCode.QUEUE_FULL) == 503
    assert http_status(ErrorCode.UNAVAILABLE) == 503
    assert http_status(ErrorCode.RESOURCE_EXHAUSTED) == 500
    assert http_status(ErrorCode.INTERNAL) == 500
