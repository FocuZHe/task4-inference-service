import asyncio
import base64
import threading

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport
from prometheus_client import CollectorRegistry

from serving.app import app as default_app, create_app
from serving.contracts import ErrorCode
from serving.limits import Limits
from serving.model_registry import RegistryError
from tests.fixtures import VIDEO_1080P


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeRegistry:
    def __init__(self, missing: bool = False) -> None:
        self.missing = missing

    def verify_active(self) -> None:
        if self.missing:
            raise RegistryError("model file is missing: detection")


class FakeDetection:
    def __init__(self, delay_s: float = 0.0, failures: int = 0) -> None:
        self.delay_s = delay_s
        self.failures = failures
        self.calls = 0
        self.started = threading.Semaphore(0)
        self.release = threading.Event()

    def predict(self, image, model_id, precision="fp32"):
        self.calls += 1
        if self.delay_s:
            self.started.release()
            self.release.wait(timeout=5)
            if self.delay_s > 0.5:
                import time
                time.sleep(self.delay_s)
        if self.calls <= self.failures:
            raise RuntimeError("injected runtime failure")
        return []


class FakeQuality:
    def score_features(self, features):
        return 0.5


def _png(width: int, height: int) -> str:
    ok, encoded = cv2.imencode(".png", np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _body(**overrides) -> dict:
    payload = {
        "requestId": "req-1",
        "modelVersion": "yolo11n-dut-new",
        "source": "eo",
        "receivedAt": "2026-11-18T00:00:00Z",
        "imageBase64": _png(32, 24),
    }
    payload.update(overrides)
    return payload


def _app(**kwargs):
    return create_app(
        registry=kwargs.pop("registry", FakeRegistry()),
        detection=kwargs.pop("detection", FakeDetection()),
        quality=kwargs.pop("quality", FakeQuality()),
        metrics_registry=CollectorRegistry(),
        **kwargs,
    )


def test_default_app_serves_quality_requests():
    with TestClient(default_app) as client:
        response = client.post("/infer/frame?task=quality", json=_body(modelVersion="quality-ridge-v1"))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["errorCode"] is None
    assert 0.0 <= body["result"]["qualityScore"] <= 1.0


def test_frame_returns_contract_fields():
    app = _app()
    with TestClient(app) as client:
        response = client.post("/infer/frame", json=_body())
    assert response.status_code == 200
    body = response.json()
    for key in ("requestId", "modelVersion", "source", "receivedAt", "result", "latencyMs", "errorCode"):
        assert key in body
    assert body["errorCode"] is None
    assert body["result"]["boxes"] == []


def test_batch_endpoint_uses_one_batched_predict():
    detection = FakeDetection()
    detection.batch_sizes = []

    def predict_batch_encoded(payloads, model_id, precision="fp32"):
        detection.batch_sizes.append(len(payloads))
        return [(16, 16, []) for _ in payloads]

    detection.predict_batch_encoded = predict_batch_encoded
    frames = [_png(16, 16) for _ in range(8)]
    app = _app(detection=detection)
    with TestClient(app) as client:
        response = client.post("/infer/batch", json=_body(imageBase64=None, framesBase64=frames))
    assert response.status_code == 200
    assert len(response.json()["result"]["frames"]) == 8
    assert detection.calls == 0
    assert detection.batch_sizes == [8]


def test_batch_returns_eight_results():
    frames = [_png(16, 16) for _ in range(8)]
    app = _app()
    with TestClient(app) as client:
        response = client.post("/infer/batch", json=_body(imageBase64=None, framesBase64=frames))
    assert response.status_code == 200
    assert len(response.json()["result"]["frames"]) == 8


def test_video_fixture_returns_at_least_one_frame():
    app = _app()
    with TestClient(app) as client:
        response = client.post("/infer/video", json=_body(imageBase64=None, fixtureId="ten-second-1080p"))
    assert response.status_code == 200
    assert len(response.json()["result"]["frames"]) >= 1
    assert VIDEO_1080P.is_file()


def test_oversized_image_is_invalid():
    app = _app()
    with TestClient(app) as client:
        response = client.post("/infer/frame", json=_body(imageBase64=_png(1921, 8)))
    assert response.status_code == 400
    assert response.json()["errorCode"] == ErrorCode.INVALID_INPUT


def test_fifth_queued_request_is_rejected():
    detection = FakeDetection(delay_s=0.01)
    app = _app(detection=detection, limits=Limits(queue_capacity=4, max_in_flight=1, timeout_s=5))

    async def run():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            first = asyncio.create_task(client.post("/infer/frame", json=_body(requestId="hold")))
            await asyncio.to_thread(detection.started.acquire)
            waiting = [
                asyncio.create_task(client.post("/infer/frame", json=_body(requestId=f"wait-{index}")))
                for index in range(4)
            ]
            for _ in range(50):
                if app.state.service.queued >= 4:
                    break
                await asyncio.sleep(0.02)
            rejected = await client.post("/infer/frame", json=_body(requestId="overflow"))
            detection.release.set()
            await first
            await asyncio.gather(*waiting)
            return rejected

    response = asyncio.run(run())
    assert response.status_code == 503
    assert response.json()["errorCode"] == ErrorCode.QUEUE_FULL


def test_timeout_returns_timeout_code():
    app = _app(detection=FakeDetection(delay_s=1.0), limits=Limits(timeout_s=0.1))
    with TestClient(app) as client:
        response = client.post("/infer/frame", json=_body())
    assert response.status_code == 408
    assert response.json()["errorCode"] == ErrorCode.TIMEOUT


def test_breaker_opens_and_half_opens_after_30_seconds():
    clock = Clock()
    detection = FakeDetection(failures=5)
    app = _app(detection=detection, clock=clock)
    with TestClient(app) as client:
        for index in range(5):
            response = client.post("/infer/frame", json=_body(requestId=f"fail-{index}"))
            assert response.status_code == 500
            assert response.json()["errorCode"] == ErrorCode.INTERNAL
        blocked = client.post("/infer/frame", json=_body(requestId="blocked"))
        assert blocked.status_code == 503
        assert blocked.json()["errorCode"] == ErrorCode.UNAVAILABLE
        clock.now += 30
        probe = client.post("/infer/frame", json=_body(requestId="probe"))
        assert probe.status_code == 200
        assert probe.json()["errorCode"] is None
        closed = client.post("/infer/frame", json=_body(requestId="closed"))
        assert closed.status_code == 200


def test_health_ready_and_metrics():
    app = _app()
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").status_code == 200
        client.post("/infer/frame", json=_body())
        metrics = client.get("/metrics").text
    for name in ("inference_latency_ms", "queue_wait_ms", "process_rss_bytes", "gpu_memory_bytes"):
        assert name in metrics
    assert "gpu_memory_bytes 0.0" in metrics or "gpu_memory_bytes 0" in metrics
    missing = _app(registry=FakeRegistry(missing=True))
    with TestClient(missing) as client:
        ready = client.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["errorCode"] == ErrorCode.MODEL_MISSING
