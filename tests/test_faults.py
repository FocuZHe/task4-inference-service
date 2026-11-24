import asyncio
import base64
import threading

import cv2
import httpx
import numpy as np
from fastapi.testclient import TestClient
from httpx import ASGITransport
from prometheus_client import CollectorRegistry

from serving.app import create_app
from serving.contracts import ErrorCode
from serving.limits import Limits
from serving.model_registry import ROOT, ModelRegistry, RegistryError


class MissingRegistry:
    def verify_active(self) -> None:
        raise RegistryError("model file is missing: detection")


class ReadyRegistry:
    def verify_active(self) -> None:
        return None


class FaultDetection:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.started = threading.Semaphore(0)
        self.release = threading.Event()

    def predict(self, image, model_id, precision="fp32"):
        if self.mode == "missing":
            raise FileNotFoundError("model file is missing")
        if self.mode == "memory":
            raise MemoryError("injected memory error")
        if self.mode == "timeout":
            import time
            time.sleep(1)
            return []
        if self.mode == "block":
            self.started.release()
            self.release.wait(timeout=5)
            return []
        return []


def _png(width: int, height: int) -> str:
    ok, encoded = cv2.imencode(".png", np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _body(**overrides) -> dict:
    payload = {
        "requestId": "fault-1",
        "modelVersion": "yolo11n-dut-new",
        "source": "eo",
        "receivedAt": "2026-11-18T00:00:00Z",
        "imageBase64": _png(32, 24),
    }
    payload.update(overrides)
    return payload


def _client_app(detection, limits=None, registry=None):
    return create_app(
        registry or ReadyRegistry(),
        (detection, None),
        limits or Limits(),
        metrics_registry=CollectorRegistry(),
    )


def test_missing_model_returns_model_missing():
    app = _client_app(FaultDetection("missing"), registry=MissingRegistry())
    with TestClient(app) as client:
        response = client.post("/infer/frame", json=_body())
        ready = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["errorCode"] == ErrorCode.MODEL_MISSING
    assert ready.status_code == 503
    assert ready.json()["errorCode"] == ErrorCode.MODEL_MISSING


def test_corrupt_image_returns_invalid_input():
    app = _client_app(FaultDetection("ok"))
    with TestClient(app) as client:
        response = client.post("/infer/frame", json=_body(imageBase64="YQ=="))
    assert response.status_code == 400
    assert response.json()["errorCode"] == ErrorCode.INVALID_INPUT


def test_timeout_returns_timeout():
    app = _client_app(FaultDetection("timeout"), limits=Limits(timeout_s=0.1))
    with TestClient(app) as client:
        response = client.post("/infer/frame", json=_body())
    assert response.status_code == 408
    assert response.json()["errorCode"] == ErrorCode.TIMEOUT


def test_memory_error_returns_resource_exhausted():
    app = _client_app(FaultDetection("memory"))
    with TestClient(app) as client:
        response = client.post("/infer/frame", json=_body())
    assert response.status_code == 500
    assert response.json()["errorCode"] == ErrorCode.RESOURCE_EXHAUSTED


def test_full_queue_returns_queue_full():
    detection = FaultDetection("block")
    app = _client_app(detection, limits=Limits(queue_capacity=4, max_in_flight=1, timeout_s=5))

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


def test_restart_keeps_health_and_ready_when_hashes_match():
    registry = ModelRegistry.load(ROOT / "artifacts" / "registry" / "models.json")
    first = create_app(registry, (FaultDetection("ok"), None), Limits(), metrics_registry=CollectorRegistry())
    with TestClient(first) as client:
        assert client.get("/health").status_code == 200
    restarted = create_app(registry, (FaultDetection("ok"), None), Limits(), metrics_registry=CollectorRegistry())
    with TestClient(restarted) as client:
        assert client.get("/health").json() == {"status": "ok"}
        ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ok"
