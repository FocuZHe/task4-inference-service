from __future__ import annotations

import logging
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None

if orjson is not None:
    async def _read_json(self):
        if not hasattr(self, "_json"):
            self._json = orjson.loads(await self.body())
        return self._json

    Request.json = _read_json  # type: ignore[method-assign]
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from serving.contracts import InferenceRequest, InferenceResponse, http_status
from serving.detection_runtime import CPU_PROVIDER, CUDA_PROVIDER, DetectionRuntime
from serving.health import CircuitBreaker
from serving.inference import InferenceService
from serving.limits import Limits
from serving.model_registry import ROOT, ModelRegistry, RegistryError


def create_app(
    registry: ModelRegistry | None = None,
    runtimes: tuple | None = None,
    limits: Limits | None = None,
    *,
    detection=None,
    quality=None,
    clock=None,
    metrics_registry: CollectorRegistry | None = None,
) -> FastAPI:
    limits = limits or Limits()
    if limits.timeout_s == 2.0 and __import__("os").environ.get("TASK4_TIMEOUT_S"):
        limits = Limits(timeout_s=float(__import__("os").environ["TASK4_TIMEOUT_S"]))
    if runtimes is not None:
        detection = runtimes[0] if detection is None else detection
        if len(runtimes) > 1 and quality is None:
            quality = runtimes[1]
    registry = registry if registry is not None else ModelRegistry.load(ROOT / "artifacts" / "registry" / "models.json")
    use_default_runtimes = detection is None
    if use_default_runtimes:
        detection = DetectionRuntime()
        if quality is None:
            from serving.quality_runtime import QualityRuntime

            quality = QualityRuntime()
    provider = getattr(detection, "provider", CPU_PROVIDER)
    service = InferenceService(detection, quality, limits)
    breaker = CircuitBreaker(clock)
    collectors = metrics_registry or CollectorRegistry()
    latency = Histogram("inference_latency_ms", "End-to-end inference latency in milliseconds", registry=collectors)
    queue_wait = Histogram("queue_wait_ms", "Queue wait before inference in milliseconds", registry=collectors)
    errors = Counter("inference_errors_total", "Inference responses with an error code", ["errorCode"], registry=collectors)
    rss = Gauge("process_rss_bytes", "Process resident memory in bytes", registry=collectors)
    gpu = Gauge(
        "gpu_memory_bytes",
        "Total memory used on visible NVIDIA devices in bytes; zero for CPU and NaN if unavailable",
        registry=collectors,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if use_default_runtimes and provider == CUDA_PROVIDER:
            model = registry.active("detection")
            detection.warmup(model.modelId)
            logging.getLogger("uvicorn.error").info("Detection model %s warmed up with %s", model.modelId, provider)
        yield

    app = FastAPI(title="Task 4 inference", lifespan=lifespan)
    app.state.service = service
    app.state.breaker = breaker
    app.state.registry = registry

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        payload = _error_payload(exc, "INVALID_INPUT")
        return JSONResponse(payload, status_code=400)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        try:
            registry.verify_active()
        except RegistryError as exc:
            code = "MODEL_MISSING" if "missing" in str(exc) else "UNAVAILABLE"
            return JSONResponse({"status": "unavailable", "errorCode": code}, status_code=503)
        if detection is None or not hasattr(registry, "verify_active"):
            return JSONResponse({"status": "unavailable", "errorCode": "UNAVAILABLE"}, status_code=503)
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> Response:
        import psutil

        rss.set(psutil.Process().memory_info().rss)
        gpu.set(_gpu_memory_bytes() if provider == CUDA_PROVIDER else 0)
        return Response(generate_latest(collectors), media_type="text/plain; version=0.0.4")

    @app.post("/infer/cancel/{request_id}")
    def cancel(request_id: str) -> dict:
        service.cancel(request_id)
        return {"requestId": request_id, "cancelled": True}

    @app.post("/infer/frame")
    async def frame(request: InferenceRequest, task: str = "detection"):
        if request.imageBase64 is None:
            return _rejected(request, "INVALID_INPUT")
        return await _infer(request, task)

    @app.post("/infer/batch")
    async def batch(request: InferenceRequest, task: str = "detection"):
        if request.framesBase64 is None:
            return _rejected(request, "INVALID_INPUT")
        return await _infer(request, task)

    @app.post("/infer/video")
    async def video(request: InferenceRequest, task: str = "detection"):
        if request.videoBase64 is None and request.fixtureId is None:
            return _rejected(request, "INVALID_INPUT")
        return await _infer(request, task)

    async def _infer(request: InferenceRequest, task: str):
        if not breaker.allow():
            response = _rejected_model(request, "UNAVAILABLE")
            _observe(response, latency, queue_wait, errors, service)
            return _json(response)
        response = await service.serve(request, task)
        breaker.record(response.errorCode)
        _observe(response, latency, queue_wait, errors, service)
        return _json(response)

    return app


def _gpu_memory_bytes() -> float:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        rows = result.stdout.split()
        return sum(int(row) for row in rows) * 1024**2 if rows else float("nan")
    except (OSError, subprocess.SubprocessError, ValueError):
        return float("nan")


def _json(response: InferenceResponse) -> JSONResponse:
    status = 200 if response.errorCode is None else http_status(response.errorCode)
    return JSONResponse(response.model_dump(mode="json"), status_code=status)


def _rejected(request: InferenceRequest, code: str) -> JSONResponse:
    body = InferenceResponse(
        requestId=request.requestId,
        modelVersion=request.modelVersion,
        source=request.source,
        receivedAt=request.receivedAt,
        result={},
        latencyMs=0,
        errorCode=code,
    )
    return _json(body)


def _rejected_model(request: InferenceRequest, code: str) -> InferenceResponse:
    return InferenceResponse(
        requestId=request.requestId,
        modelVersion=request.modelVersion,
        source=request.source,
        receivedAt=request.receivedAt,
        result={},
        latencyMs=0,
        errorCode=code,
    )


def _observe(response: InferenceResponse, latency: Histogram, queue_wait: Histogram, errors: Counter, service: InferenceService) -> None:
    latency.observe(response.latencyMs)
    queue_wait.observe(service.last_queue_wait_s * 1000)
    if response.errorCode is not None:
        errors.labels(errorCode=response.errorCode.value).inc()


def _error_payload(exc: RequestValidationError, code: str) -> dict:
    body = exc.body if isinstance(exc.body, dict) else {}
    return {
        "requestId": body.get("requestId", "unknown"),
        "modelVersion": body.get("modelVersion", "unknown"),
        "source": "eo",
        "receivedAt": datetime.now(timezone.utc).isoformat(),
        "result": {},
        "latencyMs": 0,
        "errorCode": code,
    }


app = create_app()
