from __future__ import annotations

import argparse
import base64
import json
import math

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.client import HTTPConnection
from pathlib import Path

from benchmark import ROOT, first_1080p_frame, fixture_jpegs

REPORT = ROOT / "artifacts" / "reports" / "latency.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--image", type=Path)
    inputs.add_argument("--fixture-frames", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--port", type=int, default=8871)
    parser.add_argument("--existing-server", action="store_true", help="Use a running loopback service")
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args()
    if args.requests < 1 or not 1 <= args.batch <= 8:
        parser.error("--requests must be positive and --batch must be between 1 and 8")
    if args.fixture_frames < 1:
        parser.error("--fixture-frames must be positive")
    frames = fixture_jpegs(args.fixture_frames) if args.fixture_frames > 1 else [first_1080p_frame(args.image)]
    _, width, height = frames[0]
    if width < 1920 or height < 1080:
        raise SystemExit(f"frame is {width}x{height}, expected at least 1920x1080")
    encoded = [base64.b64encode(jpeg).decode("ascii") for jpeg, _, _ in frames]
    server = None if args.existing_server else _start_server(args.port)
    try:
        _wait_until_ready(args.port)
        single = _measure(args.port, encoded, args.requests, batch=1)
        batch = _measure(args.port, encoded, args.requests, batch=args.batch)
        rss_bytes, gpu_bytes = _memory_snapshot(args.port)
    finally:
        if server is not None:
            _stop_server(server)
    single_frames_per_s = single["frames"] / single["elapsedS"]
    batch_frames_per_s = batch["frames"] / batch["elapsedS"]
    ratio = batch_frames_per_s / single_frames_per_s if single_frames_per_s else 0.0
    single_p95 = _percentile(single["clientMs"], 95)
    single_p99 = _percentile(single["clientMs"], 99)
    batch_p95 = _percentile(batch["clientMs"], 95)
    batch_p99 = _percentile(batch["clientMs"], 99)
    p95_speedup = args.batch * single_p95 / batch_p95
    p99_speedup = args.batch * single_p99 / batch_p99
    no_errors = not single["errors"] and not batch["errors"]
    report = {
        "image": str(args.image) if args.image else f"artifacts/fixtures/ten-second-1080p.mp4#frames=0-{len(encoded) - 1}",
        "fixtureFrames": len(encoded),
        "distinctInputFrames": len(set(encoded)),
        "width": width,
        "height": height,
        "requests": args.requests,
        "batchSize": args.batch,
        "p50Ms": _percentile(single["clientMs"], 50),
        "p95Ms": single_p95,
        "p99Ms": single_p99,
        "serverP50Ms": _percentile(single["serverMs"], 50),
        "serverP95Ms": _percentile(single["serverMs"], 95),
        "serverP99Ms": _percentile(single["serverMs"], 99),
        "throughputPerS": round(single_frames_per_s, 4),
        "queueWaitP95Ms": _percentile(single["queueMs"], 95),
        "queueWaitNote": "clientLatencyMs minus server latencyMs; includes HTTP transfer of the JPEG, not only queue time",
        "errorRate": single["errors"] / args.requests,
        "rssBytes": rss_bytes,
        "gpuBytes": gpu_bytes,
        "batch8Throughput": round(batch_frames_per_s, 4),
        "singleThroughput": round(single_frames_per_s, 4),
        "batchThroughputRatio": round(ratio, 4),
        "singleFrameP95Pass": not single["errors"] and single_p95 <= 300,
        "batchThroughputPass": no_errors and ratio >= 2,
        "batchP95Speedup": round(p95_speedup, 4),
        "batchP99Speedup": round(p99_speedup, 4),
        "batchP95SpeedupPass": no_errors and p95_speedup >= 2,
        "batchP99SpeedupPass": no_errors and p99_speedup >= 2,
        "enabledMeans": [
            "imgsz 640",
            "POST /infer/batch",
            "JPEG and PNG dimensions read from the header",
            "benchmark client reuses one HTTP connection",
            "orjson for the 8-frame JSON body",
        ],
        "batchErrorRate": batch["errors"] / args.requests,
        "batchP95Ms": batch_p95,
        "batchP99Ms": batch_p99,
        "batchServerP95Ms": _percentile(batch["serverMs"], 95),
        "batchServerP99Ms": _percentile(batch["serverMs"], 99),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    keys = (
        "p95Ms", "p99Ms", "throughputPerS", "batch8Throughput", "batchThroughputRatio",
        "batchP95Ms", "batchP99Ms", "batchP95Speedup", "batchP99Speedup", "errorRate", "batchErrorRate",
    )
    print(json.dumps({key: report[key] for key in keys}, ensure_ascii=False))
    return int(bool(single["errors"] or batch["errors"]))


def _memory_snapshot(port: int) -> tuple[int | None, int | None]:
    from prometheus_client.parser import text_string_to_metric_families

    names = ("process_rss_bytes", "gpu_memory_bytes")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
            metrics = text_string_to_metric_families(response.read().decode("utf-8"))
            values = {
                sample.name: int(sample.value)
                for metric in metrics
                for sample in metric.samples
                if sample.name in names and math.isfinite(sample.value)
            }
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None, None
    return values.get(names[0]), values.get(names[1])


def _measure(port: int, encoded: list[str], requests: int, batch: int) -> dict:
    connection = HTTPConnection("127.0.0.1", port, timeout=60)
    try:
        body, status = _post(port, _payload(encoded, batch, "warmup"), connection)
        if status != 200 or body.get("errorCode"):
            raise SystemExit(f"warmup failed (batch={batch}, HTTP {status}, {body.get('errorCode')})")
        server_ms: list[float] = []
        client_ms: list[float] = []
        errors = 0
        started = time.perf_counter()
        for index in range(requests):
            request_id = f"{'batch' if batch > 1 else 'single'}-{index}"
            payload = _payload(encoded, batch, request_id, start=index * batch)
            client_started = time.perf_counter()
            body, status = _post(port, payload, connection)
            client_ms.append((time.perf_counter() - client_started) * 1000)
            if status != 200 or body.get("errorCode"):
                errors += 1
            server_ms.append(float(body.get("latencyMs") or 0))
        elapsed = time.perf_counter() - started
    finally:
        connection.close()
    queue_ms = [max(0.0, client - server) for client, server in zip(client_ms, server_ms)]
    return {"serverMs": server_ms, "clientMs": client_ms, "queueMs": queue_ms, "errors": errors, "elapsedS": elapsed, "frames": requests * batch}


def _payload(encoded: list[str], batch: int, request_id: str, *, start: int = 0) -> dict:
    payload = {
        "requestId": request_id,
        "modelVersion": "yolo11n-dut-new",
        "source": "eo",
        "receivedAt": datetime.now(timezone.utc).isoformat(),
    }
    if batch == 1:
        payload["imageBase64"] = encoded[start % len(encoded)]
    else:
        payload["framesBase64"] = [encoded[(start + offset) % len(encoded)] for offset in range(batch)]
    return payload


def _post(port: int, payload: dict, connection: HTTPConnection | None = None) -> tuple[dict, int]:
    path = "/infer/frame" if "imageBase64" in payload else "/infer/batch"
    body = orjson.dumps(payload) if orjson is not None else json.dumps(payload).encode("utf-8")
    if connection is None:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return _loads(response.read()), response.status
        except urllib.error.HTTPError as exc:
            return _loads(exc.read()), exc.code
    connection.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    return _loads(response.read()), response.status


def _loads(raw: bytes) -> dict:
    if orjson is not None:
        return orjson.loads(raw)
    return json.loads(raw)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = (len(ordered) - 1) * percentile / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return round(ordered[low] * (1 - weight) + ordered[high] * weight, 4)


def _start_server(port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "serving.app:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT,
        env=_server_env(),
    )


def _wait_until_ready(port: int) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.5)
    raise SystemExit("inference service did not become healthy")


def _server_env() -> dict:
    env = os.environ.copy()
    env["TASK4_TIMEOUT_S"] = "30"
    return env


def _stop_server(server: subprocess.Popen) -> None:
    server.terminate()
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
