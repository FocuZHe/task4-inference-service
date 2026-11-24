from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from benchmark import ROOT, first_1080p_frame

LOG = ROOT / "artifacts" / "reports" / "soak.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=float, default=30)
    parser.add_argument("--interval-s", type=float, default=10)
    parser.add_argument("--port", type=int, default=8872)
    parser.add_argument("--existing-server", action="store_true", help="Use a running loopback service")
    parser.add_argument("--output", type=Path, default=LOG)
    args = parser.parse_args()
    if args.duration_s <= 0 or args.interval_s <= 0:
        parser.error("--duration-s and --interval-s must be positive")
    jpeg, width, height = first_1080p_frame()
    encoded = base64.b64encode(jpeg).decode("ascii")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("", encoding="utf-8")
    env = os.environ.copy()
    env["TASK4_TIMEOUT_S"] = "30"
    server = None if args.existing_server else subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "serving.app:app", "--host", "127.0.0.1", "--port", str(args.port)],
        cwd=ROOT,
        env=env,
    )
    unhandled = 0
    errors = 0
    try:
        _wait_until_ready(args.port, time.monotonic() + 120)
        deadline = time.monotonic() + args.duration_s
        index = 0
        while time.monotonic() < deadline:
            loop_started = time.monotonic()
            index += 1
            try:
                errors += not _submit(args.port, encoded, width, height, index, args.output)
            except Exception:
                unhandled += 1
                _write(args.output, {"at": _now(), "requestId": f"soak-{index}", "errorCode": "UNHANDLED", "latencyMs": None, "unhandled": True, "traceback": traceback.format_exc()})
                break
            remaining = args.interval_s - (time.monotonic() - loop_started)
            if remaining > 0:
                time.sleep(min(remaining, max(0.0, deadline - time.monotonic())))
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
    return int(bool(unhandled or errors))


def _submit(port: int, encoded: str, width: int, height: int, index: int, output: Path) -> bool:
    payload = {
        "requestId": f"soak-{index}",
        "modelVersion": "yolo11n-dut-new",
        "source": "eo",
        "receivedAt": _now(),
        "imageBase64": encoded,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/infer/frame",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
            status = response.status
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read().decode("utf-8"))
        status = exc.code
    _write(output, {
        "at": _now(),
        "requestId": payload["requestId"],
        "status": status,
        "errorCode": body.get("errorCode"),
        "latencyMs": body.get("latencyMs"),
        "clientLatencyMs": round((time.perf_counter() - started) * 1000, 2),
        "width": width,
        "height": height,
        "unhandled": False,
    })
    return status == 200 and not body.get("errorCode")


def _write(output: Path, row: dict) -> None:
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _wait_until_ready(port: int, deadline: float) -> None:
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.5)
    raise RuntimeError("inference service did not become healthy")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
