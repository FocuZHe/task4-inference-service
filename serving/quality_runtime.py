from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from functools import cached_property
from pathlib import Path
from threading import Lock

import joblib
import numpy as np

from serving.model_registry import ROOT
from serving.ort_runtime import restricted_cpu_threads

ONNX_DIR = ROOT / "artifacts" / "onnx"
REPORT_PATH = ROOT / "artifacts" / "reports" / "quality-parity.json"
QUALITY_PATH = ROOT / "任务三模型" / "data" / "raw" / "konvid-1k" / "models-official" / "quality_ridge.joblib"
EVENT_PATH = ROOT / "任务三模型" / "data" / "raw" / "konvid-1k" / "synthetic" / "full-frame-aligned-v3" / "models" / "event-rf.joblib"
SMOKE_VIDEO = ROOT / "任务三模型" / "data" / "raw" / "task3-smoke" / "engineering-smoke-10s.avi"
TASK3_ROOT = ROOT / "任务三模型"


class QualityRuntime:
    def __init__(self) -> None:
        self.quality_artifact = joblib.load(QUALITY_PATH)
        self.quality_model = self.quality_artifact["model"]
        self.quality_features = tuple(self.quality_artifact["features"])
        self._sessions: dict[str, object] = {}
        self._session_lock = Lock()

    @cached_property
    def event_artifact(self):
        return joblib.load(EVENT_PATH)

    @property
    def event_model(self):
        return self.event_artifact["model"]

    @property
    def event_features(self) -> tuple[str, ...]:
        return tuple(self.event_artifact["featureColumns"])

    @property
    def event_feature_count(self) -> int:
        return len(self.event_features)

    def score_features(self, features: np.ndarray) -> float:
        value = _run(self._session("quality-ridge-v1-fp32"), _matrix(features, len(self.quality_features)))
        return float(np.asarray(value).reshape(-1)[0])

    def event_proba(self, window: np.ndarray) -> dict[str, float]:
        probabilities = _run(self._session("event-rf-v1-fp32"), _matrix(window, self.event_feature_count), probabilities=True)
        labels = [str(label) for label in self.event_model.classes_]
        vector = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        return {label: float(vector[index]) for index, label in enumerate(labels)}

    def _session(self, name: str):
        with self._session_lock:
            if name not in self._sessions:
                self._sessions[name] = self._open_session(name)
            return self._sessions[name]

    def _open_session(self, name: str):
        import onnxruntime as ort

        path = ONNX_DIR / f"{name}.onnx"
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "event-rf-v1-fp32":
                _convert(self.event_model, self.event_feature_count, path, classifier=True)
            else:
                _convert(self.quality_model, len(self.quality_features), path, classifier=False)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        threads = restricted_cpu_threads()
        if threads is not None:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def build_quality_parity_report(runtime: QualityRuntime | None = None) -> dict:
    runtime = runtime or QualityRuntime()
    sample = np.array([[120.0, 110.0, 0.02, 0.01, 40.0, 3.5]], dtype=np.float64)
    native_score = float(runtime.quality_model.predict(sample)[0])
    onnx_score = runtime.score_features(sample)
    event_sample = np.random.default_rng(42).normal(size=(1, runtime.event_feature_count))
    native_class = str(runtime.event_model.predict(event_sample)[0])
    onnx_class = max(runtime.event_proba(event_sample), key=runtime.event_proba(event_sample).get)
    smoke = _smoke_comparison(runtime)
    conversion_errors = _try_reduced_precision()
    int8_score = score_ridge_int8(sample)
    report = {
        "fixedVector": {
            "nativeScore": native_score,
            "onnxScore": onnx_score,
            "absDiff": abs(native_score - onnx_score),
            "eventClass": native_class,
            "onnxEventClass": onnx_class,
        },
        "smoke": smoke,
        "files": [
            {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(ONNX_DIR.glob("quality-ridge-v1-*.onnx")) + sorted(ONNX_DIR.glob("event-rf-v1-*.onnx"))
        ],
        "conversionError": conversion_errors,
        "ridgeInt8": {
            "nativeScore": native_score,
            "int8Score": int8_score,
            "absDiff": abs(native_score - int8_score),
            "graph": "per-feature DequantizeLinear + Gemm; imputer and scaler folded for finite inputs",
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _convert(model, width: int, destination: Path, *, classifier: bool) -> None:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    options = {id(model): {"zipmap": False}} if classifier else None
    converted = convert_sklearn(
        model,
        initial_types=[("features", FloatTensorType([None, width]))],
        target_opset=12,
        options=options,
    )
    destination.write_bytes(converted.SerializeToString())


def _try_reduced_precision() -> dict[str, str]:
    errors: dict[str, str] = {}
    for name in ("quality-ridge-v1", "event-rf-v1"):
        source = ONNX_DIR / f"{name}-fp32.onnx"
        fp16 = ONNX_DIR / f"{name}-fp16.onnx"
        if not fp16.is_file():
            error = _convert_reduced(source, fp16, "fp16", timeout_s=45 if name.startswith("quality") else 60)
            if error:
                fp16.unlink(missing_ok=True)
                errors[fp16.name] = error
    ridge_int8 = ONNX_DIR / "quality-ridge-v1-int8.onnx"
    if not ridge_int8.is_file():
        _write_ridge_int8(ridge_int8)
    errors["event-rf-v1-int8.onnx"] = (
        "Unsupported: TreeEnsembleClassifier uses domain ai.onnx.ml. "
        "quantize_dynamic only accepts ai.onnx Conv/MatMul/Gemm. Serving stays on event-rf-v1-fp32.onnx."
    )
    return errors


def score_ridge_int8(features: np.ndarray) -> float:
    session = _ridge_int8_session()
    median, _, _ = _ridge_folded()
    matrix = _impute_nonfinite(_matrix(features, median.size), median)
    value = session.run(None, {"features": matrix})[0]
    return float(np.asarray(value).reshape(-1)[0])


def record_int8_status() -> dict:
    sample = np.array([[120.0, 110.0, 0.02, 0.01, 40.0, 3.5]], dtype=np.float64)
    native = float(joblib.load(QUALITY_PATH)["model"].predict(sample)[0])
    quantized = score_ridge_int8(sample)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8")) if REPORT_PATH.is_file() else {}
    errors = dict(report.get("conversionError") or {})
    errors.pop("quality-ridge-v1-int8.onnx", None)
    errors["event-rf-v1-int8.onnx"] = _try_reduced_precision()["event-rf-v1-int8.onnx"]
    path = ONNX_DIR / "quality-ridge-v1-int8.onnx"
    files = [item for item in report.get("files", []) if item.get("path") != path.name]
    files.append({"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size})
    report["conversionError"] = errors
    report["files"] = files
    report["ridgeInt8"] = {
        "nativeScore": native,
        "int8Score": quantized,
        "absDiff": abs(native - quantized),
        "graph": "per-feature DequantizeLinear + Gemm; imputer and scaler folded for finite inputs",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _ridge_int8_session():
    import onnxruntime as ort

    path = ONNX_DIR / "quality-ridge-v1-int8.onnx"
    if not path.is_file():
        _write_ridge_int8(path)
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _ridge_folded() -> tuple[np.ndarray, np.ndarray, np.float32]:
    imputer, scaler, ridge = (step for _, step in joblib.load(QUALITY_PATH)["model"].steps)
    median = np.asarray(imputer.statistics_, dtype=np.float32)
    mean = np.asarray(scaler.mean_, dtype=np.float32)
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    slope = np.asarray(ridge.coef_, dtype=np.float32).reshape(-1) / scale
    bias = np.float32(np.float32(ridge.intercept_) - np.dot(mean, slope))
    return median, slope.astype(np.float32), bias


def _write_ridge_int8(destination: Path) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    _, slope, bias = _ridge_folded()
    feature_scale = np.maximum(np.abs(slope) / np.float32(127.0), np.float32(1e-12))
    quantized = np.clip(np.round(slope / feature_scale), -128, 127).astype(np.int8)
    graph = helper.make_graph(
        [
            helper.make_node("DequantizeLinear", ["weight", "weight_scale", "weight_zero"], ["weight_f"], axis=0),
            helper.make_node("Gemm", ["features", "weight_f", "bias"], ["score"]),
        ],
        "quality-ridge-int8",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, int(slope.size)])],
        [helper.make_tensor_value_info("score", TensorProto.FLOAT, [None, 1])],
        [
            numpy_helper.from_array(quantized.reshape(-1, 1), "weight"),
            numpy_helper.from_array(feature_scale, "weight_scale"),
            numpy_helper.from_array(np.zeros(slope.shape, dtype=np.int8), "weight_zero"),
            numpy_helper.from_array(np.asarray([bias], dtype=np.float32), "bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)


def _impute_nonfinite(matrix: np.ndarray, median: np.ndarray) -> np.ndarray:
    filled = np.array(matrix, dtype=np.float32, copy=True)
    missing = ~np.isfinite(filled)
    if missing.any():
        filled[missing] = np.broadcast_to(median, filled.shape)[missing]
    return np.ascontiguousarray(filled)


def _convert_reduced(source: Path, destination: Path, precision: str, timeout_s: int) -> str | None:
    import subprocess
    import sys

    script = r"""
import shutil, sys, tempfile
from pathlib import Path
source, destination, precision = sys.argv[1:]
if precision == "fp16":
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    onnx.save(convert_float_to_float16(onnx.load(source), keep_io_types=True), destination)
else:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    with tempfile.TemporaryDirectory() as temporary:
        copied = Path(temporary) / "model.onnx"
        produced = Path(temporary) / "model-int8.onnx"
        shutil.copyfile(source, copied)
        quantize_dynamic(str(copied), str(produced), weight_type=QuantType.QInt8)
        shutil.copyfile(produced, destination)
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script, str(source), str(destination), precision],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return f"TimeoutError: {precision} conversion exceeded {timeout_s}s; FP32 ONNX remains the serving model"
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        return detail or f"conversion failed with exit code {completed.returncode}"
    return None


def _smoke_comparison(runtime: QualityRuntime) -> dict:
    if str(TASK3_ROOT) not in sys.path:
        sys.path.insert(0, str(TASK3_ROOT))
    from scripts.diagnose_events import predict_intervals
    from scripts.extract_combined_features import extract_quality_and_event_features

    with tempfile.TemporaryDirectory() as temporary:
        video = Path(temporary) / "engineering-smoke-10s.avi"
        shutil.copyfile(SMOKE_VIDEO, video)
        digest = _sha256(video)
        record = {
            "recordId": "task3-engineering-smoke-10s",
            "videoId": "task3-engineering-smoke-10s",
            "sourceVideoId": "task3-engineering-smoke-10s",
            "sourceVideoSha256": digest,
            "split": "service",
            "isSynthetic": True,
            "anomaly": None,
            "path": str(video),
            "videoSha256": digest,
        }
        quality_windows, event_rows = extract_quality_and_event_features(record)
    values = {
        name: float(np.mean([float(row[name]) for row in quality_windows]))
        for name in runtime.quality_features
    }
    matrix = np.asarray([[values[name] for name in runtime.quality_features]], dtype=np.float64)
    native_raw = float(runtime.quality_model.predict(matrix)[0])
    onnx_raw = runtime.score_features(matrix)
    normalization = runtime.quality_artifact["qualityScoreNormalization"]
    low, high = float(normalization["mosMin"]), float(normalization["mosMax"])
    native_score = float(np.clip((native_raw - low) / (high - low), 0.0, 1.0))
    onnx_score = float(np.clip((onnx_raw - low) / (high - low), 0.0, 1.0))
    intervals = predict_intervals(event_rows, runtime.event_artifact).get("task3-engineering-smoke-10s", {})
    reported_intervals = {
        key: value
        for key, value in intervals.items()
        if key != "__processingTimeMs"
    }
    return {
        "videoId": "task3-engineering-smoke-10s",
        "isSynthetic": True,
        "nativeQualityScore": native_score,
        "onnxQualityScore": onnx_score,
        "absDiff": abs(native_score - onnx_score),
        "intervals": reported_intervals,
    }


def _matrix(features: np.ndarray, width: int) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[1] != width:
        raise ValueError(f"expected {width} features, got {matrix.shape[1]}")
    return np.ascontiguousarray(matrix)


def _run(session, matrix: np.ndarray, probabilities: bool = False):
    result = session.run(None, {"features": matrix})
    if not probabilities:
        return result[0]
    for value in result:
        array = np.asarray(value)
        if array.ndim == 2 and array.shape[1] > 1:
            return array
    return result[-1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
