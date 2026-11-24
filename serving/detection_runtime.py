from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal

import cv2
import numpy as np

from serving.image_decode import decode_bgr
from serving.model_registry import ROOT, ModelRegistry
from serving.ort_runtime import restricted_cpu_threads

ONNX_DIR = ROOT / "artifacts" / "onnx"
REPORT_PATH = ROOT / "artifacts" / "reports" / "detection-parity.json"
CONFIDENCE = 0.25
IOU = 0.7
IMAGE_SIZE = 640
CUDA_BATCH_SIZE = 8
CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
Precision = Literal["fp32", "fp16", "int8"]


@dataclass(frozen=True)
class DetectionBox:
    bboxXyxy: tuple[float, float, float, float]
    label: Literal["drone"]
    confidence: float
    classId: int


class DetectionRuntime:
    def __init__(self) -> None:
        self.provider = os.environ.get("TASK4_ORT_PROVIDER", CPU_PROVIDER)
        self._cpu_limit = restricted_cpu_threads() if self.provider == CPU_PROVIDER else None
        self._sessions: dict[tuple[str, str], object] = {}
        self._session_lock = Lock()
        self._parallel: dict[str, list] = {}
        self._cuda_batches: dict[str, object] = {}
        self._pool = ThreadPoolExecutor(max_workers=min(self._cpu_limit or 8, 8), thread_name_prefix="detect-batch")
        if self._cpu_limit is not None or self.provider == CUDA_PROVIDER:
            cv2.setNumThreads(1)

    def predict(self, image: np.ndarray, model_id: str, precision: Precision = "fp32") -> list[DetectionBox]:
        return _infer(self._session(model_id, precision), image)

    def warmup(self, model_id: str) -> None:
        self.predict(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8), model_id)
        if self.provider == CUDA_PROVIDER and batch_onnx_path(model_id).is_file():
            inputs = np.zeros((CUDA_BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            _run_cuda_batch(self._cuda_batch_session(model_id), inputs)

    def predict_batch_encoded(self, payloads: list[str], model_id: str, precision: Precision = "fp32") -> list[tuple[int, int, list[DetectionBox]]]:
        if (
            self.provider == CUDA_PROVIDER
            and precision == "fp32"
            and len(payloads) == CUDA_BATCH_SIZE
            and batch_onnx_path(model_id).is_file()
        ):
            prepared = list(self._pool.map(_prepare_encoded, payloads))
            inputs = np.concatenate([item[0] for item in prepared], axis=0)
            raw = _run_cuda_batch(self._cuda_batch_session(model_id), inputs)
            return [
                (width, height, _decode(raw[index:index + 1], width, height, ratio, pad))
                for index, (_, ratio, pad, width, height) in enumerate(prepared)
            ]
        if self.provider == CUDA_PROVIDER or precision != "fp32" or len(payloads) <= 1:
            frames = map(decode_bgr, payloads)
            return [(frame.shape[1], frame.shape[0], self.predict(frame, model_id, precision)) for frame in frames]
        workers = self._parallel_sessions(model_id)
        results: list[tuple[int, int, list[DetectionBox]] | None] = [None] * len(payloads)

        def run(index: int) -> None:
            frame = decode_bgr(payloads[index])
            results[index] = (frame.shape[1], frame.shape[0], _infer(workers[index % len(workers)], frame))

        self._run_parallel(run, len(payloads))
        return results  # type: ignore[return-value]

    def predict_batch(self, images: list[np.ndarray], model_id: str, precision: Precision = "fp32") -> list[list[DetectionBox]]:
        if self.provider == CUDA_PROVIDER or precision != "fp32" or len(images) <= 1:
            return [self.predict(image, model_id, precision) for image in images]
        workers = self._parallel_sessions(model_id)
        results: list[list[DetectionBox] | None] = [None] * len(images)

        def run(index: int) -> None:
            results[index] = _infer(workers[index % len(workers)], images[index])

        self._run_parallel(run, len(images))
        return results  # type: ignore[return-value]

    def _run_parallel(self, run, count: int) -> None:
        if self._cpu_limit is not None:
            list(self._pool.map(run, range(count)))
            return
        previous = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            list(self._pool.map(run, range(count)))
        finally:
            cv2.setNumThreads(previous)

    def _parallel_sessions(self, model_id: str) -> list:
        with self._session_lock:
            if model_id not in self._parallel:
                count = min(self._cpu_limit or 8, 8)
                threads = 1 if self._cpu_limit is not None and self._cpu_limit < 12 else 2
                self._parallel[model_id] = [
                    _open_session(onnx_path(model_id, "fp32"), self.provider, intra_op_threads=threads, spinning=False)
                    for _ in range(count)
                ]
            return self._parallel[model_id]

    def _session(self, model_id: str, precision: Precision):
        key = (model_id, precision)
        with self._session_lock:
            if key not in self._sessions:
                self._sessions[key] = _open_session(onnx_path(model_id, precision), self.provider)
            return self._sessions[key]

    def _cuda_batch_session(self, model_id: str):
        with self._session_lock:
            if model_id not in self._cuda_batches:
                self._cuda_batches[model_id] = _open_session(batch_onnx_path(model_id), CUDA_PROVIDER)
            return self._cuda_batches[model_id]


def onnx_path(model_id: str, precision: Precision) -> Path:
    return ONNX_DIR / f"{model_id}-{precision}.onnx"


def batch_onnx_path(model_id: str) -> Path:
    return ONNX_DIR / f"{model_id}-batch-fp32.onnx"


def _open_session(path: Path, provider: str, intra_op_threads: int | None = None, spinning: bool = True):
    import onnxruntime as ort

    if not path.is_file():
        raise FileNotFoundError(path)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_op_threads is None:
        intra_op_threads = 1 if provider == CUDA_PROVIDER else restricted_cpu_threads()
    if intra_op_threads is not None:
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if not spinning:
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    if provider != CUDA_PROVIDER:
        return ort.InferenceSession(str(path), sess_options=options, providers=[provider])

    if CUDA_PROVIDER not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is unavailable. Install the GPU build of ONNX Runtime.")
    # The Linux image installs CUDA and cuDNN as NVIDIA Python packages.
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    cuda_options = {
        "device_id": 0,
        "use_tf32": 0,
        "cudnn_conv_algo_search": "HEURISTIC",
        "gpu_mem_limit": 3 * 1024**3,
    }
    session = ort.InferenceSession(str(path), sess_options=options, providers=[(CUDA_PROVIDER, cuda_options), CPU_PROVIDER])
    if session.get_providers()[0] != CUDA_PROVIDER:
        raise RuntimeError("CUDAExecutionProvider failed to initialize. Check NVIDIA driver, CUDA and cuDNN.")
    session.disable_fallback()
    return session


def _prepare_encoded(payload: str) -> tuple[np.ndarray, float, tuple[float, float], int, int]:
    frame = decode_bgr(payload)
    tensor, ratio, pad = _preprocess(frame)
    return tensor, ratio, pad, frame.shape[1], frame.shape[0]


def _run_cuda_batch(session, inputs: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    raw = session.run(None, {input_name: inputs})[0]
    if raw.shape[0] != inputs.shape[0]:
        raise RuntimeError("FP32 batch output cardinality mismatch")
    return raw


def _infer(session, image: np.ndarray) -> list[DetectionBox]:
    tensor, ratio, pad = _preprocess(image)
    input_meta = session.get_inputs()[0]
    if "float16" in input_meta.type:
        tensor = tensor.astype(np.float16)
    raw = session.run(None, {input_meta.name: tensor})[0]
    return _decode(raw, image.shape[1], image.shape[0], ratio, pad)


def build_detection_parity_report() -> dict:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "artifacts" / "runtime-config"))
    registry = ModelRegistry.load(ROOT / "artifacts" / "registry" / "models.json")
    model_ids = [record.modelId for record in registry.records.values() if record.family == "detection"]
    conversion_errors: dict[str, str] = {}
    for model_id in model_ids:
        conversion_errors.update(_export_variants(registry.records[model_id].absolute_path, model_id))

    runtime = DetectionRuntime()
    samples = sorted((ROOT / "任务一模型" / "demo" / "samples").glob("*.jpg"))
    rows: list[dict] = []
    errors: list[dict] = []
    files: list[dict] = []
    for model_id in model_ids:
        weight = registry.records[model_id].absolute_path
        for precision in ("fp32", "fp16", "int8"):
            path = onnx_path(model_id, precision)
            if not path.is_file():
                continue
            files.append({"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size})
            for sample in samples:
                image = _read_bgr(sample)
                pytorch_boxes = _pytorch_predict(weight, image)
                onnx_boxes = runtime.predict(image, model_id, precision)  # type: ignore[arg-type]
                agreement, max_distance, max_confidence = _compare(pytorch_boxes, onnx_boxes)
                row = {
                    "precision": precision,
                    "modelId": model_id,
                    "sample": sample.name,
                    "pytorchBoxes": len(pytorch_boxes),
                    "onnxBoxes": len(onnx_boxes),
                    "classAgreement": agreement,
                    "maxCenterDistancePx": round(max_distance, 4),
                    "maxConfidenceAbsDiff": round(max_confidence, 6),
                }
                rows.append(row)
                if max_distance > 5 or max_confidence > 0.05:
                    errors.append(row)
    report = {"rows": rows, "errors": errors, "files": files, "conversionError": conversion_errors}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _export_variants(weight: Path, model_id: str) -> dict[str, str]:
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    errors: dict[str, str] = {}
    fp32 = onnx_path(model_id, "fp32")
    if not fp32.is_file():
        from ultralytics import YOLO

        exported = Path(YOLO(str(weight)).export(format="onnx", imgsz=IMAGE_SIZE, opset=12, dynamic=False, half=False, simplify=True))
        shutil.move(exported, fp32)
    fp16 = onnx_path(model_id, "fp16")
    if not fp16.is_file():
        try:
            _write_fp16(fp32, fp16)
        except Exception as exc:
            errors[fp16.name] = f"{type(exc).__name__}: {exc}"
    int8 = onnx_path(model_id, "int8")
    if not int8.is_file():
        try:
            _write_int8(fp32, int8)
        except Exception as exc:
            errors[int8.name] = f"{type(exc).__name__}: {exc}"
    return errors


def _write_fp16(source: Path, destination: Path) -> None:
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = onnx.load(source)
    converted = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(converted, destination)


def _write_int8(source: Path, destination: Path) -> None:
    import tempfile

    from onnxruntime.quantization import QuantType, quantize_dynamic

    with tempfile.TemporaryDirectory() as temporary:
        ascii_source = Path(temporary) / "model.onnx"
        ascii_destination = Path(temporary) / "model-int8.onnx"
        shutil.copyfile(source, ascii_source)
        quantize_dynamic(str(ascii_source), str(ascii_destination), weight_type=QuantType.QInt8)
        shutil.copyfile(ascii_destination, destination)


def _pytorch_predict(weight: Path, image: np.ndarray) -> list[DetectionBox]:
    from ultralytics import YOLO

    result = YOLO(str(weight)).predict(image, imgsz=IMAGE_SIZE, conf=CONFIDENCE, iou=IOU, rect=False, device="cpu", verbose=False)[0]
    boxes: list[DetectionBox] = []
    if result.boxes is None:
        return boxes
    for x1, y1, x2, y2, confidence, class_id in result.boxes.data.cpu().tolist():
        boxes.append(DetectionBox((float(x1), float(y1), float(x2), float(y2)), "drone", float(confidence), int(class_id)))
    return boxes


def _preprocess(image: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
    canvas, ratio, pad = _letterbox(image, IMAGE_SIZE)
    rgb = canvas[..., ::-1] if canvas.ndim == 3 and canvas.shape[2] == 3 else cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = np.empty((1, 3, canvas.shape[0], canvas.shape[1]), dtype=np.float32)
    np.divide(rgb.transpose(2, 0, 1), np.float32(255), out=tensor[0], dtype=np.float32)
    return tensor, ratio, pad


def _letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    resized = (int(round(width * ratio)), int(round(height * ratio)))
    dw, dh = size - resized[0], size - resized[1]
    dw /= 2
    dh /= 2
    canvas = cv2.resize(image, resized, interpolation=cv2.INTER_LINEAR) if (width, height) != resized else image
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    canvas = cv2.copyMakeBorder(canvas, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return canvas, ratio, (dw, dh)


def _decode(raw: np.ndarray, width: int, height: int, ratio: float, pad: tuple[float, float]) -> list[DetectionBox]:
    prediction = np.squeeze(raw, axis=0)
    if prediction.shape[0] < prediction.shape[1]:
        prediction = prediction.T
    boxes = prediction[:, :4]
    class_scores = prediction[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    confidence = class_scores.max(axis=1)
    keep = confidence >= CONFIDENCE
    boxes, confidence, class_ids = boxes[keep], confidence[keep], class_ids[keep]
    if len(boxes) == 0:
        return []
    xyxy = _xywh_to_xyxy(boxes)
    xyxy[:, [0, 2]] -= pad[0]
    xyxy[:, [1, 3]] -= pad[1]
    xyxy /= ratio
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)
    order = _nms(xyxy, confidence, IOU)[:300]
    return [
        DetectionBox(tuple(float(value) for value in xyxy[index]), "drone", float(confidence[index]), int(class_ids[index]))
        for index in order
    ]


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = np.empty_like(boxes)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while len(order):
        current = int(order[0])
        kept.append(current)
        if len(order) == 1:
            break
        rest = order[1:]
        intersection = _intersection(boxes[current], boxes[rest])
        union = _area(boxes[current]) + _area(boxes[rest]) - intersection
        iou = intersection / np.maximum(union, 1e-6)
        order = rest[iou <= iou_threshold]
    return kept


def _area(boxes: np.ndarray) -> np.ndarray:
    return np.maximum(boxes[..., 2] - boxes[..., 0], 0) * np.maximum(boxes[..., 3] - boxes[..., 1], 0)


def _intersection(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    return np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)


def _compare(pytorch_boxes: list[DetectionBox], onnx_boxes: list[DetectionBox]) -> tuple[float, float, float]:
    labels = [box.label for box in pytorch_boxes + onnx_boxes]
    if not labels:
        return 1.0, 0.0, 0.0
    agreement = 1.0 if all(label == "drone" for label in labels) else 0.0
    unused = set(range(len(onnx_boxes)))
    max_distance = 0.0
    max_confidence = 0.0
    for box in pytorch_boxes:
        if not unused:
            break
        match = min(unused, key=lambda index: _center_distance(box, onnx_boxes[index]))
        distance = _center_distance(box, onnx_boxes[match])
        confidence_gap = abs(box.confidence - onnx_boxes[match].confidence)
        max_distance = max(max_distance, distance)
        max_confidence = max(max_confidence, confidence_gap)
        unused.remove(match)
    return agreement, max_distance, max_confidence


def _center_distance(left: DetectionBox, right: DetectionBox) -> float:
    lx = (left.bboxXyxy[0] + left.bboxXyxy[2]) / 2
    ly = (left.bboxXyxy[1] + left.bboxXyxy[3]) / 2
    rx = (right.bboxXyxy[0] + right.bboxXyxy[2]) / 2
    ry = (right.bboxXyxy[1] + right.bboxXyxy[3]) / 2
    return float(np.hypot(lx - rx, ly - ry))


def _read_bgr(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
