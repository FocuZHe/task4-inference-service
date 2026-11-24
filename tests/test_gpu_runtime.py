from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import onnxruntime as ort
import pytest

from serving import detection_runtime
from serving import ort_runtime
from serving.detection_runtime import CPU_PROVIDER, CUDA_PROVIDER, DetectionRuntime


def test_ort_limits_threads_to_restricted_cpu_affinity(monkeypatch):
    monkeypatch.setattr(ort_runtime.os, "cpu_count", lambda: 128)
    monkeypatch.setattr(ort_runtime.os, "sched_getaffinity", lambda pid: {126, 127}, raising=False)

    assert ort_runtime.restricted_cpu_threads() == 2


def test_cuda_request_fails_when_gpu_provider_is_missing(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    model.touch()
    monkeypatch.setattr(ort, "get_available_providers", lambda: [CPU_PROVIDER])
    constructor = Mock()
    monkeypatch.setattr(ort, "InferenceSession", constructor)

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider is unavailable"):
        detection_runtime._open_session(model, CUDA_PROVIDER)

    constructor.assert_not_called()


def test_cuda_request_rejects_session_that_fell_back_to_cpu(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    model.touch()
    monkeypatch.setattr(ort, "get_available_providers", lambda: [CUDA_PROVIDER, CPU_PROVIDER])
    monkeypatch.setattr(ort, "preload_dlls", lambda: None, raising=False)
    session = SimpleNamespace(get_providers=lambda: [CPU_PROVIDER])
    monkeypatch.setattr(ort, "InferenceSession", Mock(return_value=session))

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider failed to initialize"):
        detection_runtime._open_session(model, CUDA_PROVIDER)


def test_cuda_warmup_and_batch_reuse_the_same_session(monkeypatch):
    monkeypatch.setenv("TASK4_ORT_PROVIDER", CUDA_PROVIDER)
    session = object()
    open_session = Mock(return_value=session)
    infer = Mock(return_value=[])
    monkeypatch.setattr(detection_runtime, "_open_session", open_session)
    monkeypatch.setattr(detection_runtime, "_infer", infer)
    runtime = DetectionRuntime()
    frame = np.zeros((24, 32, 3), dtype=np.uint8)

    runtime.warmup("model")
    assert runtime.predict_batch([frame] * 8, "model") == [[] for _ in range(8)]

    open_session.assert_called_once_with(detection_runtime.onnx_path("model", "fp32"), CUDA_PROVIDER)
    assert infer.call_count == 9
    assert all(call.args[0] is session for call in infer.call_args_list)


def test_cuda_warms_and_runs_eight_encoded_frames_together(tmp_path, monkeypatch):
    monkeypatch.setenv("TASK4_ORT_PROVIDER", CUDA_PROVIDER)
    monkeypatch.setattr(detection_runtime, "ONNX_DIR", tmp_path)
    model_path = detection_runtime.batch_onnx_path("model")
    model_path.touch()
    batches = Mock()
    batches.get_inputs.return_value = [SimpleNamespace(name="images")]
    batches.run.return_value = [np.arange(8).reshape(8, 1, 1)]
    static = object()
    open_session = Mock(side_effect=[static, batches])
    monkeypatch.setattr(detection_runtime, "_open_session", open_session)
    monkeypatch.setattr(detection_runtime, "_infer", Mock(return_value=[]))
    monkeypatch.setattr(
        detection_runtime, "_prepare_encoded",
        lambda payload: (np.zeros((1, 3, 2, 2), dtype=np.float32), 1.0, (0, 0), int(payload), 24),
    )
    monkeypatch.setattr(detection_runtime, "_decode", lambda raw, *_: [int(raw[0, 0, 0])])
    runtime = DetectionRuntime()

    runtime.warmup("model")
    result = runtime.predict_batch_encoded([str(i + 32) for i in range(8)], "model")

    assert result == [(i + 32, 24, [i]) for i in range(8)]
    open_session.assert_called_with(model_path, CUDA_PROVIDER)
    assert open_session.call_count == batches.run.call_count == 2
    assert [call.args[1]["images"].shape for call in batches.run.call_args_list] == [
        (8, 3, 640, 640), (8, 3, 2, 2),
    ]
