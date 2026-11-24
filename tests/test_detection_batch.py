from unittest.mock import Mock

import numpy as np

from serving import detection_runtime
from serving.detection_runtime import DetectionRuntime


def test_batched_fp32_boxes_match_single_frame_predictions():
    runtime = DetectionRuntime()
    images = [
        np.random.default_rng(42).integers(0, 255, (360, 640, 3), dtype=np.uint8),
        np.random.default_rng(43).integers(0, 255, (1080, 1920, 3), dtype=np.uint8),
    ]
    batched = runtime.predict_batch(images, "yolo11n-dut-new")
    singles = [runtime.predict(image, "yolo11n-dut-new") for image in images]
    assert len(batched) == len(singles)
    for group, single in zip(batched, singles):
        assert [(box.label, box.classId) for box in group] == [(box.label, box.classId) for box in single]
        for left, right in zip(group, single):
            assert abs(left.confidence - right.confidence) < 0.05
            assert max(abs(a - b) for a, b in zip(left.bboxXyxy, right.bboxXyxy)) < 1.0


def test_restricted_cpu_batch_limits_sessions_and_preserves_order(monkeypatch):
    monkeypatch.setenv("TASK4_ORT_PROVIDER", detection_runtime.CPU_PROVIDER)
    monkeypatch.setattr(detection_runtime, "restricted_cpu_threads", lambda: 4)
    monkeypatch.setattr(detection_runtime.cv2, "setNumThreads", lambda threads: None)
    open_session = Mock()
    monkeypatch.setattr(detection_runtime, "_open_session", open_session)
    monkeypatch.setattr(detection_runtime, "_infer", lambda session, image: int(image[0, 0, 0]))
    runtime = DetectionRuntime()
    images = [np.full((2, 2, 3), value, dtype=np.uint8) for value in range(8)]

    assert runtime.predict_batch(images, "model") == list(range(8))
    assert open_session.call_count == 4
    assert all(call.args[0] == detection_runtime.onnx_path("model", "fp32") for call in open_session.call_args_list)
    open_session.assert_called_with(
        detection_runtime.onnx_path("model", "fp32"),
        detection_runtime.CPU_PROVIDER,
        intra_op_threads=1,
        spinning=False,
    )
