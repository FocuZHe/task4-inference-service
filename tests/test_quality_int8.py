import json

import joblib
import numpy as np

from serving.quality_runtime import QUALITY_PATH, REPORT_PATH, record_int8_status, score_ridge_int8

SAMPLE = np.array([[120.0, 110.0, 0.02, 0.01, 40.0, 3.5]], dtype=np.float64)


def test_ridge_int8_matches_joblib_on_fixed_vector():
    native = float(joblib.load(QUALITY_PATH)["model"].predict(SAMPLE)[0])
    quantized = score_ridge_int8(SAMPLE)
    assert abs(native - quantized) < 1e-2


def test_event_forest_int8_stays_unsupported():
    report = record_int8_status()
    saved = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    message = saved["conversionError"]["event-rf-v1-int8.onnx"]
    assert "ai.onnx.ml" in message
    assert "quality-ridge-v1-int8.onnx" not in saved["conversionError"]
    assert any(item["path"] == "quality-ridge-v1-int8.onnx" for item in saved["files"])
    assert report["ridgeInt8"]["absDiff"] < 1e-2
