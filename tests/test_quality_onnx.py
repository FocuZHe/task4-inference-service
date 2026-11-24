import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts" / "reports" / "quality-parity.json"


@pytest.fixture(scope="module")
def runtime():
    from serving.quality_runtime import QualityRuntime

    return QualityRuntime()


def test_ridge_fp32_matches_joblib(runtime):
    sample = np.array([[120.0, 110.0, 0.02, 0.01, 40.0, 3.5]], dtype=np.float64)
    native = runtime.quality_model.predict(sample)[0]
    onnx_value = runtime.score_features(sample)
    assert abs(float(native) - float(onnx_value)) < 1e-4


def test_event_fp32_class_and_probability_match(runtime):
    sample = np.random.default_rng(42).normal(size=(1, runtime.event_feature_count))
    native_class = str(runtime.event_model.predict(sample)[0])
    native_proba = runtime.event_model.predict_proba(sample)[0]
    onnx_proba = runtime.event_proba(sample)
    onnx_class = max(onnx_proba, key=onnx_proba.get)
    assert onnx_class == native_class
    paired = [abs(float(onnx_proba[str(name)]) - float(native_proba[index])) for index, name in enumerate(runtime.event_model.classes_)]
    assert max(paired) < 1e-4


def test_smoke_video_quality_score_matches(runtime):
    from serving.quality_runtime import build_quality_parity_report

    report = build_quality_parity_report(runtime)
    assert REPORT.is_file()
    assert report["smoke"]["absDiff"] < 1e-3
    assert "macroF1" not in json.dumps(report)
    assert "intervals" in report["smoke"]
