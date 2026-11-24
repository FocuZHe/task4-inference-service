import json
from pathlib import Path

import pytest

from serving.model_registry import ModelRegistry

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "artifacts" / "registry" / "models.json"


def _load(tmp_path: Path) -> ModelRegistry:
    target = tmp_path / "models.json"
    target.write_text(CANONICAL.read_text(encoding="utf-8"), encoding="utf-8")
    return ModelRegistry.load(target)


def test_rollback_restores_previous_detection(tmp_path):
    registry = _load(tmp_path)
    assert registry.active("detection").modelId == "yolo11n-dut-new"
    registry.activate("yolo11n-dut-old")
    registry.rollback("detection")
    assert registry.active("detection").modelId == "yolo11n-dut-new"


def test_hash_mismatch_refuses_activation(tmp_path):
    registry = _load(tmp_path)
    registry.records["yolo11n-dut-new"].sha256 = "0" * 64
    with pytest.raises(registry.RegistryError):
        registry.activate("yolo11n-dut-new")


def test_quality_switch_moves_both_models(tmp_path):
    registry = _load(tmp_path)
    assert registry.activate("quality-candidate-v2") == "quality-candidate-v2"
    assert registry.active("quality").modelId == "quality-candidate-v2"
    assert registry.records["event-candidate-v2"].role == "active"
    assert registry.records["quality-ridge-v1"].role == "rollback"
    assert registry.records["event-rf-v1"].role == "rollback"
    assert registry.rollback("quality") == "quality-ridge-v1"
    assert registry.records["event-rf-v1"].role == "active"


def test_switch_success_rate_is_one(tmp_path):
    registry = _load(tmp_path)
    successes = 0
    attempts = 0
    for _ in range(3):
        attempts += 1
        if registry.activate("yolo11n-dut-old") == "yolo11n-dut-old":
            successes += 1
        attempts += 1
        if registry.rollback("detection") == "yolo11n-dut-new":
            successes += 1
    assert successes / attempts == 1.0


def test_v13_candidate_is_not_registered():
    text = CANONICAL.read_text(encoding="utf-8")
    payload = json.loads(text)
    blob = json.dumps(payload)
    assert "formal-candidate-v13" not in blob
    assert "f5053c500443490b76c13c5ce1b23cce6867545fa11385c2888d5bc656995d22" not in blob
    assert len(payload["models"]) == 6
