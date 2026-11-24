from __future__ import annotations

import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from shared_contracts.detection import Detection
from shared_contracts.export import SCHEMAS, schema_document
from shared_contracts.fusion import FusedTrack, FusionRejection
from shared_contracts.observation import Observation
from shared_contracts.quality import QualityAnalyzeRequest, QualityJobAccepted, QualityJobSnapshot, VideoQualityResult
from shared_contracts.radar import RadarPlot
from shared_contracts.rf import RfClassification
from shared_contracts.site import SiteFrame

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "contracts" / "examples"
SCHEMAS_DIR = ROOT / "contracts" / "schemas"

EXAMPLE_MODELS = {
    "detection.json": Detection,
    "rf-classification.json": RfClassification,
    "observation-position.json": Observation,
    "observation-bearing.json": Observation,
    "observation-no-geometry.json": Observation,
    "fused-track.json": FusedTrack,
    "video-quality.json": VideoQualityResult,
    "quality-analyze-request.json": QualityAnalyzeRequest,
    "quality-job.json": QualityJobAccepted,
    "quality-job-queued.json": QualityJobSnapshot,
    "quality-job-failed.json": QualityJobSnapshot,
    "radar-plot.json": RadarPlot,
    "site-frame.json": SiteFrame,
    "fusion-rejection.json": FusionRejection,
}


def load_example(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


class ExampleContractTest(unittest.TestCase):
    def test_examples_match_models(self) -> None:
        self.assertEqual(set(EXAMPLE_MODELS), {path.name for path in EXAMPLES.glob("*.json")})
        for name, model in EXAMPLE_MODELS.items():
            with self.subTest(example=name):
                model.model_validate(load_example(name))

    def test_committed_schemas_match_models(self) -> None:
        for name in SCHEMAS:
            path = SCHEMAS_DIR / f"{name}.schema.json"
            committed = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(committed, schema_document(name), name)


class DetectionRuleTest(unittest.TestCase):
    def test_box_without_timezone_is_rejected(self) -> None:
        payload = load_example("detection.json")
        payload["provenance"]["observedAt"] = "2026-10-27T02:00:00"
        with self.assertRaises(ValidationError):
            Detection.model_validate(payload)

    def test_class_probabilities_must_sum_to_one(self) -> None:
        payload = load_example("detection.json")
        payload["classProbabilities"] = {"drone": 0.8, "unknown": 0.1}
        with self.assertRaises(ValidationError):
            Detection.model_validate(payload)


class ObservationRuleTest(unittest.TestCase):
    def test_missing_position_is_rejected_for_wgs84(self) -> None:
        payload = load_example("observation-position.json")
        payload["position"] = None
        with self.assertRaises(ValidationError):
            Observation.model_validate(payload)

    def test_bearing_only_observation_cannot_invent_position(self) -> None:
        payload = load_example("observation-bearing.json")
        payload["position"] = {"lat": 0, "lon": 0}
        with self.assertRaises(ValidationError):
            Observation.model_validate(payload)

    def test_old_sample_remains_a_valid_message(self) -> None:
        payload = load_example("observation-position.json")
        payload["provenance"]["receivedAt"] = "2026-10-27T02:00:20Z"
        Observation.model_validate(payload)


class MeasurementRuleTest(unittest.TestCase):
    def test_position_without_error_is_rejected(self) -> None:
        payload = load_example("observation-position.json")
        del payload["position"]["horizontalStdM"]
        with self.assertRaises(ValidationError):
            Observation.model_validate(payload)

    def test_bearing_without_origin_is_rejected(self) -> None:
        payload = load_example("observation-bearing.json")
        payload["origin"] = None
        with self.assertRaises(ValidationError):
            Observation.model_validate(payload)

    def test_invalid_radar_angle_can_still_be_recorded(self) -> None:
        payload = load_example("radar-plot.json")
        payload["azimuthDeg"] = 400
        payload["validity"] = "out_of_range"
        RadarPlot.model_validate(payload)

    def test_calibrated_site_requires_a_revision(self) -> None:
        payload = load_example("site-frame.json")
        payload["calibrationRef"] = None
        with self.assertRaises(ValidationError):
            SiteFrame.model_validate(payload)


class FusionRuleTest(unittest.TestCase):
    def test_track_cannot_authorize_action(self) -> None:
        payload = load_example("fused-track.json")
        payload["actionAuthority"] = True
        with self.assertRaises(ValidationError):
            FusedTrack.model_validate(payload)


class QualityRuleTest(unittest.TestCase):
    def test_rejected_file_cannot_carry_a_score(self) -> None:
        payload = load_example("video-quality.json")
        payload["scoreStatus"] = "rejected"
        with self.assertRaises(ValidationError):
            VideoQualityResult.model_validate(payload)

    def test_freeze_cannot_be_confirmed_from_a_still_scene(self) -> None:
        payload = load_example("video-quality.json")
        payload["anomalies"][0]["type"] = "freeze_suspected"
        payload["anomalies"][0]["status"] = "confirmed"
        with self.assertRaises(ValidationError):
            VideoQualityResult.model_validate(payload)

    def test_infrared_must_declare_limitations(self) -> None:
        payload = load_example("video-quality.json")
        payload["channelType"] = "infrared"
        payload["limitations"] = []
        with self.assertRaises(ValidationError):
            VideoQualityResult.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
