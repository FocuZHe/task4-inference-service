from __future__ import annotations

import json
from pathlib import Path

from shared_contracts.common import Provenance
from shared_contracts.detection import Detection
from shared_contracts.fusion import FusedTrack, FusionRejection
from shared_contracts.observation import Observation
from shared_contracts.quality import QualityAnalyzeRequest, QualityJobAccepted, QualityJobSnapshot, VideoQualityResult
from shared_contracts.radar import RadarPlot
from shared_contracts.rf import RfClassification
from shared_contracts.site import SiteFrame

SCHEMAS = {
    "provenance": Provenance,
    "detection": Detection,
    "rf-classification": RfClassification,
    "observation": Observation,
    "radar-plot": RadarPlot,
    "site-frame": SiteFrame,
    "fused-track": FusedTrack,
    "fusion-rejection": FusionRejection,
    "video-quality": VideoQualityResult,
    "quality-analyze-request": QualityAnalyzeRequest,
    "quality-job": QualityJobAccepted,
    "quality-job-status": QualityJobSnapshot,
}


def schema_document(name: str) -> dict:
    model = SCHEMAS[name]
    return model.model_json_schema()


def write_schemas(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in SCHEMAS:
        path = directory / f"{name}.schema.json"
        path.write_text(json.dumps(schema_document(name), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
