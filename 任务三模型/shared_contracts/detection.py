from __future__ import annotations

import math
from typing import Annotated

from pydantic import Field, AfterValidator, model_validator

from shared_contracts.common import ContractModel, Provenance, SourceModality, TargetLabel

SCHEMA_VERSION = "shared.detection/1.1"


def require_bbox(value: list[float]) -> list[float]:
    if len(value) != 4 or any(not math.isfinite(item) for item in value):
        raise ValueError("bboxXyxy must contain four finite numbers")
    x1, y1, x2, y2 = value
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bboxXyxy must have positive width and height")
    return value


def require_class_probabilities(value: dict[str, float]) -> dict[str, float]:
    allowed = {label.value for label in TargetLabel}
    if set(value) - allowed:
        raise ValueError("classProbabilities keys must be drone, bird, aircraft, or unknown")
    if any(not math.isfinite(item) or not 0 <= item <= 1 for item in value.values()):
        raise ValueError("classProbabilities must be finite and within 0 and 1")
    if not math.isclose(sum(value.values()), 1.0, abs_tol=1e-6):
        raise ValueError("classProbabilities must sum to 1")
    return value


BBoxXyxy = Annotated[list[float], AfterValidator(require_bbox)]
ClassProbabilities = Annotated[dict[str, float], AfterValidator(require_class_probabilities)]


class Detection(ContractModel):
    """Task 1 photoelectric detection. A box is not a geographic position."""

    schema_version: str = Field(alias="schemaVersion", default=SCHEMA_VERSION)
    detection_id: str = Field(alias="detectionId", min_length=1, max_length=200)
    provenance: Provenance
    frame_index: int | None = Field(default=None, alias="frameIndex", ge=0)
    video_id: str | None = Field(default=None, alias="videoId", min_length=1, max_length=200)
    media_time_s: float | None = Field(default=None, alias="mediaTimeS", ge=0)
    image_width: int = Field(alias="imageWidth", gt=0)
    image_height: int = Field(alias="imageHeight", gt=0)
    bbox_xyxy: BBoxXyxy = Field(alias="bboxXyxy")
    label: TargetLabel
    confidence: float = Field(ge=0, le=1)
    class_probabilities: ClassProbabilities | None = Field(default=None, alias="classProbabilities")
    label_confirmed: bool = Field(default=False, alias="labelConfirmed")

    @model_validator(mode="after")
    def detection_provenance(self) -> "Detection":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schemaVersion must be {SCHEMA_VERSION}")
        if self.provenance.source is not SourceModality.EO:
            raise ValueError("detection provenance.source must be eo")
        if self.provenance.model_version is None:
            raise ValueError("detection provenance.modelVersion is required")
        x1, y1, x2, y2 = self.bbox_xyxy
        if x1 < 0 or y1 < 0 or x2 > self.image_width or y2 > self.image_height:
            raise ValueError("bboxXyxy must lie inside the image")
        return self
