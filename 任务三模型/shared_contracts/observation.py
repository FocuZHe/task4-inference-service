from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from shared_contracts.common import ContractModel, Provenance
from shared_contracts.geometry import MeasuredPosition, Wgs84Point

SCHEMA_VERSION = "shared.observation/1.1"


class Observation(ContractModel):
    """Task 2 input. Position is present only when it was actually measured or converted."""

    schema_version: str = Field(alias="schemaVersion", default=SCHEMA_VERSION)
    observation_id: str = Field(alias="observationId", min_length=1, max_length=200)
    provenance: Provenance
    session_id: str = Field(alias="sessionId", min_length=1, max_length=200)
    frame_index: int | None = Field(default=None, alias="frameIndex", ge=0)
    coordinate_system: Literal["EPSG:4326", "true-north-bearing", "none"] = Field(alias="coordinateSystem")
    position: MeasuredPosition | None = None
    bearing_deg: float | None = Field(default=None, alias="bearingDeg", ge=0, lt=360)
    bearing_std_deg: float | None = Field(default=None, alias="bearingStdDeg", gt=0)
    origin: Wgs84Point | None = None
    calibration_ref: str | None = Field(default=None, alias="calibrationRef", min_length=1, max_length=200)
    quality: float | None = Field(default=None, ge=0, le=1)
    detection_id: str | None = Field(default=None, alias="detectionId", min_length=1, max_length=200)
    classification_id: str | None = Field(default=None, alias="classificationId", min_length=1, max_length=200)

    @model_validator(mode="after")
    def geometry_matches_coordinate_system(self) -> "Observation":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schemaVersion must be {SCHEMA_VERSION}")
        bearing_is_paired = (self.bearing_deg is None) == (self.bearing_std_deg is None)
        if not bearing_is_paired:
            raise ValueError("bearingDeg and bearingStdDeg must be supplied together")
        if self.coordinate_system == "EPSG:4326":
            if self.position is None or self.calibration_ref is None:
                raise ValueError("EPSG:4326 requires position and calibrationRef")
            if self.origin is not None:
                raise ValueError("EPSG:4326 must not include origin")
        elif self.coordinate_system == "true-north-bearing":
            if self.position is not None:
                raise ValueError("true-north-bearing must not include position")
            if self.bearing_deg is None or self.origin is None or self.calibration_ref is None:
                raise ValueError("true-north-bearing requires bearingDeg, bearingStdDeg, origin, and calibrationRef")
        else:
            occupied = (
                self.position is not None
                or self.bearing_deg is not None
                or self.origin is not None
                or self.calibration_ref is not None
            )
            if occupied:
                raise ValueError("coordinateSystem none must not include position, bearing, origin, or calibrationRef")
        return self
