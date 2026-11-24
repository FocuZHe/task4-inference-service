from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from shared_contracts.common import ContractModel
from shared_contracts.geometry import Wgs84Point

SCHEMA_VERSION = "shared.site-frame/1.0"


class SiteFrame(ContractModel):
    """Site origin used to convert WGS84 and local ENU. Tasks can omit it until surveyed."""

    schema_version: str = Field(alias="schemaVersion", default=SCHEMA_VERSION)
    site_id: str = Field(alias="siteId", min_length=1, max_length=128)
    frame_id: str = Field(alias="frameId", min_length=1, max_length=200)
    origin: Wgs84Point
    calibration_status: Literal["calibrated", "unknown", "invalid"] = Field(alias="calibrationStatus")
    calibration_ref: str | None = Field(default=None, alias="calibrationRef", min_length=1, max_length=200)

    @model_validator(mode="after")
    def calibrated_frame_names_its_revision(self) -> "SiteFrame":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schemaVersion must be {SCHEMA_VERSION}")
        if self.calibration_status == "calibrated" and self.calibration_ref is None:
            raise ValueError("calibrationRef is required when calibrationStatus is calibrated")
        return self
