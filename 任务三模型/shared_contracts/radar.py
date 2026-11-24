from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from shared_contracts.common import ContractModel, Provenance, SourceModality

SCHEMA_VERSION = "shared.radar-plot/1.0"


class RadarPlot(ContractModel):
    """Polar radar measurement before conversion to an Observation."""

    schema_version: str = Field(alias="schemaVersion", default=SCHEMA_VERSION)
    plot_id: str = Field(alias="plotId", min_length=1, max_length=200)
    provenance: Provenance
    session_id: str = Field(alias="sessionId", min_length=1, max_length=200)
    frame_index: int = Field(alias="frameIndex", ge=0)
    range_m: float = Field(alias="rangeM")
    azimuth_deg: float = Field(alias="azimuthDeg")
    elevation_deg: float = Field(alias="elevationDeg")
    radial_velocity_mps: float | None = Field(default=None, alias="radialVelocityMps")
    snr_db: float | None = Field(default=None, alias="snrDb")
    quality: float | None = Field(default=None, ge=0, le=1)
    point_status_flags: int | None = Field(default=None, alias="pointStatusFlags", ge=0, le=0xFFFFFFFF)
    validity: Literal["valid", "invalid_flags", "out_of_range"]

    @model_validator(mode="after")
    def usable_plot_has_physical_angles(self) -> "RadarPlot":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schemaVersion must be {SCHEMA_VERSION}")
        if self.provenance.source is not SourceModality.RADAR:
            raise ValueError("radar plot provenance.source must be radar")
        numbers = (self.range_m, self.azimuth_deg, self.elevation_deg, self.radial_velocity_mps, self.snr_db)
        if any(item is not None and not math.isfinite(item) for item in numbers):
            raise ValueError("radar measurements must be finite")
        if self.validity != "valid":
            return self
        if self.range_m < 0 or not 0 <= self.azimuth_deg < 360 or not -90 <= self.elevation_deg <= 90:
            raise ValueError("valid radar plots must have physical range and angles")
        return self
