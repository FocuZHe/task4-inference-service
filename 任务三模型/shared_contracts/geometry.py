from __future__ import annotations

from pydantic import Field, model_validator

from shared_contracts.common import ContractModel


class Wgs84Point(ContractModel):
    """A surveyed or nominal point. It does not claim a measurement error."""

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    altitude_m: float | None = Field(default=None, alias="altitudeM")


class MeasuredPosition(Wgs84Point):
    """A position estimate. Horizontal error is required; height error follows height."""

    horizontal_std_m: float = Field(alias="horizontalStdM", gt=0)
    vertical_std_m: float | None = Field(default=None, alias="verticalStdM", gt=0)

    @model_validator(mode="after")
    def height_and_vertical_error_travel_together(self) -> "MeasuredPosition":
        if self.altitude_m is not None and self.vertical_std_m is None:
            raise ValueError("altitudeM requires verticalStdM")
        if self.altitude_m is None and self.vertical_std_m is not None:
            raise ValueError("verticalStdM requires altitudeM")
        return self
