from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, AfterValidator


def require_aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone offset")
    return value


AwareDatetime = Annotated[datetime, AfterValidator(require_aware_datetime)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class SourceModality(str, Enum):
    RADAR = "radar"
    RF = "rf"
    EO = "eo"


class TargetLabel(str, Enum):
    DRONE = "drone"
    BIRD = "bird"
    AIRCRAFT = "aircraft"
    UNKNOWN = "unknown"


class DataQuality(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    UNUSABLE = "unusable"
    UNKNOWN = "unknown"


class TimeQuality(str, Enum):
    SYNCHRONIZED = "synchronized"
    ESTIMATED = "estimated"
    UNSYNCHRONIZED = "unsynchronized"
    UNKNOWN = "unknown"


class DeviceStatus(str, Enum):
    OK = "ok"
    FAULT = "fault"
    UNKNOWN = "unknown"


class EvidenceRef(ContractModel):
    """Pointer to an immutable raw artifact. Omit the list until the artifact exists."""

    uri: str = Field(min_length=1, max_length=1000)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_time_s: float | None = Field(default=None, alias="mediaTimeS", ge=0)


class Provenance(ContractModel):
    """Fields every task must carry when it emits a result or an observation."""

    source: SourceModality
    device_id: str = Field(alias="deviceId", min_length=1, max_length=128)
    observed_at: AwareDatetime = Field(alias="observedAt")
    received_at: AwareDatetime = Field(alias="receivedAt")
    data_quality: DataQuality = Field(alias="dataQuality")
    time_quality: TimeQuality = Field(alias="timeQuality")
    model_version: str | None = Field(default=None, alias="modelVersion", min_length=1, max_length=200)
    device_status: DeviceStatus = Field(default=DeviceStatus.UNKNOWN, alias="deviceStatus")
    site_id: str | None = Field(default=None, alias="siteId", min_length=1, max_length=128)
    channel_id: str | None = Field(default=None, alias="channelId", min_length=1, max_length=128)
    clock_offset_ms: int | None = Field(default=None, alias="clockOffsetMs")
    evidence_refs: list[EvidenceRef] = Field(default_factory=list, alias="evidenceRefs")
