from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from shared_contracts.common import AwareDatetime, ContractModel, SourceModality, TargetLabel
from shared_contracts.detection import ClassProbabilities
from shared_contracts.geometry import MeasuredPosition

SCHEMA_VERSION = "shared.fused-track/1.1"
REJECTION_SCHEMA_VERSION = "shared.fusion-rejection/1.0"


class Velocity(ContractModel):
    speed_mps: float = Field(alias="speedMps", ge=0)
    heading_deg: float | None = Field(default=None, alias="headingDeg", ge=0, lt=360)


class AssociationEvidence(ContractModel):
    observation_id: str = Field(alias="observationId", min_length=1, max_length=200)
    track_frame_index: int | None = Field(default=None, alias="trackFrameIndex", ge=0)
    observation_frame_index: int | None = Field(default=None, alias="observationFrameIndex", ge=0)
    time_delta_ms: int = Field(alias="timeDeltaMs", ge=0)
    bearing_delta_deg: float | None = Field(default=None, alias="bearingDeltaDeg")
    position_residual_m: float | None = Field(default=None, alias="positionResidualM", ge=0)
    decision: Literal["assigned", "rejected", "abstained"]


class Conflict(ContractModel):
    reason: str = Field(min_length=1, max_length=1000)
    observation_ids: list[str] = Field(default_factory=list, alias="observationIds")


class LabelHypothesis(ContractModel):
    label: TargetLabel
    confidence: float | None = Field(default=None, ge=0, le=1)
    class_probabilities: ClassProbabilities | None = Field(default=None, alias="classProbabilities")
    locked: bool = False


class FusedTrack(ContractModel):
    """Task 2 output. This message never grants permission to move a device."""

    schema_version: str = Field(alias="schemaVersion", default=SCHEMA_VERSION)
    track_id: str = Field(alias="trackId", min_length=1, max_length=200)
    lifecycle: Literal["tentative", "confirmed", "coasting", "deleted"]
    sources: list[SourceModality] = Field(min_length=1)
    device_ids: list[Annotated[str, Field(min_length=1, max_length=128)]] = Field(alias="deviceIds", min_length=1)
    first_seen_at: AwareDatetime = Field(alias="firstSeenAt")
    last_seen_at: AwareDatetime = Field(alias="lastSeenAt")
    published_at: AwareDatetime = Field(alias="publishedAt")
    position: MeasuredPosition
    velocity: Velocity
    freshness: Literal["expected", "aging", "stale", "expired"]
    association_evidence: list[AssociationEvidence] = Field(alias="associationEvidence", min_length=1)
    conflicts: list[Conflict]
    label_hypotheses: list[LabelHypothesis] = Field(default_factory=list, alias="labelHypotheses")
    fusion_version: str = Field(alias="fusionVersion", min_length=1, max_length=200)
    action_authority: Literal[False] = Field(alias="actionAuthority", default=False)

    @model_validator(mode="after")
    def track_times_are_ordered(self) -> "FusedTrack":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schemaVersion must be {SCHEMA_VERSION}")
        if self.last_seen_at < self.first_seen_at:
            raise ValueError("lastSeenAt must not precede firstSeenAt")
        return self


class FusionRejection(ContractModel):
    """A recorded decision not to fuse. The observation itself stays valid."""

    schema_version: str = Field(alias="schemaVersion", default=REJECTION_SCHEMA_VERSION)
    observation_id: str = Field(alias="observationId", min_length=1, max_length=200)
    reason_code: Literal[
        "time_invalid",
        "age_over_15s",
        "cross_session",
        "device_fault",
        "clock_unsynchronized",
        "calibration_missing",
        "no_spatial_measurement",
        "association_ambiguous",
        "outside_gate",
        "duplicate_frame",
        "invalid_radar_flags",
        "unusable_data",
    ] = Field(alias="reasonCode")
    fusion_version: str = Field(alias="fusionVersion", min_length=1, max_length=200)
    detail: str = Field(min_length=1, max_length=1000)
    observed_at: AwareDatetime | None = Field(default=None, alias="observedAt")
    received_at: AwareDatetime | None = Field(default=None, alias="receivedAt")

    @model_validator(mode="after")
    def known_version(self) -> "FusionRejection":
        if self.schema_version != REJECTION_SCHEMA_VERSION:
            raise ValueError(f"schemaVersion must be {REJECTION_SCHEMA_VERSION}")
        return self
