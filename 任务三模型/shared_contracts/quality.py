from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from shared_contracts.common import ContractModel, Provenance, SourceModality

SCHEMA_VERSION = "shared.video-quality/1.0"
JOB_SCHEMA_VERSION = "shared.quality-job/1.0"
JOB_STATUS_SCHEMA_VERSION = "shared.quality-job-status/1.0"


class AnomalyInterval(ContractModel):
    type: Literal["blur", "dark", "overexposed", "freeze_suspected", "decode_error"]
    start_time_s: float = Field(alias="startTimeS", ge=0)
    end_time_s: float = Field(alias="endTimeS", ge=0)
    status: Literal["suspected", "confirmed"]
    evidence_time_s: float | None = Field(default=None, alias="evidenceTimeS", ge=0)

    @model_validator(mode="after")
    def interval_is_consistent(self) -> "AnomalyInterval":
        if self.end_time_s < self.start_time_s:
            raise ValueError("endTimeS must not precede startTimeS")
        if self.type == "freeze_suspected" and self.status != "suspected":
            raise ValueError("freeze_suspected can only be reported as suspected")
        if self.evidence_time_s is not None and not self.start_time_s <= self.evidence_time_s <= self.end_time_s:
            raise ValueError("evidenceTimeS must fall inside the anomaly interval")
        return self


class VideoQualityResult(ContractModel):
    """Task 3 output. A quality score is not a detection probability."""

    schema_version: str = Field(alias="schemaVersion", default=SCHEMA_VERSION)
    job_id: str = Field(alias="jobId", min_length=1, max_length=200)
    video_id: str = Field(alias="videoId", min_length=1, max_length=200)
    provenance: Provenance
    channel_type: Literal["visible", "infrared", "unknown"] = Field(alias="channelType")
    is_synthetic: bool = Field(alias="isSynthetic")
    limitations: list[str]
    score_status: Literal["scored", "rejected"] = Field(alias="scoreStatus")
    quality_score: float | None = Field(default=None, alias="qualityScore", ge=0, le=1)
    anomalies: list[AnomalyInterval]
    processing_time_ms: int = Field(alias="processingTimeMs", ge=0)

    @model_validator(mode="after")
    def score_matches_status(self) -> "VideoQualityResult":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schemaVersion must be {SCHEMA_VERSION}")
        if self.provenance.source is not SourceModality.EO:
            raise ValueError("video quality provenance.source must be eo")
        if self.provenance.model_version is None:
            raise ValueError("video quality provenance.modelVersion is required")
        if self.score_status == "scored" and self.quality_score is None:
            raise ValueError("scored results require qualityScore")
        if self.score_status == "rejected" and self.quality_score is not None:
            raise ValueError("rejected results must not include qualityScore")
        if self.channel_type == "infrared" and not any(item.strip() for item in self.limitations):
            raise ValueError("infrared results must state their limitations")
        return self


class QualityAnalyzeRequest(ContractModel):
    schema_version: str = Field(alias="schemaVersion", default=JOB_SCHEMA_VERSION)
    video_id: str = Field(alias="videoId", min_length=1, max_length=200)

    @model_validator(mode="after")
    def known_version(self) -> "QualityAnalyzeRequest":
        if self.schema_version != JOB_SCHEMA_VERSION:
            raise ValueError(f"schemaVersion must be {JOB_SCHEMA_VERSION}")
        return self


class QualityJobError(ContractModel):
    code: Literal["empty", "corrupt", "missing_timestamp", "unsupported"]
    message: str = Field(min_length=1, max_length=1000)


class QualityJobSnapshot(ContractModel):
    """GET /quality/jobs/{jobId}. A failed job has an error and no quality score."""

    schema_version: str = Field(alias="schemaVersion", default=JOB_STATUS_SCHEMA_VERSION)
    job_id: str = Field(alias="jobId", min_length=1, max_length=200)
    state: Literal["queued", "running", "succeeded", "failed"]
    result: VideoQualityResult | None = None
    error: QualityJobError | None = None

    @model_validator(mode="after")
    def state_matches_payload(self) -> "QualityJobSnapshot":
        if self.schema_version != JOB_STATUS_SCHEMA_VERSION:
            raise ValueError(f"schemaVersion must be {JOB_STATUS_SCHEMA_VERSION}")
        if self.state in {"queued", "running"} and (self.result is not None or self.error is not None):
            raise ValueError("queued and running jobs must not include result or error")
        if self.state == "succeeded":
            if self.result is None or self.error is not None:
                raise ValueError("succeeded jobs require result and no error")
            if self.result.job_id != self.job_id:
                raise ValueError("result.jobId must match jobId")
        if self.state == "failed" and (self.error is None or self.result is not None):
            raise ValueError("failed jobs require error and no result")
        return self


class QualityJobAccepted(ContractModel):
    schema_version: str = Field(alias="schemaVersion", default=JOB_SCHEMA_VERSION)
    job_id: str = Field(alias="jobId", min_length=1, max_length=200)

    @model_validator(mode="after")
    def known_version(self) -> "QualityJobAccepted":
        if self.schema_version != JOB_SCHEMA_VERSION:
            raise ValueError(f"schemaVersion must be {JOB_SCHEMA_VERSION}")
        return self
