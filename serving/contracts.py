from __future__ import annotations

import base64
import struct
from datetime import datetime
from enum import Enum
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ErrorCode(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    MODEL_MISSING = "MODEL_MISSING"
    TIMEOUT = "TIMEOUT"
    QUEUE_FULL = "QUEUE_FULL"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    CANCELLED = "CANCELLED"
    UNAVAILABLE = "UNAVAILABLE"
    INTERNAL = "INTERNAL"


HTTP_STATUS_FOR_ERROR = {
    ErrorCode.INVALID_INPUT: 400,
    ErrorCode.MODEL_MISSING: 503,
    ErrorCode.TIMEOUT: 408,
    ErrorCode.CANCELLED: 429,
    ErrorCode.QUEUE_FULL: 503,
    ErrorCode.UNAVAILABLE: 503,
    ErrorCode.RESOURCE_EXHAUSTED: 500,
    ErrorCode.INTERNAL: 500,
}


def http_status(error_code: ErrorCode) -> int:
    return HTTP_STATUS_FOR_ERROR[ErrorCode(error_code)]


def _require_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("receivedAt must include a timezone offset")
    return value


def _decode_image(value: str) -> tuple[int, int]:
    try:
        raw = base64.b64decode(value[:87380] + "=" * ((-len(value[:87380])) % 4), validate=False)
    except Exception as exc:
        raise ValueError("image is not valid base64") from exc
    header = _header_size(raw)
    if header is None:
        try:
            raw = base64.b64decode(value, validate=True)
        except Exception as exc:
            raise ValueError("image is not valid base64") from exc
        width, height = _image_size(raw)
    else:
        width, height = header
    if width <= 0 or height <= 0:
        raise ValueError("image could not be decoded")
    if max(height, width) > 1920:
        raise ValueError("image long edge exceeds 1920")
    return width, height


def _image_size(raw: bytes) -> tuple[int, int]:
    header = _header_size(raw)
    if header is not None:
        return header
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("image could not be decoded")
    height, width = image.shape[:2]
    return width, height


def _header_size(raw: bytes) -> tuple[int, int] | None:
    if raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24:
        width, height = struct.unpack(">II", raw[16:24])
        return int(width), int(height)
    if raw.startswith(b"\xff\xd8"):
        return _jpeg_size(raw)
    return None


def _jpeg_size(raw: bytes) -> tuple[int, int] | None:
    sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    index = 2
    limit = len(raw)
    while index + 1 < limit:
        if raw[index] != 0xFF:
            return None
        marker = raw[index + 1]
        index += 2
        if marker in (0x01, *range(0xD0, 0xD9)):
            continue
        if index + 2 > limit:
            return None
        length = int.from_bytes(raw[index : index + 2], "big")
        if length < 2 or index + length > limit:
            return None
        if marker in sof and index + 7 <= limit:
            height, width = struct.unpack(">HH", raw[index + 3 : index + 7])
            return int(width), int(height)
        index += length
    return None


class DetectionBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bboxXyxy: list[float] = Field(min_length=4, max_length=4)
    label: Literal["drone", "bird", "aircraft", "unknown"]
    confidence: float = Field(ge=0, le=1)
    imageWidth: int = Field(gt=0)
    imageHeight: int = Field(gt=0)

    @model_validator(mode="after")
    def box_is_inside_image(self) -> "DetectionBox":
        x1, y1, x2, y2 = self.bboxXyxy
        if not all(np.isfinite(item) for item in (x1, y1, x2, y2)):
            raise ValueError("bboxXyxy must contain finite numbers")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bboxXyxy must have positive width and height")
        if x1 < 0 or y1 < 0 or x2 > self.imageWidth or y2 > self.imageHeight:
            raise ValueError("bboxXyxy must lie inside the image")
        return self


class DetectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    boxes: list[DetectionBox]


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["eo"]
    deviceId: str = Field(min_length=1, max_length=128)
    observedAt: datetime
    receivedAt: datetime
    dataQuality: Literal["ok", "degraded", "unusable", "unknown"]
    timeQuality: Literal["synchronized", "estimated", "unsynchronized", "unknown"]
    modelVersion: str = Field(min_length=1, max_length=200)
    deviceStatus: Literal["ok", "fault", "unknown"] = "unknown"
    evidenceRefs: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def timestamps_are_aware(self) -> "Provenance":
        _require_timezone(self.observedAt)
        _require_timezone(self.receivedAt)
        return self


class AnomalyInterval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["blur", "dark", "overexposed", "freeze_suspected", "decode_error"]
    startTimeS: float = Field(ge=0)
    endTimeS: float = Field(ge=0)
    status: Literal["suspected", "confirmed"]
    evidenceTimeS: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def interval_is_consistent(self) -> "AnomalyInterval":
        if self.endTimeS < self.startTimeS:
            raise ValueError("endTimeS must not precede startTimeS")
        if self.type == "freeze_suspected" and self.status != "suspected":
            raise ValueError("freeze_suspected can only be reported as suspected")
        if self.evidenceTimeS is not None and not self.startTimeS <= self.evidenceTimeS <= self.endTimeS:
            raise ValueError("evidenceTimeS must fall inside the anomaly interval")
        return self


class QualityResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schemaVersion: Literal["shared.video-quality/1.0"]
    jobId: str = Field(min_length=1, max_length=200)
    videoId: str = Field(min_length=1, max_length=200)
    provenance: Provenance
    channelType: Literal["visible", "infrared", "unknown"]
    isSynthetic: bool
    limitations: list[str]
    scoreStatus: Literal["scored", "rejected"]
    qualityScore: float | None = Field(default=None, ge=0, le=1)
    anomalies: list[AnomalyInterval]
    processingTimeMs: int = Field(ge=0)

    @model_validator(mode="after")
    def score_matches_status(self) -> "QualityResult":
        if self.scoreStatus == "scored" and self.qualityScore is None:
            raise ValueError("scored results require qualityScore")
        if self.scoreStatus == "rejected" and self.qualityScore is not None:
            raise ValueError("rejected results must not include qualityScore")
        if self.channelType == "infrared" and not any(item.strip() for item in self.limitations):
            raise ValueError("infrared results must state their limitations")
        return self


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requestId: str = Field(min_length=1, max_length=200)
    modelVersion: str = Field(min_length=1, max_length=200)
    source: Literal["eo"]
    receivedAt: datetime
    imageBase64: str | None = None
    framesBase64: list[str] | None = Field(default=None, min_length=1, max_length=8)
    videoBase64: str | None = None
    fixtureId: str | None = None

    @model_validator(mode="after")
    def input_is_usable(self) -> "InferenceRequest":
        _require_timezone(self.receivedAt)
        supplied = [self.imageBase64, self.framesBase64, self.videoBase64, self.fixtureId]
        if sum(item is not None for item in supplied) != 1:
            raise ValueError("request must include exactly one of imageBase64, framesBase64, videoBase64, or fixtureId")
        if self.imageBase64 is not None:
            _decode_image(self.imageBase64)
        if self.framesBase64 is not None:
            for frame in self.framesBase64:
                _decode_image(frame)
        if self.videoBase64 is not None:
            try:
                raw = base64.b64decode(self.videoBase64, validate=True)
            except Exception as exc:
                raise ValueError("videoBase64 is not valid base64") from exc
            if not raw:
                raise ValueError("videoBase64 is empty")
        return self


class InferenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requestId: str = Field(min_length=1, max_length=200)
    modelVersion: str = Field(min_length=1, max_length=200)
    source: Literal["eo"]
    receivedAt: datetime
    result: dict[str, Any]
    latencyMs: float = Field(ge=0)
    errorCode: ErrorCode | None = None

    @model_validator(mode="after")
    def result_matches_outcome(self) -> "InferenceResponse":
        _require_timezone(self.receivedAt)
        if self.errorCode is not None:
            return self
        if "frames" in self.result:
            for frame in self.result["frames"]:
                DetectionResult.model_validate(frame)
            return self
        if "boxes" in self.result:
            DetectionResult.model_validate(self.result)
            return self
        if self.result.get("schemaVersion") == "shared.video-quality/1.0":
            QualityResult.model_validate(self.result)
            return self
        raise ValueError("successful result must contain detection boxes or a video quality result")


def error_response(
    *,
    requestId: str,
    modelVersion: str,
    receivedAt: datetime,
    errorCode: ErrorCode,
    latencyMs: float,
    source: Literal["eo"] = "eo",
) -> InferenceResponse:
    return InferenceResponse(
        requestId=requestId,
        modelVersion=modelVersion,
        source=source,
        receivedAt=receivedAt,
        result={},
        latencyMs=latencyMs,
        errorCode=errorCode,
    )
