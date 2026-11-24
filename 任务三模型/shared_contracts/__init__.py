"""Cross-task message contracts for detection, fusion, and video quality."""

from shared_contracts.common import (
    DataQuality,
    DeviceStatus,
    EvidenceRef,
    Provenance,
    SourceModality,
    TargetLabel,
    TimeQuality,
)
from shared_contracts.detection import SCHEMA_VERSION as DETECTION_SCHEMA_VERSION
from shared_contracts.detection import Detection
from shared_contracts.fusion import REJECTION_SCHEMA_VERSION as FUSION_REJECTION_SCHEMA_VERSION
from shared_contracts.fusion import SCHEMA_VERSION as FUSED_TRACK_SCHEMA_VERSION
from shared_contracts.fusion import AssociationEvidence, Conflict, FusedTrack, FusionRejection, LabelHypothesis, Velocity
from shared_contracts.geometry import MeasuredPosition, Wgs84Point
from shared_contracts.observation import SCHEMA_VERSION as OBSERVATION_SCHEMA_VERSION
from shared_contracts.observation import Observation
from shared_contracts.quality import JOB_SCHEMA_VERSION as QUALITY_JOB_SCHEMA_VERSION
from shared_contracts.quality import JOB_STATUS_SCHEMA_VERSION as QUALITY_JOB_STATUS_SCHEMA_VERSION
from shared_contracts.quality import SCHEMA_VERSION as VIDEO_QUALITY_SCHEMA_VERSION
from shared_contracts.quality import (
    AnomalyInterval,
    QualityAnalyzeRequest,
    QualityJobAccepted,
    QualityJobError,
    QualityJobSnapshot,
    VideoQualityResult,
)
from shared_contracts.radar import SCHEMA_VERSION as RADAR_PLOT_SCHEMA_VERSION
from shared_contracts.radar import RadarPlot
from shared_contracts.rf import SCHEMA_VERSION as RF_CLASSIFICATION_SCHEMA_VERSION
from shared_contracts.rf import RfClassification
from shared_contracts.site import SCHEMA_VERSION as SITE_FRAME_SCHEMA_VERSION
from shared_contracts.site import SiteFrame

__all__ = [
    "AnomalyInterval",
    "AssociationEvidence",
    "Conflict",
    "DETECTION_SCHEMA_VERSION",
    "DataQuality",
    "Detection",
    "DeviceStatus",
    "EvidenceRef",
    "FUSED_TRACK_SCHEMA_VERSION",
    "FUSION_REJECTION_SCHEMA_VERSION",
    "FusedTrack",
    "FusionRejection",
    "LabelHypothesis",
    "MeasuredPosition",
    "OBSERVATION_SCHEMA_VERSION",
    "Observation",
    "Provenance",
    "QUALITY_JOB_SCHEMA_VERSION",
    "QUALITY_JOB_STATUS_SCHEMA_VERSION",
    "QualityAnalyzeRequest",
    "QualityJobAccepted",
    "QualityJobError",
    "QualityJobSnapshot",
    "RADAR_PLOT_SCHEMA_VERSION",
    "RF_CLASSIFICATION_SCHEMA_VERSION",
    "RadarPlot",
    "RfClassification",
    "SITE_FRAME_SCHEMA_VERSION",
    "SiteFrame",
    "SourceModality",
    "TargetLabel",
    "TimeQuality",
    "VIDEO_QUALITY_SCHEMA_VERSION",
    "Velocity",
    "VideoQualityResult",
    "Wgs84Point",
]
