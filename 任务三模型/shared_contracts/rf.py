from __future__ import annotations

from pydantic import Field, model_validator

from shared_contracts.common import ContractModel, Provenance, SourceModality, TargetLabel

SCHEMA_VERSION = "shared.rf-classification/1.0"


class RfClassification(ContractModel):
    """Task 1 radio classification. rfLabel stays the model vocabulary."""

    schema_version: str = Field(alias="schemaVersion", default=SCHEMA_VERSION)
    classification_id: str = Field(alias="classificationId", min_length=1, max_length=200)
    provenance: Provenance
    rf_label: str = Field(alias="rfLabel", min_length=1, max_length=200)
    target_label: TargetLabel | None = Field(default=None, alias="targetLabel")
    confidence: float = Field(ge=0, le=1)
    snr_db: float | None = Field(default=None, alias="snrDb")

    @model_validator(mode="after")
    def rf_provenance(self) -> "RfClassification":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schemaVersion must be {SCHEMA_VERSION}")
        if self.provenance.source is not SourceModality.RF:
            raise ValueError("rf classification provenance.source must be rf")
        if self.provenance.model_version is None:
            raise ValueError("rf classification provenance.modelVersion is required")
        return self
