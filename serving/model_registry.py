from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


class RegistryError(RuntimeError):
    """A registered model file does not match its recorded identity."""


@dataclass
class ModelRecord:
    modelId: str
    family: str
    role: str
    path: str
    sha256: str
    format: str
    trainingSource: str
    inputSize: int
    classes: dict[str, list[str]]
    rollbackTo: str | None
    profile: str

    @property
    def absolute_path(self) -> Path:
        return ROOT / self.path


class ModelRegistry:
    RegistryError = RegistryError

    def __init__(self, records: dict[str, ModelRecord], source_path: Path) -> None:
        self.records = records
        self.source_path = source_path

    @classmethod
    def load(cls, path: Path) -> "ModelRegistry":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        records: dict[str, ModelRecord] = {}
        for raw in document["models"]:
            record = _record_from_json(raw)
            _verify_file_hash(record)
            records[record.modelId] = record
        return cls(records, Path(path))

    def verify_active(self) -> None:
        active = [record for record in self.records.values() if record.role == "active"]
        if not active:
            raise RegistryError("no active model is registered")
        for record in active:
            _verify_file_hash(record)

    def active(self, family: str) -> ModelRecord:
        chosen = [
            record
            for record in self.records.values()
            if record.family == family and record.role == "active" and _is_primary(record)
        ]
        if len(chosen) != 1:
            raise RegistryError(f"{family} does not have one active model")
        return chosen[0]

    def activate(self, model_id: str) -> str:
        target = self._require(model_id)
        _verify_file_hash(target)
        if target.family == "quality":
            companions = [
                record
                for record in self.records.values()
                if record.family == "quality" and record.profile == target.profile
            ]
            if len(companions) != 2:
                raise RegistryError("quality models must switch as a pair")
            for record in companions:
                _verify_file_hash(record)
            for record in self.records.values():
                if record.family == "quality":
                    record.role = "active" if record.profile == target.profile else "rollback"
        else:
            for record in self.records.values():
                if record.family == target.family:
                    record.role = "active" if record.modelId == target.modelId else "rollback"
        self._persist_roles()
        return target.modelId

    def rollback(self, family: str) -> str:
        current = self.active(family)
        if not current.rollbackTo:
            raise RegistryError(f"{current.modelId} has no rollback target")
        return self.activate(current.rollbackTo)

    def _require(self, model_id: str) -> ModelRecord:
        try:
            return self.records[model_id]
        except KeyError as exc:
            raise RegistryError(f"unknown model {model_id}") from exc

    def _persist_roles(self) -> None:
        document = json.loads(self.source_path.read_text(encoding="utf-8"))
        roles = {record.modelId: record.role for record in self.records.values()}
        for raw in document["models"]:
            raw["role"] = roles[raw["modelId"]]
        self.source_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _record_from_json(raw: dict[str, Any]) -> ModelRecord:
    return ModelRecord(
        modelId=raw["modelId"],
        family=raw["family"],
        role=raw["role"],
        path=raw["path"],
        sha256=raw["sha256"],
        format=raw["format"],
        trainingSource=raw["trainingSource"],
        inputSize=int(raw["inputSize"]),
        classes=raw["classes"],
        rollbackTo=raw.get("rollbackTo"),
        profile=raw["profile"],
    )


def _is_primary(record: ModelRecord) -> bool:
    if record.family == "quality":
        return record.modelId.startswith("quality-")
    return True


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file_hash(record: ModelRecord) -> None:
    path = record.absolute_path
    if not path.is_file():
        raise RegistryError(f"model file is missing: {record.path}")
    actual = _file_sha256(path)
    if actual != record.sha256:
        raise RegistryError(f"SHA-256 mismatch for {record.modelId}")
