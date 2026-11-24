"""Audit Task 3 v2 provenance without training or evaluating a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FROZEN_FILES = {
    "manifestOfficial": "data/raw/konvid-1k/manifest-official.jsonl",
    "featuresOfficial": "data/raw/konvid-1k/features-official.jsonl",
    "qualityRidge": "data/raw/konvid-1k/models-official/quality_ridge.joblib",
    "eventRandomForest": "data/raw/konvid-1k/synthetic/full-frame-aligned-v3/models/event-rf.joblib",
    "manifestCorrectedV3": "data/raw/konvid-1k/synthetic/full-frame-aligned-v3/manifest-corrected.jsonl",
    "eventFeaturesCorrectedV3": "data/raw/konvid-1k/synthetic/full-frame-aligned-v3/event-features-corrected.jsonl",
    "qualityFrozenReportReproducible": "data/raw/konvid-1k/models-official/test_evaluation-reproducible.json",
    "qualityFrozenTestEvaluation": "data/raw/konvid-1k/models-official/test_evaluation.json",
    "eventFrozenReport": "data/raw/konvid-1k/synthetic/full-frame-aligned-v3/models/test-report.json",
    "eventFrozenResults": "data/raw/konvid-1k/synthetic/full-frame-aligned-v3/models/test-results.jsonl",
}

CANDIDATE_FILES = {
    "quality": (
        "data/raw/konvid-1k/candidate-v2/candidate_report.json",
        "data/raw/konvid-1k/candidate-v2/quality_candidate.joblib",
        "data/raw/konvid-1k/features-official.jsonl",
        "qualityCandidate",
    ),
    "event": (
        "data/raw/konvid-1k/candidate-v2/events/event_candidate_report.json",
        "data/raw/konvid-1k/candidate-v2/events/event-v2-candidate.joblib",
        "data/raw/konvid-1k/synthetic/full-frame-aligned-v3/event-features-corrected.jsonl",
        "eventCandidate",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_jsonl_splits(path: Path) -> dict[str, dict[str, int]]:
    rows: dict[str, int] = {}
    source_groups: dict[str, set[str]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON on line {line_number}") from exc
        split = str(row.get("split") or "")
        if not split:
            raise ValueError(f"{path}: missing split on line {line_number}")
        rows[split] = rows.get(split, 0) + 1
        group = str(row.get("sourceVideoId") or row.get("video_id") or row.get("videoId") or "")
        if group:
            source_groups.setdefault(split, set()).add(group)
    return {"rows": rows, "sourceGroups": {split: len(values) for split, values in source_groups.items()}}


def verify_candidate_report(report_path: Path, artifact_path: Path, input_path: Path | None = None) -> list[str]:
    errors: list[str] = []
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read candidate report: {exc}"]
    if report.get("testNotUsed") is not True:
        errors.append("testNotUsed must be true")
    input_hash = str(report.get("input", {}).get("sha256") or "")
    if len(input_hash) != 64:
        errors.append("candidate report input.sha256 is missing")
    elif input_path is not None and input_hash != sha256_file(input_path):
        errors.append("candidate report input sha256 does not match input file")
    model_hash = str(report.get("modelSha256") or "")
    if model_hash != sha256_file(artifact_path):
        errors.append("candidate report modelSha256 does not match artifact")
    if report.get("modelFile") != artifact_path.name:
        errors.append("candidate report modelFile does not match artifact")
    counts = report.get("splitCounts")
    if not isinstance(counts, dict) or not counts.get("train") or not counts.get("val"):
        errors.append("candidate report must contain non-empty train and val counts")
    if any(key in report for key in ("testMetrics", "testResults", "testPredictions")):
        errors.append("candidate report must not contain test evaluation payloads")
    return errors


def _check_hashes(root: Path, provenance: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    expected = provenance.get("frozenInputs", {})
    for key, relative in FROZEN_FILES.items():
        path = root / relative
        actual = sha256_file(path) if path.is_file() else None
        checks.append({"name": key, "path": relative, "expected": expected.get(key), "actual": actual, "ok": actual == expected.get(key)})
    return checks


def audit(root: Path, *, require_candidates: bool = True) -> dict[str, Any]:
    provenance_path = root / "docs/task3-v2-provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    report: dict[str, Any] = {"schemaVersion": "task3.v2-audit/1.0", "frozenHashes": _check_hashes(root, provenance), "errors": []}
    report["errors"].extend(check["name"] for check in report["frozenHashes"] if not check["ok"])

    official = root / FROZEN_FILES["manifestOfficial"]
    corrected = root / FROZEN_FILES["manifestCorrectedV3"]
    features = root / FROZEN_FILES["eventFeaturesCorrectedV3"]
    report["officialManifest"] = count_jsonl_splits(official)
    report["correctedManifest"] = count_jsonl_splits(corrected)
    report["eventFeatures"] = count_jsonl_splits(features)
    expected_splits = provenance.get("splitCounts", {})
    if report["officialManifest"]["sourceGroups"] != expected_splits.get("officialVideos", {}):
        report["errors"].append("official manifest split counts differ from provenance")
    if report["eventFeatures"]["rows"] != expected_splits.get("controlledEventWindows", {}):
        report["errors"].append("event feature split counts differ from provenance")

    candidate_checks = {}
    for name, (report_relative, artifact_relative, input_relative, provenance_key) in CANDIDATE_FILES.items():
        candidate_report = root / report_relative
        artifact = root / artifact_relative
        input_path = root / input_relative
        if not candidate_report.is_file() or not artifact.is_file() or not input_path.is_file():
            if require_candidates:
                report["errors"].append(f"missing {name} candidate report or artifact")
            candidate_checks[name] = {"ok": False, "report": report_relative, "artifact": artifact_relative, "input": input_relative}
            continue
        errors = verify_candidate_report(candidate_report, artifact, input_path)
        expected_artifact_hash = str(provenance.get("candidateArtifacts", {}).get(provenance_key) or "")
        actual_artifact_hash = sha256_file(artifact)
        if expected_artifact_hash != actual_artifact_hash:
            errors.append("candidate artifact hash does not match provenance")
        candidate_checks[name] = {"ok": not errors, "errors": errors, "report": report_relative, "artifact": artifact_relative, "input": input_relative}
        report["errors"].extend(f"{name}: {error}" for error in errors)
    report["candidateChecks"] = candidate_checks
    partials = list((root / "data/raw/konvid-1k").rglob("*.partial.mp4"))
    report["partialMp4Count"] = len(partials)
    if len(partials) < 15:
        report["errors"].append("partial mp4 count decreased below the audited baseline of 15")
    report["ok"] = not report["errors"]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-missing-candidates", action="store_true")
    args = parser.parse_args(argv)
    report = audit(args.root.resolve(), require_candidates=not args.allow_missing_candidates)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
