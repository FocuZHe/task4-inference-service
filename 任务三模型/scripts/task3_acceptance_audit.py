"""Read-only, requirement-level acceptance audit for Task 3.

The audit consumes existing reports only. It never trains, tunes, downloads,
or converts a research candidate into a production model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LIVE_VQC_OFFICIAL_URL_PREFIXES = (
    "https://www.colorado.edu/lab/live/",
    "https://live.ece.utexas.edu/research/LIVEVQC/",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _ordered_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _resolve_evidence_path(value: Any) -> Path | None:
    if isinstance(value, Path):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _verify_evidence_file(
    value: Any,
    expected_sha: Any,
    label: str,
    reasons: list[str],
) -> Path | None:
    path = _resolve_evidence_path(value)
    if path is None:
        reasons.append(f"{label} path is missing")
        return None
    if not path.is_file():
        reasons.append(f"{label} file does not exist")
        return path
    if not _is_sha256(expected_sha):
        reasons.append(f"{label} SHA-256 is invalid")
        return path
    try:
        actual_sha = _ordered_sha256([path])
    except OSError as exc:
        reasons.append(f"{label} file cannot be hashed: {exc}")
    else:
        if actual_sha.lower() != str(expected_sha).lower():
            reasons.append(f"{label} SHA-256 does not match the recorded value")
    return path


def _check_quality(report_path: Path, report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    srcc = metrics.get("srcc")
    mae = metrics.get("maeMos")
    sample_count = metrics.get("sampleCount")
    reasons: list[str] = []
    if report.get("dataset") != "KoNViD-1k":
        reasons.append("dataset must be KoNViD-1k")
    if report.get("split") != "test":
        reasons.append("quality report must be for test split")
    if report.get("testUsedForSelection") is not False:
        reasons.append("testUsedForSelection must be false")
    if report.get("inputFeatureProfileStatus") != "metadata_present":
        reasons.append("quality input feature profile is not verified")
    provenance = report.get("annotationProvenance") if isinstance(report.get("annotationProvenance"), dict) else {}
    if provenance.get("annotationSource") != "official KoNViD-1k metadata package":
        reasons.append("quality report does not identify the official KoNViD MOS source")
    for field, value in (
        ("packageSha256", provenance.get("packageSha256")),
        ("featuresSha256", report.get("featuresSha256")),
        ("modelSha256", report.get("modelSha256")),
    ):
        if not _is_sha256(value):
            reasons.append(f"quality report has no valid {field}")
    if sample_count != 180:
        reasons.append("quality test sample count must be 180")
    if not isinstance(srcc, (int, float)) or isinstance(srcc, bool) or not math.isfinite(float(srcc)) or not -1.0 <= float(srcc) <= 1.0 or float(srcc) < 0.60:
        reasons.append("quality test SRCC is below 0.60")
    if not isinstance(mae, (int, float)) or isinstance(mae, bool) or not math.isfinite(float(mae)) or float(mae) < 0:
        reasons.append("quality report MAE is missing or invalid")

    intervals = report.get("confidenceIntervals") if isinstance(report.get("confidenceIntervals"), dict) else {}
    if intervals.get("level") != 0.95 or not isinstance(intervals.get("method"), str) or not intervals.get("method", "").strip():
        reasons.append("quality report must contain a 95% bootstrap confidence interval")
    for field, minimum, maximum in (("srcc", -1.0, 1.0), ("maeMos", 0.0, math.inf)):
        interval = intervals.get(field)
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) for value in interval)
            or float(interval[0]) > float(interval[1])
            or float(interval[0]) < minimum
            or float(interval[1]) > maximum
        ):
            reasons.append(f"quality {field} confidence interval is invalid")
    bootstrap = report.get("bootstrap") if isinstance(report.get("bootstrap"), dict) else {}
    requested = bootstrap.get("requestedSamples")
    valid = bootstrap.get("validSrccSamples")
    if (
        not isinstance(requested, int)
        or isinstance(requested, bool)
        or requested < 100
        or not isinstance(valid, int)
        or isinstance(valid, bool)
        or not 1 <= valid <= requested
        or not isinstance(bootstrap.get("seed"), int)
        or isinstance(bootstrap.get("seed"), bool)
    ):
        reasons.append("quality report bootstrap run metadata is incomplete")

    selection_path = _verify_evidence_file(
        report.get("selectionRecordPath"),
        report.get("selectionRecordSha256"),
        "quality selection record",
        reasons,
    )
    if selection_path is not None and selection_path.is_file():
        try:
            selection = _read_json(selection_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            reasons.append(f"quality selection record cannot be read: {exc}")
        else:
            if selection.get("frozen") is not True or selection.get("testUsedForSelection") is not False:
                reasons.append("quality selection record must freeze the candidate without using test")
            for report_field, selection_field in (
                ("modelSha256", "modelSha256"),
                ("featuresSha256", "featuresSha256"),
                ("profile", "profile"),
            ):
                if str(report.get(report_field) or "").lower() != str(selection.get(selection_field) or "").lower():
                    reasons.append(f"quality selection record {selection_field} does not match the evaluation")
            embedded = report.get("selectionRecord") if isinstance(report.get("selectionRecord"), dict) else {}
            if embedded.get("path") != report.get("selectionRecordPath") or str(embedded.get("sha256") or "").lower() != str(report.get("selectionRecordSha256") or "").lower():
                reasons.append("quality evaluation selectionRecord reference is inconsistent")

            candidate_path = selection_path.parent / "candidate_report.json"
            try:
                candidate = _read_json(candidate_path)
            except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
                reasons.append(f"quality candidate report cannot be read: {exc}")
            else:
                if candidate.get("testNotUsed") is not True:
                    reasons.append("quality candidate report must state testNotUsed=true")
                if candidate.get("selectedCandidate") != selection.get("candidate"):
                    reasons.append("quality candidate report selection does not match frozen record")
                if candidate.get("featureProfile") != report.get("profile"):
                    reasons.append("quality candidate feature profile does not match evaluation")
                if candidate.get("inputFeatureProfile") != report.get("inputFeatureProfile"):
                    reasons.append("quality input feature profile does not match candidate report")
                input_info = candidate.get("input") if isinstance(candidate.get("input"), dict) else {}
                model_path = _resolve_evidence_path(str(selection_path.parent / str(candidate.get("modelFile") or "")))
                _verify_evidence_file(input_info.get("path"), report.get("featuresSha256"), "quality feature input", reasons)
                if str(input_info.get("sha256") or "").lower() != str(report.get("featuresSha256") or "").lower():
                    reasons.append("quality feature hash does not match candidate input provenance")
                _verify_evidence_file(model_path, report.get("modelSha256"), "quality selected model", reasons)
                if str(candidate.get("modelSha256") or "").lower() != str(report.get("modelSha256") or "").lower():
                    reasons.append("quality model hash does not match candidate report")

    manifest_path = _resolve_evidence_path(provenance.get("manifest"))
    package_name = provenance.get("packageFilename")
    if manifest_path is None or not manifest_path.is_file():
        reasons.append("official KoNViD manifest is missing")
    elif not isinstance(package_name, str) or not package_name or Path(package_name).name != package_name:
        reasons.append("official KoNViD metadata package filename is invalid")
    else:
        package_path = manifest_path.parent / package_name
        if not package_path.is_file():
            reasons.append("official KoNViD metadata package is missing")
        else:
            try:
                package_bytes = package_path.stat().st_size
                package_sha = _ordered_sha256([package_path])
            except OSError as exc:
                reasons.append(f"official KoNViD metadata package cannot be verified: {exc}")
            else:
                if package_bytes != provenance.get("packageBytes") or package_sha.lower() != str(provenance.get("packageSha256") or "").lower():
                    reasons.append("official KoNViD metadata package size or SHA-256 does not match")
    return {
        "ready": not reasons,
        "reasons": reasons,
        "metrics": {"srcc": srcc, "maeMos": metrics.get("maeMos"), "sampleCount": sample_count},
    }


def _check_formal_event_provenance(
    report_path: Path,
    report: dict[str, Any],
    provenance_paths: list[Path],
) -> list[str]:
    reasons: list[str] = []
    try:
        actual_report_sha = _ordered_sha256([report_path])
    except OSError as exc:
        return [f"formal event report cannot be hashed: {exc}"]

    candidates = list(provenance_paths)
    adjacent = report_path.with_suffix(".provenance.json")
    if adjacent not in candidates:
        candidates.append(adjacent)
    provenance: dict[str, Any] | None = None
    for candidate_path in candidates:
        try:
            candidate = _read_json(candidate_path)
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            continue
        candidate_freeze = candidate.get("candidateFreeze") if isinstance(candidate.get("candidateFreeze"), dict) else {}
        confirmation = candidate_freeze.get("testConfirmation") if isinstance(candidate_freeze.get("testConfirmation"), dict) else {}
        report_reference = confirmation.get("report") if isinstance(confirmation.get("report"), dict) else {}
        if str(report_reference.get("sha256") or "").lower() == actual_report_sha.lower():
            provenance = candidate
            break
    if provenance is None:
        return ["formal event test has no matching candidate provenance sidecar"]
    if provenance.get("schemaVersion") != "task3.formal-candidate-v13-provenance/1.0":
        reasons.append("formal event provenance schema is not recognized")
    selection_policy = provenance.get("selectionPolicy") if isinstance(provenance.get("selectionPolicy"), dict) else {}
    if selection_policy.get("testNotUsed") is not True or selection_policy.get("thresholdSelection") != "validation-only":
        reasons.append("formal event provenance must record train/validation-only selection")

    candidate_inputs = provenance.get("candidateInputs") if isinstance(provenance.get("candidateInputs"), dict) else {}
    candidate_features_path = candidate_inputs.get("eventFeaturesPath")
    candidate_features_sha = candidate_inputs.get("eventFeatures")
    _verify_evidence_file(candidate_features_path, candidate_features_sha, "formal event train/validation features", reasons)
    corrected_manifest = candidate_inputs.get("correctedManifest") if isinstance(candidate_inputs.get("correctedManifest"), dict) else {}
    if corrected_manifest.get("trainValOnly") is not True:
        reasons.append("formal event corrected manifest must be declared train/validation-only")
    _verify_evidence_file(corrected_manifest.get("path"), corrected_manifest.get("sha256"), "formal event corrected manifest", reasons)
    source_files = candidate_inputs.get("orderedSourceFiles")
    if not isinstance(source_files, list) or not source_files:
        reasons.append("formal event provenance must list ordered source feature files")
    else:
        for index, source in enumerate(source_files):
            if not isinstance(source, dict):
                reasons.append(f"formal event ordered source file {index} is invalid")
                continue
            _verify_evidence_file(source.get("path"), source.get("sha256"), f"formal event ordered source file {index}", reasons)

    candidate = provenance.get("candidateFreeze") if isinstance(provenance.get("candidateFreeze"), dict) else {}
    if candidate.get("candidateOnly") is not True or candidate.get("productionReady") is not False:
        reasons.append("formal event test candidate must remain candidate-only, not production-ready")
    if not isinstance(candidate.get("validationThresholds"), dict) or not candidate["validationThresholds"]:
        reasons.append("formal event provenance must record frozen validation thresholds")
    if not isinstance(candidate.get("timePostprocess"), dict) or not candidate["timePostprocess"]:
        reasons.append("formal event provenance must record frozen time post-processing")
    selection_report = candidate.get("selectionReport") if isinstance(candidate.get("selectionReport"), dict) else {}
    _verify_evidence_file(selection_report.get("path"), selection_report.get("sha256"), "formal event selection report", reasons)

    model = candidate.get("model") if isinstance(candidate.get("model"), dict) else {}
    _verify_evidence_file(model.get("path"), model.get("sha256"), "formal event frozen model", reasons)
    if str(model.get("sha256") or "").lower() != str(report.get("modelSha256") or "").lower():
        reasons.append("formal event model SHA-256 does not match the test report")

    confirmation = candidate.get("testConfirmation") if isinstance(candidate.get("testConfirmation"), dict) else {}
    report_reference = confirmation.get("report") if isinstance(confirmation.get("report"), dict) else {}
    report_reference_path = _verify_evidence_file(
        report_reference.get("path"), report_reference.get("sha256"), "formal event test report", reasons
    )
    if report_reference_path is not None and report_reference_path.resolve() != report_path.resolve():
        reasons.append("formal event provenance points to a different test report")
    if str(report_reference.get("sha256") or "").lower() != actual_report_sha.lower():
        reasons.append("formal event test report SHA-256 does not match provenance")

    test_input = confirmation.get("input") if isinstance(confirmation.get("input"), dict) else {}
    _verify_evidence_file(test_input.get("path"), test_input.get("sha256"), "formal event test input", reasons)
    if test_input.get("split") != "test":
        reasons.append("formal event provenance input split must be test")
    if str(test_input.get("sha256") or "").lower() != str(report.get("inputSha256") or "").lower():
        reasons.append("formal event test input SHA-256 does not match the test report")
    if test_input.get("videoCount") != report.get("testVideoCount") or test_input.get("windowRecordCount") != report.get("testWindowCount"):
        reasons.append("formal event test input counts do not match the test report")
    if confirmation.get("testUsedForSelection") is not False or confirmation.get("oneTimeOnly") is not True:
        reasons.append("formal event test must be recorded as a one-time, non-selection confirmation")

    metrics = confirmation.get("metrics") if isinstance(confirmation.get("metrics"), dict) else {}
    for field in ("macroF1", "freezeRecallAtLeast3s", "lowMotionNegativeFreezeFalseAlarmRate"):
        if metrics.get(field) != report.get(field):
            reasons.append(f"formal event provenance metric {field} does not match the test report")
    targets = confirmation.get("acceptanceTargets") if isinstance(confirmation.get("acceptanceTargets"), dict) else {}
    if (
        targets.get("macroF1Minimum") != 0.90
        or targets.get("freezeRecallAtLeast3sMinimum") != 0.95
        or targets.get("lowMotionNegativeFreezeFalseAlarmRateMaximum") != 0.05
    ):
        reasons.append("formal event provenance acceptance targets do not match Task 3")
    return reasons


def _check_formal_event(
    path: Path,
    report: dict[str, Any],
    provenance_paths: list[Path],
) -> dict[str, Any]:
    reasons: list[str] = []
    if report.get("schemaVersion") != "task3.controlled-event-test/1.0":
        reasons.append("formal event report schema is not recognized")
    if report.get("dataset") != "KoNViD-1k-controlled-distortions":
        reasons.append("formal event report dataset is not KoNViD-1k-controlled-distortions")
    if report.get("testUsedForSelection") is not False:
        reasons.append("formal event testUsedForSelection must be false")
    if report.get("validationUsedForSelection") is not True or report.get("trainUsedForFit") is not True:
        reasons.append("formal event report must record train fit and validation selection")
    if report.get("testVideoCount") != 2340:
        reasons.append("formal event test video count must be 2340")
    if report.get("testWindowCount") != 37466:
        reasons.append("formal event test window count must be 37466")
    metrics = {
        "macroF1": report.get("macroF1"),
        "freezeRecallAtLeast3s": report.get("freezeRecallAtLeast3s"),
        "lowMotionNegativeFreezeFalseAlarmRate": report.get("lowMotionNegativeFreezeFalseAlarmRate"),
    }
    failed_metrics: list[str] = []
    for name, value in metrics.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            failed_metrics.append(name)
    if not failed_metrics:
        if float(metrics["macroF1"]) < 0.90:
            failed_metrics.append("macroF1")
        if float(metrics["freezeRecallAtLeast3s"]) < 0.95:
            failed_metrics.append("freezeRecallAtLeast3s")
        if float(metrics["lowMotionNegativeFreezeFalseAlarmRate"]) > 0.05:
            failed_metrics.append("lowMotionNegativeFreezeFalseAlarmRate")
    if failed_metrics:
        reasons.append("formal controlled-event test fails one or more acceptance thresholds")
    reasons.extend(_check_formal_event_provenance(path, report, provenance_paths))
    return {
        "path": str(path),
        "reportKind": "formal-test",
        "ready": not reasons,
        "formalAcceptanceReady": not reasons,
        "researchGatePassed": False,
        "reasons": reasons,
        "failedMetrics": failed_metrics,
        "metrics": metrics,
        "modelSha256": report.get("modelSha256"),
    }


def _check_event(path: Path, report: dict[str, Any], provenance_paths: list[Path]) -> dict[str, Any]:
    if report.get("schemaVersion") == "task3.controlled-event-test/1.0":
        return _check_formal_event(path, report, provenance_paths)
    reasons: list[str] = []
    folds = report.get("outerFolds")
    if report.get("split") != "train-only":
        reasons.append("event report must be train-only")
    if report.get("testNotUsed") is not True:
        reasons.append("event report must state testNotUsed=true")
    if report.get("groupKey") != "sourceVideoId":
        reasons.append("event report must group outer folds by sourceVideoId")
    if report.get("outerSplitCount") != 5:
        reasons.append("event report must contain the registered five outer folds")
    if not isinstance(report.get("featureProfile"), str) or not str(report.get("featureProfile")).startswith("event-features-"):
        reasons.append("event report must identify an event feature profile")
    if report.get("seed") != 42:
        reasons.append("event report must use the registered seed=42")
    if report.get("innerSplitCount") != 3:
        reasons.append("event report must contain the registered three inner folds")
    if report.get("sourceGroupCount") != 840:
        reasons.append("event report must cover 840 source groups")
    if report.get("videoCount") != 10920:
        reasons.append("event report must cover 10920 train/val variant videos")
    input_info = report.get("input") if isinstance(report.get("input"), dict) else {}
    if not _is_sha256(input_info.get("sha256")):
        reasons.append("event report must contain input provenance with a valid SHA-256")
    if not isinstance(input_info.get("path"), list) or not input_info.get("path"):
        reasons.append("event report input provenance must list source feature files")
    else:
        feature_paths = [Path(str(value)) for value in input_info["path"]]
        try:
            actual_input_sha = _ordered_sha256(feature_paths)
        except (OSError, ValueError) as exc:
            reasons.append(f"event report input provenance cannot read feature files: {exc}")
        else:
            if actual_input_sha.lower() != str(input_info.get("sha256") or "").lower():
                reasons.append("event report input SHA-256 does not match feature files")
        provenance_path = input_info.get("provenancePath")
        if not isinstance(provenance_path, str) or not provenance_path.strip():
            reasons.append("event report input provenance must identify its registration file")
        else:
            try:
                registered = _read_json(Path(provenance_path))
                registered_sha = registered.get("candidateInputs", {}).get("eventFeatures")
            except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
                reasons.append(f"event report registration file cannot be read: {exc}")
            else:
                if str(registered_sha or "").lower() != str(input_info.get("sha256") or "").lower():
                    reasons.append("event report input SHA-256 does not match registered provenance")
    if not isinstance(folds, list) or len(folds) != 5:
        reasons.append("event report must contain exactly five outer folds")
        folds = []
    failed_folds: list[int] = []
    fold_results: list[dict[str, Any]] = []
    outer_scoring_sets: list[set[str]] = []
    for index, fold in enumerate(folds):
        if isinstance(fold, dict):
            source_groups = []
            for field in ("fitSources", "calibrationSources", "outerScoringSources", "outerTrainSources"):
                values = fold.get(field)
                if not isinstance(values, list) or not values or not all(isinstance(value, str) and value for value in values):
                    reasons.append(f"event fold {index} must list {field}")
                    continue
                if len(set(values)) != len(values):
                    reasons.append(f"event fold {index} has duplicate source groups")
                source_groups.append(set(values))
                if field == "outerScoringSources":
                    outer_scoring_sets.append(set(values))
            if len(source_groups) == 4 and any(left & right for position, left in enumerate(source_groups[:3]) for right in source_groups[position + 1:3]):
                reasons.append(f"event fold {index} has overlapping source groups")
            if len(source_groups) == 4 and source_groups[3] != source_groups[0] | source_groups[1]:
                reasons.append(f"event fold {index} outerTrainSources do not equal fit plus calibration")
        metrics = fold.get("outerMetrics") if isinstance(fold, dict) and isinstance(fold.get("outerMetrics"), dict) else {}
        values = (
            metrics.get("macroF1"),
            metrics.get("freezeRecallAtLeast3s"),
            metrics.get("lowMotionNegativeFreezeFalseAlarmRate"),
        )
        metric_values = {
            "macroF1": values[0],
            "freezeRecallAtLeast3s": values[1],
            "lowMotionNegativeFreezeFalseAlarmRate": values[2],
        }
        invalid_metrics = [name for name, value in metric_values.items() if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0]
        if invalid_metrics:
            failed_folds.append(index)
            fold_results.append({
                "fold": index,
                "passed": False,
                "metrics": metric_values,
                "failedMetrics": invalid_metrics,
            })
            continue
        failed_metrics = []
        if float(values[0]) < 0.90:
            failed_metrics.append("macroF1")
        if float(values[1]) < 0.95:
            failed_metrics.append("freezeRecallAtLeast3s")
        if float(values[2]) > 0.05:
            failed_metrics.append("lowMotionNegativeFreezeFalseAlarmRate")
        passed = not failed_metrics
        fold_results.append({
            "fold": index,
            "passed": passed,
            "metrics": metric_values,
            "failedMetrics": failed_metrics,
        })
        if not passed:
            failed_folds.append(index)
    if failed_folds:
        reasons.append("one or more outer folds fail the joint event thresholds")
    if len(outer_scoring_sets) == len(folds):
        if any(left & right for index, left in enumerate(outer_scoring_sets) for right in outer_scoring_sets[index + 1:]):
            reasons.append("outer scoring source groups overlap across folds")
        if len(set().union(*outer_scoring_sets)) != report.get("sourceGroupCount"):
            reasons.append("outer scoring source groups do not cover the registered source count")
    research_gate_passed = not reasons
    formal_reason = "train-only nested validation is research evidence, not a formal test acceptance report"
    return {
        "path": str(path),
        "reportKind": "train-only-research",
        "ready": False,
        "formalAcceptanceReady": False,
        "researchGatePassed": research_gate_passed,
        "reasons": [*reasons, formal_reason],
        "foldCount": len(folds),
        "failedFolds": failed_folds,
        "foldResults": fold_results,
        "outerSummary": report.get("outerSummary"),
    }


def _check_real(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"ready": False, "reasons": ["real annotation readiness report was not supplied"], "path": None}
    try:
        report = _read_json(path)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        return {"ready": False, "reasons": [f"cannot read real annotation readiness report: {exc}"], "path": str(path)}
    ready = report.get("status") == "ready" and report.get("ready") is True
    return {
        "ready": ready,
        "reasons": [] if ready else ["real annotation readiness report is not ready"],
        "path": str(path),
        "summary": report.get("summary"),
    }


def _check_live(path: Path | None, report_path: Path | None = None) -> dict[str, Any]:
    if path is None:
        return {"ready": False, "reasons": ["LIVE-VQC external manifest was not supplied"], "path": None}
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        return {"ready": False, "reasons": [f"cannot read LIVE-VQC manifest: {exc}"], "path": str(path)}
    reasons: list[str] = []
    if len(rows) != 585 or not all(isinstance(row, dict) for row in rows):
        reasons.append("LIVE-VQC manifest must contain exactly 585 records")
    record_ids: set[str] = set()
    video_ids: set[str] = set()
    video_hashes: dict[str, str] = {}
    package_hashes: set[str] = set()
    official_urls: set[str] = set()
    licenses: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("dataset") != "LIVE-VQC" or row.get("split") != "external_test" or row.get("externalTestOnly") is not True:
            reasons.append("LIVE-VQC rows must be explicit external_test records")
            continue
        if row.get("trainingAllowed") is not False or row.get("selectionAllowed") is not False:
            reasons.append("LIVE-VQC records must forbid training and selection")
            continue
        record_id = str(row.get("recordId") or "").strip()
        if not record_id:
            reasons.append("LIVE-VQC records require recordId")
            continue
        if record_id in record_ids:
            reasons.append("LIVE-VQC recordId values must be unique")
            continue
        record_ids.add(record_id)
        video_id = str(row.get("videoId") or row.get("video_id") or "").strip()
        video_sha = str(row.get("videoSha256") or "").strip().lower()
        if not video_id:
            reasons.append("LIVE-VQC records require videoId")
            continue
        if video_id in video_ids:
            reasons.append("LIVE-VQC videoId values must be unique")
            continue
        video_ids.add(video_id)
        if not _is_sha256(video_sha):
            reasons.append("LIVE-VQC records require a valid videoSha256")
            continue
        if video_id in video_hashes and video_hashes[video_id] != video_sha:
            reasons.append("LIVE-VQC videoSha256 must be consistent within a video")
            continue
        video_hashes[video_id] = video_sha
        package_sha = str(row.get("packageSha256") or "").strip().lower()
        if not _is_sha256(package_sha):
            reasons.append("LIVE-VQC records require a valid packageSha256")
            continue
        package_hashes.add(package_sha)
        official_urls.add(str(row.get("officialUrl") or "").strip())
        licenses.add(str(row.get("license") or "").strip())
    if not reasons and len(record_ids) != len(rows):
        reasons.append("LIVE-VQC recordId values must be unique")
    if not reasons and len(package_hashes) != 1:
        reasons.append("LIVE-VQC packageSha256 must be consistent across records")
    if not reasons and len(official_urls) != 1:
        reasons.append("LIVE-VQC officialUrl must be consistent across records")
    if not reasons and (len(licenses) != 1 or not next(iter(licenses), "")):
        reasons.append("LIVE-VQC license must be present and consistent across records")
    if not reasons and not any(next(iter(official_urls), "").startswith(prefix) for prefix in LIVE_VQC_OFFICIAL_URL_PREFIXES):
        reasons.append("LIVE-VQC officialUrl must use a recognized official source")
    if not reasons and report_path is None:
        reasons.append("LIVE-VQC external evaluation report was not supplied")
    external_report: dict[str, Any] | None = None
    if not reasons and report_path is not None:
        try:
            external_report = _read_json(report_path)
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
            reasons.append(f"cannot read LIVE-VQC external evaluation report: {exc}")
        if external_report is not None:
            if external_report.get("schemaVersion") != "task3.quality-external-evaluation/1.0":
                reasons.append("LIVE-VQC external report has an unexpected schema")
            if external_report.get("dataset") != "LIVE-VQC" or external_report.get("split") != "external_test":
                reasons.append("LIVE-VQC external report must identify external_test")
            if external_report.get("externalTestOnly") is not True:
                reasons.append("LIVE-VQC external report must state externalTestOnly=true")
            if external_report.get("trainingAllowed") is not False or external_report.get("selectionAllowed") is not False:
                reasons.append("LIVE-VQC external report must forbid training and selection")
            if external_report.get("testUsedForSelection") is not False:
                reasons.append("LIVE-VQC external report must state testUsedForSelection=false")
            metrics = external_report.get("metrics") if isinstance(external_report.get("metrics"), dict) else {}
            if metrics.get("sampleCount") != 585 or not isinstance(metrics.get("srcc"), (int, float)) or isinstance(metrics.get("srcc"), bool) or not math.isfinite(float(metrics.get("srcc"))):
                reasons.append("LIVE-VQC external report must contain 585 finite-SRCC samples")
            provenance = external_report.get("annotationProvenance") if isinstance(external_report.get("annotationProvenance"), dict) else {}
            manifest_package = str(rows[0].get("packageSha256") or "").lower() if rows else ""
            if provenance.get("packageSha256") != manifest_package:
                reasons.append("LIVE-VQC external report packageSha256 does not match manifest")
            manifest_url = next(iter(official_urls), "")
            if provenance.get("officialUrl") != manifest_url:
                reasons.append("LIVE-VQC external report officialUrl does not match manifest")
            manifest_license = next(iter(licenses), "")
            if str(provenance.get("license") or "").strip() != manifest_license:
                reasons.append("LIVE-VQC external report license does not match manifest")
    return {
        "ready": not reasons,
        "reasons": reasons,
        "path": str(path),
        "reportPath": str(report_path) if report_path else None,
        "recordCount": len(rows),
    }


def _parse_utc_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _check_live_chain(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "ready": False,
            "reasons": ["live-chain telemetry report was not supplied"],
            "path": None,
            "counts": {"frames": 0, "networkSamples": 0, "heartbeats": 0},
        }
    try:
        report = _read_json(path)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "ready": False,
            "reasons": [f"cannot read live-chain telemetry report: {exc}"],
            "path": str(path),
            "counts": {"frames": 0, "networkSamples": 0, "heartbeats": 0},
        }

    reasons: list[str] = []
    if report.get("schemaVersion") != "task3.live-chain-telemetry/1.0":
        reasons.append("live-chain telemetry schema is not recognized")
    if report.get("captureMode") != "live":
        reasons.append("live-chain captureMode must be live")
    if report.get("isSynthetic") is not False:
        reasons.append("synthetic live-chain telemetry is not acceptable evidence")

    session_id = report.get("sessionId")
    device_id = report.get("deviceId")
    if not isinstance(session_id, str) or not session_id.strip():
        reasons.append("live-chain report must identify a sessionId")
    if not isinstance(device_id, str) or not device_id.strip():
        reasons.append("live-chain report must identify a deviceId")
    for field in ("sourceProtocol", "sourceUriRedacted"):
        if not isinstance(report.get(field), str) or not report[field].strip():
            reasons.append(f"live-chain report must identify {field}")
    if not _is_sha256(report.get("sourceUriSha256")):
        reasons.append("live-chain report must contain a valid sourceUriSha256")

    started_at = _parse_utc_timestamp(report.get("startedAt"))
    ended_at = _parse_utc_timestamp(report.get("endedAt"))
    if started_at is None:
        reasons.append("live-chain startedAt must be a timezone-aware ISO-8601 timestamp")
    if ended_at is None:
        reasons.append("live-chain endedAt must be a timezone-aware ISO-8601 timestamp")
    if started_at is not None and ended_at is not None and ended_at < started_at:
        reasons.append("live-chain endedAt must not precede startedAt")

    telemetry = report.get("telemetry") if isinstance(report.get("telemetry"), dict) else {}
    telemetry_path_value = telemetry.get("path")
    telemetry_sha = telemetry.get("sha256")
    declared_count = telemetry.get("recordCount")
    if not isinstance(telemetry_path_value, str) or not telemetry_path_value.strip():
        reasons.append("live-chain report must identify the raw telemetry JSONL path")
        telemetry_path = None
    else:
        telemetry_path = Path(telemetry_path_value)
        if not telemetry_path.is_absolute():
            telemetry_path = path.parent / telemetry_path
    if not _is_sha256(telemetry_sha):
        reasons.append("live-chain report must contain a valid telemetry SHA-256")
    if not isinstance(declared_count, int) or isinstance(declared_count, bool) or declared_count < 1:
        reasons.append("live-chain report must contain a positive telemetry recordCount")

    rows: list[dict[str, Any]] = []
    if telemetry_path is not None:
        try:
            raw = telemetry_path.read_bytes()
            actual_sha = hashlib.sha256(raw).hexdigest()
            if _is_sha256(telemetry_sha) and actual_sha.lower() != str(telemetry_sha).lower():
                reasons.append("live-chain telemetry SHA-256 does not match raw JSONL")
            for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    reasons.append(f"live-chain telemetry JSONL line {line_number} is invalid: {exc.msg}")
                    continue
                if not isinstance(row, dict):
                    reasons.append(f"live-chain telemetry JSONL line {line_number} must be an object")
                    continue
                rows.append(row)
        except (FileNotFoundError, OSError, UnicodeDecodeError) as exc:
            reasons.append(f"cannot verify live-chain telemetry JSONL: {exc}")
    if isinstance(declared_count, int) and not isinstance(declared_count, bool) and declared_count != len(rows):
        reasons.append("live-chain telemetry recordCount does not match JSONL records")

    counts = {"frames": 0, "networkSamples": 0, "heartbeats": 0}
    frame_sequences: list[int] = []
    frame_pts: list[float] = []
    for row_index, row in enumerate(rows, start=1):
        if row.get("sessionId") != session_id:
            reasons.append(f"live-chain telemetry record {row_index} has a different sessionId")
        event_type = row.get("eventType")
        if event_type == "frame":
            counts["frames"] += 1
            sequence = row.get("sequence")
            pts = row.get("ptsS")
            if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
                reasons.append(f"live-chain frame record {row_index} has an invalid sequence")
            else:
                frame_sequences.append(sequence)
            if not isinstance(pts, (int, float)) or isinstance(pts, bool) or not math.isfinite(float(pts)) or float(pts) < 0:
                reasons.append(f"live-chain frame record {row_index} has an invalid ptsS")
            else:
                frame_pts.append(float(pts))
            if not isinstance(row.get("decodeStatus"), str) or not row["decodeStatus"].strip():
                reasons.append(f"live-chain frame record {row_index} must identify decodeStatus")
            event_time = _parse_utc_timestamp(row.get("arrivedAt"))
            if event_time is None:
                reasons.append(f"live-chain frame record {row_index} has an invalid arrivedAt timestamp")
        elif event_type == "network":
            counts["networkSamples"] += 1
            if not isinstance(row.get("connected"), bool):
                reasons.append(f"live-chain network record {row_index} must contain a boolean connected state")
            rtt = row.get("rttMs")
            if rtt is not None and (
                not isinstance(rtt, (int, float))
                or isinstance(rtt, bool)
                or not math.isfinite(float(rtt))
                or float(rtt) < 0
            ):
                reasons.append(f"live-chain network record {row_index} has an invalid rttMs")
            lost = row.get("packetsLost")
            if lost is not None and (not isinstance(lost, int) or isinstance(lost, bool) or lost < 0):
                reasons.append(f"live-chain network record {row_index} has an invalid packetsLost")
            event_time = _parse_utc_timestamp(row.get("sampledAt"))
            if event_time is None:
                reasons.append(f"live-chain network record {row_index} has an invalid sampledAt timestamp")
        elif event_type == "heartbeat":
            counts["heartbeats"] += 1
            if row.get("deviceId") != device_id:
                reasons.append(f"live-chain heartbeat record {row_index} has a different deviceId")
            if not isinstance(row.get("status"), str) or not row["status"].strip():
                reasons.append(f"live-chain heartbeat record {row_index} must identify status")
            event_time = _parse_utc_timestamp(row.get("sampledAt"))
            if event_time is None:
                reasons.append(f"live-chain heartbeat record {row_index} has an invalid sampledAt timestamp")
        else:
            reasons.append(f"live-chain telemetry record {row_index} has an unsupported eventType")
            event_time = None
        if event_time is not None and started_at is not None and ended_at is not None:
            if not started_at <= event_time <= ended_at:
                reasons.append(f"live-chain telemetry record {row_index} falls outside the session time range")

    if counts["frames"] < 2:
        reasons.append("live-chain telemetry needs at least two frames to verify PTS progression")
    if counts["networkSamples"] < 1:
        reasons.append("live-chain telemetry is missing network state samples")
    if counts["heartbeats"] < 1:
        reasons.append("live-chain telemetry is missing device heartbeat samples")
    if frame_sequences and (len(set(frame_sequences)) != len(frame_sequences) or any(a >= b for a, b in zip(frame_sequences, frame_sequences[1:]))):
        reasons.append("live-chain frame sequence values must be unique and strictly increasing")
    if len(frame_pts) >= 2 and any(a > b for a, b in zip(frame_pts, frame_pts[1:])):
        reasons.append("live-chain frame PTS must not move backwards within a session")
    if len(frame_pts) >= 2 and frame_pts[-1] <= frame_pts[0]:
        reasons.append("live-chain frame PTS must advance during the session")

    return {
        "ready": not reasons,
        "reasons": reasons,
        "path": str(path),
        "telemetryPath": str(telemetry_path) if telemetry_path is not None else None,
        "counts": counts,
        "verificationScope": "raw-file hash and structural consistency; capture authenticity is not independently established",
    }


def _check_performance(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"ready": False, "reasons": ["performance benchmark report was not supplied"], "path": None}
    try:
        report = _read_json(path)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        return {"ready": False, "reasons": [f"cannot read performance benchmark report: {exc}"], "path": str(path)}

    reasons: list[str] = []
    if report.get("schemaVersion") != "task3.performance-benchmark/1.0":
        reasons.append("performance benchmark schema is not recognized")
    fixture = report.get("fixture") if isinstance(report.get("fixture"), dict) else {}
    if fixture.get("scope") != "performance-only" or fixture.get("trainingAllowed") is not False:
        reasons.append("performance fixture must be marked performance-only and excluded from training")
    if fixture.get("sourceDataset") != "KoNViD-1k" or fixture.get("sourceSplit") != "train":
        reasons.append("performance fixture must identify the official KoNViD train source")
    expected_fixture = {"width": 1920, "height": 1080, "fps": 25, "durationSeconds": 600, "frameCount": 15000}
    for field, expected in expected_fixture.items():
        if fixture.get(field) != expected:
            reasons.append(f"performance fixture must be {expected_fixture['width']}x{expected_fixture['height']} at 25 fps for 600 seconds" if field == "width" else f"performance fixture {field} is not {expected}")
            break
    if not _is_sha256(fixture.get("sha256")):
        reasons.append("performance fixture must contain a valid SHA-256")
    fixture_path = fixture.get("path")
    if fixture_path:
        actual_path = Path(str(fixture_path))
        if not actual_path.is_file():
            reasons.append("performance fixture path does not exist")
        else:
            try:
                if _ordered_sha256([actual_path]).lower() != str(fixture.get("sha256")).lower():
                    reasons.append("performance fixture SHA-256 does not match the recorded file")
            except OSError as exc:
                reasons.append(f"performance fixture cannot be hashed: {exc}")
    source_manifest_path = fixture.get("sourceManifestPath")
    source_manifest_sha = fixture.get("sourceManifestSha256")
    source_video_id = fixture.get("sourceVideoId")
    source_video_sha = fixture.get("sourceVideoSha256")
    if not isinstance(source_manifest_path, str) or not _is_sha256(source_manifest_sha):
        reasons.append("performance report must identify the official source manifest and its SHA-256")
    else:
        manifest_file = Path(source_manifest_path)
        try:
            if not manifest_file.is_file() or _ordered_sha256([manifest_file]).lower() != str(source_manifest_sha).lower():
                reasons.append("performance source manifest is missing or its SHA-256 does not match")
            else:
                source_rows = [json.loads(line) for line in manifest_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                source_row = next((row for row in source_rows if row.get("video_id", row.get("videoId")) == source_video_id), None)
                if not isinstance(source_row, dict) or source_row.get("split") != "train":
                    reasons.append("performance source video is not registered in the official train split")
                else:
                    registered_source_sha = source_row.get("sha256", source_row.get("videoSha256"))
                    source_file = Path(str(source_row.get("path") or ""))
                    if str(registered_source_sha or "").lower() != str(source_video_sha or "").lower():
                        reasons.append("performance source video SHA-256 does not match the official manifest")
                    if not _is_sha256(source_video_sha) or not source_file.is_file() or _ordered_sha256([source_file]).lower() != str(source_video_sha).lower():
                        reasons.append("performance official source video is missing or its SHA-256 does not match")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            reasons.append(f"performance source manifest cannot be verified: {exc}")

    hardware = report.get("hardware") if isinstance(report.get("hardware"), dict) else {}
    if not isinstance(hardware.get("cpu"), str) or not hardware.get("cpu", "").strip():
        reasons.append("performance report must identify the CPU")
    for field in ("logicalThreads", "physicalMemoryBytes"):
        value = hardware.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            reasons.append(f"performance report must contain a positive integer {field}")

    model = report.get("model") if isinstance(report.get("model"), dict) else {}
    for field in ("qualitySha256", "eventSha256"):
        if not _is_sha256(model.get(field)):
            reasons.append(f"performance report must contain a valid model {field}")
    for path_field, hash_field in (("qualityPath", "qualitySha256"), ("eventPath", "eventSha256")):
        model_path = model.get(path_field)
        if not isinstance(model_path, str):
            reasons.append(f"performance report must identify {path_field}")
        else:
            model_file = Path(model_path)
            try:
                if not model_file.is_file() or _ordered_sha256([model_file]).lower() != str(model.get(hash_field) or "").lower():
                    reasons.append(f"performance {path_field} is missing or its SHA-256 does not match")
            except OSError as exc:
                reasons.append(f"performance {path_field} cannot be verified: {exc}")
    methodology = report.get("methodology") if isinstance(report.get("methodology"), dict) else {}
    if not isinstance(methodology.get("memoryMethod"), str) or not methodology.get("memoryMethod", "").strip():
        reasons.append("performance report must identify its peak-memory measurement method")
    runs = report.get("runs")
    run_map: dict[str, dict[str, Any]] = {}
    if not isinstance(runs, list):
        reasons.append("performance report must contain cold and warmed runs")
    else:
        for run in runs:
            if not isinstance(run, dict) or run.get("kind") not in {"cold", "warmed"}:
                reasons.append("performance runs must be named cold or warmed")
                continue
            kind = str(run["kind"])
            if kind in run_map:
                reasons.append("performance runs must contain each kind exactly once")
                continue
            value = run.get("processingTimeMs")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
                reasons.append(f"performance {kind} run time is invalid")
            elif float(run.get("wallTimeMs", value)) > 600_000:
                reasons.append(f"performance {kind} run exceeds 10 minute limit")
            run_map[kind] = run
    if set(run_map) != {"cold", "warmed"}:
        reasons.append("performance report must contain exactly one cold and one warmed run")
    peak_rss = report.get("peakRssBytes")
    if not isinstance(peak_rss, int) or isinstance(peak_rss, bool) or peak_rss <= 0:
        reasons.append("performance peakRssBytes must be a positive integer")
    elif peak_rss > 2 * 1024**3:
        reasons.append("performance peak RSS exceeds 2 GiB")
    return {
        "ready": not reasons,
        "reasons": reasons,
        "path": str(path),
        "metrics": {
            "coldProcessingTimeMs": run_map.get("cold", {}).get("processingTimeMs"),
            "warmedProcessingTimeMs": run_map.get("warmed", {}).get("processingTimeMs"),
            "peakRssBytes": peak_rss,
            "qualityModelSha256": model.get("qualitySha256"),
            "eventModelSha256": model.get("eventSha256"),
        },
    }


def audit_task3_acceptance(
    quality_report_path: Path,
    event_report_paths: list[Path],
    *,
    event_provenance_paths: list[Path] | None = None,
    real_annotation_report: Path | None = None,
    live_vqc_manifest: Path | None = None,
    live_vqc_report: Path | None = None,
    live_chain_report: Path | None = None,
    performance_report_path: Path | None = None,
) -> dict[str, Any]:
    try:
        quality_report = _read_json(quality_report_path)
        quality = _check_quality(quality_report_path, quality_report)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        quality = {"ready": False, "reasons": [f"cannot read quality report: {exc}"], "path": str(quality_report_path)}
    events = []
    for path in event_report_paths:
        try:
            events.append(_check_event(path, _read_json(path), event_provenance_paths or []))
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
            events.append({"path": str(path), "ready": False, "reasons": [f"cannot read event report: {exc}"]})
    formal_events = [item for item in events if item.get("reportKind") == "formal-test"]
    research_events = [item for item in events if item.get("reportKind") == "train-only-research"]
    # Historical formal runs remain visible, but a failed baseline must not veto a
    # later pre-registered candidate that passes its one-time frozen test.
    formal_ready = any(item.get("formalAcceptanceReady", False) for item in formal_events)
    research_passed = bool(research_events) and all(item.get("researchGatePassed", False) for item in research_events)
    controlled = {
        "ready": formal_ready,
        "formalAcceptanceReady": formal_ready,
        "researchGatePassed": research_passed,
        "reports": events,
        "reasons": (
            []
            if formal_ready
            else [
                "no supplied formal controlled-event test report passes all acceptance thresholds",
                *sorted({reason for item in events for reason in item.get("reasons", [])}),
            ]
        ),
    }
    real = _check_real(real_annotation_report)
    live = _check_live(live_vqc_manifest, live_vqc_report)
    live_chain = _check_live_chain(live_chain_report)
    performance = _check_performance(performance_report_path)
    quality_model_sha = quality_report.get("modelSha256") if isinstance(quality_report, dict) else None
    if performance.get("ready") and str(performance.get("metrics", {}).get("qualityModelSha256", "")).lower() != str(quality_model_sha or "").lower():
        performance["ready"] = False
        performance.setdefault("reasons", []).append("performance quality model does not match the quality test model")
    passing_formal_model_hashes = {
        str(item.get("modelSha256") or "").lower()
        for item in events
        if item.get("formalAcceptanceReady") and _is_sha256(item.get("modelSha256"))
    }
    performance_event_hash = str(performance.get("metrics", {}).get("eventModelSha256", "")).lower()
    if performance.get("ready") and (not passing_formal_model_hashes or performance_event_hash not in passing_formal_model_hashes):
        performance["ready"] = False
        performance.setdefault("reasons", []).append("performance event model does not match a passing formal controlled-event test")
    checks = {
        "qualityRegression": quality,
        "controlledAnomaly": controlled,
        "realVideoAnnotations": real,
        "liveVqcExternal": live,
        "liveChainTelemetry": live_chain,
        "performance": performance,
    }
    return {
        "schemaVersion": "task3.acceptance-audit/1.0",
        "ready": all(item["ready"] for item in checks.values()),
        "status": "ready" if all(item["ready"] for item in checks.values()) else "not_ready",
        "checks": checks,
        "limitations": [
            "该审计只验证已有报告和门控证据，不训练模型、不选择阈值、不下载数据。",
            "受控合成事件不能替代真实设备故障标注；LIVE-VQC 只能作为冻结后的 external_test。",
            "直播链路门仅验证报告结构与原始遥测哈希，不独立证明采集设备、现场环境或记录真实性。",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--event-report", type=Path, action="append", required=True)
    parser.add_argument("--event-provenance", type=Path, action="append", default=[])
    parser.add_argument("--real-annotation-report", type=Path)
    parser.add_argument("--live-vqc-manifest", type=Path)
    parser.add_argument("--live-vqc-report", type=Path)
    parser.add_argument("--live-chain-report", type=Path)
    parser.add_argument("--performance-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    report = audit_task3_acceptance(
        args.quality_report,
        args.event_report,
        event_provenance_paths=args.event_provenance,
        real_annotation_report=args.real_annotation_report,
        live_vqc_manifest=args.live_vqc_manifest,
        live_vqc_report=args.live_vqc_report,
        live_chain_report=args.live_chain_report,
        performance_report_path=args.performance_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
