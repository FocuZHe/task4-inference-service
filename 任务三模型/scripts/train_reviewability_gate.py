"""Train research-only reviewability scores from adjudicated human labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scripts.extract_features import extract_video_features
    from scripts.reviewability_annotations import resolve_adjudicated_labels
    from scripts.train_quality import FRAME_STATISTICS_COLUMNS
except ModuleNotFoundError:
    from extract_features import extract_video_features
    from reviewability_annotations import resolve_adjudicated_labels
    from train_quality import FRAME_STATISTICS_COLUMNS


SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
MIN_SOURCE_GROUPS = 10
MIN_HUMAN_VIDEOS = 100
FEATURE_COLUMNS = tuple(FRAME_STATISTICS_COLUMNS)
FEATURE_PROFILE = "quality-features-v1-native-resolution"
EXTRACTION_CONFIG = {
    "max_long_side": 0,
    "sample_fps": 2.0,
    "window_s": 10.0,
    "step_s": 5.0,
}
CANDIDATE_NAMES = ("logistic_c_1", "random_forest_leaf_5")
FORBIDDEN_FEATURE_KEYS = {
    "mos", "eventLabel", "predictedLabel", "predictedConfidence", "modelScore", "eventProbability", "eventProbabilities",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            rows.append(row)
    return rows


def _valid_sha(value: object) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _reject_disallowed_provenance(value: object, location: str) -> None:
    """Screen every nested provenance object before decoding, fitting or export."""
    if isinstance(value, dict):
        for key, item in value.items():
            name = str(key).replace("_", "").lower()
            text = str(item).strip().lower()
            if name == "split" and text in {"test", "external_test"}:
                raise ValueError(f"{location}: forbidden split {text}")
            if name == "issynthetic" and item is not False:
                raise ValueError(f"{location}: synthetic or ambiguous synthetic provenance is forbidden")
            if name in {"source", "dataset", "datarole"} and "synthetic" in text:
                raise ValueError(f"{location}: synthetic provenance is forbidden")
            if name in {field.replace("_", "").lower() for field in FORBIDDEN_FEATURE_KEYS}:
                raise ValueError(f"{location}: forbidden supervision field {key}")
            if name in {"source", "dataset", "sourceurl", "officialurl"}:
                normalized = text.replace("-", "").replace("_", "").replace(" ", "")
                if "konvid" in normalized or "livevqc" in normalized:
                    raise ValueError(f"{location}: forbidden quality dataset provenance")
            _reject_disallowed_provenance(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_disallowed_provenance(item, f"{location}[{index}]")
    elif isinstance(value, str) and "ENGINEERING_ONLY" in value.upper():
        raise ValueError(f"{location}: engineering-only input is forbidden")


def _validate_label_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    if not rows:
        raise ValueError("no adjudicated labels")
    seen: set[str] = set()
    uncertain_count = 0
    accepted: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        video_id = str(row.get("videoId") or "").strip()
        if not video_id or video_id in seen:
            raise ValueError(f"duplicate or missing videoId at label row {index}")
        seen.add(video_id)
        for field in ("resolutionMethod", "label"):
            if not isinstance(row.get(field), str):
                raise ValueError(f"{video_id}: {field} must be a string")
        if "split" in row and not isinstance(row["split"], str):
            raise ValueError(f"{video_id}: split must be a string")
        if FORBIDDEN_FEATURE_KEYS & set(row) or row.get("split") in {"test", "external_test"}:
            raise ValueError(f"{video_id}: MOS/event/test-set labels are forbidden")
        if not _valid_sha(row.get("videoSha256")):
            raise ValueError(f"{video_id}: invalid videoSha256")
        if not str(row.get("sourceUrl") or "").strip() or not str(row.get("license") or "").strip():
            raise ValueError(f"{video_id}: sourceUrl/license provenance is required")
        resolution_method, resolution_id = row.get("resolutionMethod"), row.get("resolutionId")
        if resolution_method not in {"dual_agreement", "third_party_arbitration"} or not _valid_sha(resolution_id):
            raise ValueError(f"{video_id}: invalid resolution provenance")
        provenance = row.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"{video_id}: provenance must be an object")
        annotators = provenance.get("annotatorIds")
        row_hashes = provenance.get("annotatorRowHashes")
        if (not isinstance(annotators, list) or len(annotators) != 2
                or not all(isinstance(value, str) and value.strip() for value in annotators)
                or len({value.strip() for value in annotators}) != 2):
            raise ValueError(f"{video_id}: two distinct annotator IDs are required")
        if not isinstance(row_hashes, list) or len(row_hashes) != 2 or not all(_valid_sha(digest) for digest in row_hashes):
            raise ValueError(f"{video_id}: two annotation row hashes are required")
        arbitration_hash = provenance.get("arbitrationRecordHash")
        if resolution_method == "third_party_arbitration" and (not _valid_sha(arbitration_hash) or arbitration_hash != resolution_id):
            raise ValueError(f"{video_id}: arbitration record hash is required")
        if resolution_method == "dual_agreement" and arbitration_hash is not None:
            raise ValueError(f"{video_id}: dual agreement must not claim an arbitration record")
        label = row.get("label")
        if label == "uncertain":
            uncertain_count += 1
            continue
        if label not in {"reviewable", "not_reviewable"}:
            raise ValueError(f"{video_id}: unsupported human reviewability label")
        accepted.append(row)
    if not accepted:
        raise ValueError("no binary adjudicated labels")
    return accepted, uncertain_count


def validate_source_inputs(
    final_rows: list[dict[str, Any]],
    source_paths: dict[str, Path],
    expected_hashes: dict[str, str],
) -> dict[str, Any]:
    """Recompute final labels from raw human sidecars and verify manifest bytes."""
    required = {"manifest", "annotator_a", "annotator_b", "adjudication"}
    if set(source_paths) != required or not required <= set(expected_hashes):
        raise ValueError("source sidecars and their trusted expected SHA-256 values are required")
    actual_hashes: dict[str, str] = {}
    for role in sorted(required):
        path = Path(source_paths[role])
        if not path.is_file() or not _valid_sha(expected_hashes[role]):
            raise ValueError(f"source sidecar is absent or has no trusted SHA-256: {role}")
        actual_hashes[role] = _sha256(path)
        if actual_hashes[role] != str(expected_hashes[role]).lower():
            raise ValueError(f"source sidecar SHA-256 mismatch: {role}")
    payloads = {role: _read_jsonl(Path(source_paths[role])) for role in sorted(required)}
    for role, payload in payloads.items():
        _reject_disallowed_provenance(payload, role)
    _reject_disallowed_provenance(final_rows, "final_labels")
    verified = _verify_source_payloads(final_rows, payloads, Path(source_paths["manifest"]))
    return {"inputHashes": actual_hashes, **verified}


def _verify_source_payloads(final_rows: list[dict[str, Any]], payloads: dict[str, list[dict[str, Any]]], manifest_path: Path):
    """Check resolution arithmetic and byte bindings; caller enforces admission."""
    recomputed = resolve_adjudicated_labels(payloads["annotator_a"], payloads["annotator_b"], payloads["adjudication"])
    supplied = sorted(final_rows, key=lambda row: str(row.get("videoId") or ""))
    if _canonical_sha256(supplied) != _canonical_sha256(recomputed):
        raise ValueError("final labels do not match recomputed human resolution provenance")
    manifests = payloads["manifest"]
    by_id: dict[str, dict[str, Any]] = {}
    for record in manifests:
        video_id = str(record.get("videoId") or "").strip()
        if not video_id or video_id in by_id:
            raise ValueError("source manifest videoIds must be unique and non-empty")
        source_text = " ".join(str(record.get(field) or "").lower() for field in ("source", "dataset"))
        if (record.get("split") in {"test", "external_test"} or record.get("isSynthetic") is True
                or any(name in source_text for name in ("konvid", "live-vqc", "live_vqc", "livevqc"))):
            raise ValueError("source manifest cannot contain test/external_test or synthetic data")
        by_id[video_id] = record
    if set(by_id) != {str(row["videoId"]) for row in final_rows}:
        raise ValueError("source manifest and final labels must cover exactly the same videos")
    for final in final_rows:
        video_id = str(final["videoId"])
        record = by_id[video_id]
        for field in ("videoSha256", "sourceUrl", "license"):
            if str(record.get(field) or "").strip() != str(final.get(field) or "").strip():
                raise ValueError(f"{video_id}: source manifest and final label {field} differ")
        duration = record.get("durationS", record.get("videoDurationS"))
        if isinstance(duration, bool) or not math.isfinite(float(duration)) or float(duration) != float(final["videoDurationS"]):
            raise ValueError(f"{video_id}: source manifest and final label duration differ")
        record_path = Path(str(record.get("path") or ""))
        if not record_path.is_absolute():
            record_path = manifest_path.parent / record_path
        if not record_path.is_file() or _sha256(record_path) != str(final["videoSha256"]).lower():
            raise ValueError(f"{video_id}: source manifest video bytes do not match label SHA-256")
    return {"manifestRows": by_id}


def _extract_feature_rows(path: Path, label_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError("feature manifest is missing")
    feature_rows = _read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(feature_rows):
        _reject_disallowed_provenance(row, f"features[{index}]")
        video_id = str(row.get("videoId") or "").strip()
        if not video_id or video_id in by_id:
            raise ValueError(f"feature manifest has duplicate or missing videoId at row {index}")
        if FORBIDDEN_FEATURE_KEYS & set(row):
            raise ValueError(f"{video_id}: MOS/event/model outputs are forbidden feature inputs")
        by_id[video_id] = row
    labels_by_id = {str(row["videoId"]): row for row in label_rows}
    if set(by_id) != set(labels_by_id):
        raise ValueError("feature and label videoId sets differ")

    verified: list[dict[str, Any]] = []
    for video_id in sorted(labels_by_id):
        label, feature = labels_by_id[video_id], by_id[video_id]
        digest = str(label["videoSha256"]).strip().lower()
        if str(feature.get("videoSha256") or "").strip().lower() != digest:
            raise ValueError(f"{video_id}: feature and label video SHA-256 differ")
        if str(feature.get("sourceUrl") or "").strip() != str(label["sourceUrl"]).strip():
            raise ValueError(f"{video_id}: feature and label sourceUrl differ")
        if str(feature.get("license") or "").strip() != str(label["license"]).strip():
            raise ValueError(f"{video_id}: feature and label license differ")
        if feature.get("featureProfile") != FEATURE_PROFILE or feature.get("extractionConfig") != EXTRACTION_CONFIG:
            raise ValueError(f"{video_id}: fixed native extraction profile/config is required")
        video_path_value = feature.get("videoPath")
        if not isinstance(video_path_value, str) or not video_path_value.strip():
            raise ValueError(f"{video_id}: videoPath is required to verify source bytes")
        video_path = Path(video_path_value).resolve(strict=True)
        if not video_path.is_file() or _sha256(video_path) != digest:
            raise ValueError(f"{video_id}: source video bytes do not match the human label hash")
        windows, metadata = extract_video_features(video_path, **EXTRACTION_CONFIG)
        if _sha256(video_path) != digest:
            raise ValueError(f"{video_id}: source video bytes changed during extraction")
        if metadata.get("feature_profile") != FEATURE_PROFILE or metadata.get("extraction_config") != EXTRACTION_CONFIG:
            raise ValueError(f"{video_id}: extractor returned unexpected profile/config")
        values_by_column: dict[str, list[float]] = {column: [] for column in FEATURE_COLUMNS}
        for window in windows:
            for column in FEATURE_COLUMNS:
                try:
                    value = float(window[column])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{video_id}: extracted feature {column} is missing or invalid") from exc
                if not math.isfinite(value):
                    raise ValueError(f"{video_id}: extracted feature {column} is non-finite")
                values_by_column[column].append(value)
        aggregated = [float(np.mean(values_by_column[column])) for column in FEATURE_COLUMNS]
        feature_values = feature.get("featureValues")
        if not isinstance(feature_values, list) or len(feature_values) != len(FEATURE_COLUMNS):
            raise ValueError(f"{video_id}: cached featureValues must match the 30-column profile")
        try:
            claimed = np.asarray([float(value) for value in feature_values], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{video_id}: cached features are invalid") from exc
        if not np.isfinite(claimed).all() or not np.allclose(claimed, aggregated, rtol=1e-9, atol=1e-10):
            raise ValueError(f"{video_id}: cached features do not match re-extracted source video")
        verified.append({
            "videoId": video_id,
            "videoSha256": digest,
            "sourceUrl": str(label["sourceUrl"]).strip(),
            "license": str(label["license"]).strip(),
            "videoPath": str(video_path),
            "featureValues": aggregated,
            "extractionMetadata": metadata,
        })
    return verified


def grouped_folds(rows: list[dict[str, Any]], count: int, seed: int):
    if count < 2:
        raise ValueError("fold count must be at least two")
    groups = sorted({str(row["videoSha256"]).lower() for row in rows})
    if len(groups) < count:
        raise ValueError("insufficient source SHA groups for fixed fold count")
    np.random.default_rng(seed).shuffle(groups)
    for held_groups_array in np.array_split(np.asarray(groups, dtype=object), count):
        held_groups = set(held_groups_array.tolist())
        train_indices = np.asarray([i for i, row in enumerate(rows) if str(row["videoSha256"]).lower() not in held_groups], dtype=int)
        held_indices = np.asarray([i for i, row in enumerate(rows) if str(row["videoSha256"]).lower() in held_groups], dtype=int)
        yield train_indices, held_indices


def _candidate_factory(name: str):
    if name == "logistic_c_1":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", random_state=SEED, max_iter=2000),
        )
    if name == "random_forest_leaf_5":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=4,
                random_state=SEED,
            ),
        )
    raise ValueError(f"unknown candidate: {name}")


def _scores(model: Any, values: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_score"):
        result = np.asarray(model.predict_score(values), dtype=np.float64)
    elif hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError("candidate did not fit the reviewable positive class")
        result = np.asarray(model.predict_proba(values), dtype=np.float64)[:, classes.index(1)]
    else:
        raise ValueError("candidate must expose an uncalibrated continuous score")
    if result.shape != (len(values),) or not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError("candidate scores must be finite values in [0, 1]")
    return result


def _source_group_bootstrap(y: np.ndarray, scores: np.ndarray, groups: np.ndarray, *, seed: int = SEED, samples: int = 1000):
    unique = sorted(set(groups.tolist()))
    buckets = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"auroc": [], "balancedAccuracyAtHalf": []}
    for _ in range(samples):
        selected_groups = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([buckets[group] for group in selected_groups])
        sampled_y, sampled_scores = y[indices], scores[indices]
        if len(np.unique(sampled_y)) == 2:
            values["auroc"].append(float(roc_auc_score(sampled_y, sampled_scores)))
            values["balancedAccuracyAtHalf"].append(float(balanced_accuracy_score(sampled_y, sampled_scores >= 0.5)))
    return {
        "samples": samples,
        "validSamples": len(values["auroc"]),
        "auroc95": [float(x) for x in np.quantile(values["auroc"], [0.025, 0.975])] if values["auroc"] else None,
        "balancedAccuracyAtHalf95": [float(x) for x in np.quantile(values["balancedAccuracyAtHalf"], [0.025, 0.975])] if values["balancedAccuracyAtHalf"] else None,
    }


def _summarize(y: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    predicted = (scores >= 0.5).astype(int)
    return {
        "sampleCount": int(len(y)),
        "coverage": float(len(y) / len(y)) if len(y) else 0.0,
        "auroc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "balancedAccuracyAtHalf": float(balanced_accuracy_score(y, predicted)) if len(np.unique(y)) == 2 else None,
        "recallNotReviewable": float(recall_score(y, predicted, pos_label=0)) if np.any(y == 0) else None,
        "recallReviewable": float(recall_score(y, predicted, pos_label=1)) if np.any(y == 1) else None,
        "sourceGroupBootstrap95": _source_group_bootstrap(y, scores, groups),
    }


def _nested_fold_plan(labels: list[dict[str, Any]], outer_splits: int, inner_splits: int):
    """Validate every fixed fold's class coverage before model construction."""
    if any(row.get("label") not in {"reviewable", "not_reviewable"} for row in labels):
        raise ValueError("only binary human reviewability labels may enter fitting")
    source_labels: dict[str, str] = {}
    for row in labels:
        digest, label = str(row["videoSha256"]).lower(), str(row["label"])
        if digest in source_labels and source_labels[digest] != label:
            raise ValueError("conflicting labels for identical source SHA-256")
        source_labels[digest] = label
    y = np.asarray([int(row["label"] == "reviewable") for row in labels])
    if len(np.unique(y)) != 2:
        raise ValueError("both human reviewability classes are required")
    outer_plan = list(grouped_folds(labels, outer_splits, SEED))
    inner_plans = []
    for fold_index, (train_indices, held_indices) in enumerate(outer_plan):
        if len(np.unique(y[train_indices])) != 2:
            raise ValueError(f"outer fold {fold_index} training split lacks a class")
        inner_rows = [labels[index] for index in train_indices]
        inner_plan = list(grouped_folds(inner_rows, inner_splits, SEED + 100 + fold_index))
        for inner_train_local, inner_hold_local in inner_plan:
            if len(np.unique(y[train_indices[inner_train_local]])) != 2:
                raise ValueError(f"inner fold lacks a class in outer fold {fold_index}")
        inner_plans.append(inner_plan)
    return outer_plan, inner_plans


def evaluate_nested_oof(
    labels: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    *,
    model_factory: Callable[[str], Any] = _candidate_factory,
    outer_splits: int = OUTER_FOLDS,
    inner_splits: int = INNER_FOLDS,
    fit_stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Evaluate fixed model families with source-grouped nested OOF predictions."""
    if len(labels) != len(feature_rows) or not labels:
        raise ValueError("labels and features must have equal non-zero length")
    ids = [str(row["videoId"]) for row in labels]
    if len(ids) != len(set(ids)) or ids != [str(row["videoId"]) for row in feature_rows]:
        raise ValueError("labels and features must be uniquely paired in the same order")
    if any(row.get("label") not in {"reviewable", "not_reviewable"} for row in labels):
        raise ValueError("only binary human reviewability labels may enter fitting")
    if any(str(left.get("videoSha256")).lower() != str(right.get("videoSha256")).lower() for left, right in zip(labels, feature_rows)):
        raise ValueError("feature and label SHA-256 must match")
    y = np.asarray([1 if row["label"] == "reviewable" else 0 for row in labels], dtype=int)
    groups = np.asarray([str(row["videoSha256"]).lower() for row in labels], dtype=object)
    x = np.asarray([row["featureValues"] for row in feature_rows], dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != len(labels) or not np.isfinite(x).all():
        raise ValueError("feature matrix must be finite and two-dimensional")
    if len(np.unique(y)) != 2:
        raise ValueError("both human reviewability classes are required")
    outer_plan, inner_plans = _nested_fold_plan(labels, outer_splits, inner_splits)
    fit_stats = fit_stats if fit_stats is not None else {}
    fit_stats.update({"completedFits": 0, "attemptedFits": 0})
    predictions = np.full(len(labels), np.nan, dtype=np.float64)
    selected_families: list[str | None] = [None] * len(labels)
    outer_reports: list[dict[str, Any]] = []
    fit_count = 0
    for fold_index, (train_indices, held_indices) in enumerate(outer_plan):
        y_train = y[train_indices]
        if len(np.unique(y_train)) != 2:
            raise ValueError(f"outer fold {fold_index} training split lacks a class")
        inner_scores: dict[str, float] = {}
        for family in CANDIDATE_NAMES:
            inner_predictions = np.full(len(train_indices), np.nan, dtype=np.float64)
            for inner_train_local, inner_hold_local in inner_plans[fold_index]:
                inner_train, inner_hold = train_indices[inner_train_local], train_indices[inner_hold_local]
                if len(np.unique(y[inner_train])) != 2:
                    raise ValueError(f"inner fold lacks a class in outer fold {fold_index}")
                model = model_factory(family)
                fit_stats["attemptedFits"] += 1
                model.fit(x[inner_train], y[inner_train])
                fit_count += 1
                fit_stats["completedFits"] += 1
                inner_predictions[inner_hold_local] = _scores(model, x[inner_hold])
            if not np.isfinite(inner_predictions).all():
                raise ValueError("inner out-of-fold scores are incomplete")
            inner_scores[family] = float(balanced_accuracy_score(y_train, inner_predictions >= 0.5))
        selected = max(CANDIDATE_NAMES, key=lambda family: inner_scores[family])
        model = model_factory(selected)
        fit_stats["attemptedFits"] += 1
        model.fit(x[train_indices], y_train)
        fit_count += 1
        fit_stats["completedFits"] += 1
        fold_scores = _scores(model, x[held_indices])
        predictions[held_indices] = fold_scores
        for index in held_indices:
            selected_families[int(index)] = selected
        outer_reports.append({
            "outerFold": fold_index,
            "selectedFamily": selected,
            "innerBalancedAccuracy": inner_scores,
            "trainClassCounts": {"not_reviewable": int(np.sum(y_train == 0)), "reviewable": int(np.sum(y_train == 1))},
            "heldClassCounts": {"not_reviewable": int(np.sum(y[held_indices] == 0)), "reviewable": int(np.sum(y[held_indices] == 1))},
            "fitVideoSha256": sorted(set(groups[train_indices].tolist())),
            "heldVideoSha256": sorted(set(groups[held_indices].tolist())),
        })
    if not np.isfinite(predictions).all() or any(value is None for value in selected_families):
        raise ValueError("outer out-of-fold predictions are incomplete")
    overall = _summarize(y, predictions, groups)
    rows = [
        {
            "videoId": str(labels[index]["videoId"]),
            "videoSha256": str(labels[index]["videoSha256"]).lower(),
            "outerFold": next(report["outerFold"] for report in outer_reports if str(labels[index]["videoSha256"]).lower() in report["heldVideoSha256"]),
            "humanLabel": str(labels[index]["label"]),
            "continuousScore": float(predictions[index]),
            "selectedFamily": selected_families[index],
        }
        for index in range(len(labels))
    ]
    return {
        "status": "research_only",
        "automaticGateValidated": False,
        "calibratedProbability": False,
        "scoreMeaning": "uncalibrated continuous reviewability score in [0, 1]",
        "candidateOnly": True,
        "outerFolds": outer_reports,
        "oofPredictions": rows,
        "metrics": overall,
        "fitCount": fit_count,
    }


def _not_ready(reason: str, *, uncertain_count: int = 0) -> dict[str, Any]:
    return {
        "status": "not_ready",
        "automaticGateValidated": False,
        "fitCount": 0,
        "uncertainLabelCount": int(uncertain_count),
        "reason": reason,
    }


def _write_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def run_training(
    labels_path: Path,
    features_path: Path,
    output_dir: Path,
    *,
    model_factory: Callable[[str], Any] = _candidate_factory,
    source_paths: dict[str, Path] | None = None,
    expected_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Validate all human/source inputs before creating output or fitting."""
    labels_path, features_path, output_dir = Path(labels_path), Path(features_path), Path(output_dir)
    if not labels_path.is_file():
        return _not_ready("missing_final_labels")
    try:
        raw_labels = _read_jsonl(labels_path)
        labels, uncertain_count = _validate_label_rows(raw_labels)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        reason = "engineering_only_input" if str(exc) == "engineering_only_input" else f"invalid_labels: {exc}"
        return _not_ready(reason)
    try:
        _reject_disallowed_provenance(raw_labels, "final_labels")
    except ValueError as exc:
        reason = "engineering_only_input" if "engineering-only" in str(exc) else str(exc)
        return _not_ready(reason, uncertain_count=uncertain_count)
    try:
        source_validation = validate_source_inputs(raw_labels, source_paths or {}, expected_hashes or {})
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return _not_ready(f"invalid_source_provenance: {exc}", uncertain_count=uncertain_count)
    if uncertain_count:
        return _not_ready("unresolved_uncertain_labels", uncertain_count=uncertain_count)
    if len(labels) < MIN_HUMAN_VIDEOS:
        return _not_ready("fewer_than_100_human_videos")
    if len({str(row["videoSha256"]).lower() for row in labels}) < MIN_SOURCE_GROUPS:
        return _not_ready("fewer_than_10_independent_source_groups", uncertain_count=uncertain_count)
    if output_dir.exists():
        return _not_ready("output_directory_already_exists", uncertain_count=uncertain_count)
    try:
        for role, input_path in (("labels", labels_path), ("features", features_path)):
            expected = (expected_hashes or {}).get(role)
            if not _valid_sha(expected) or not input_path.is_file() or _sha256(input_path) != str(expected).lower():
                raise ValueError(f"missing or mismatched trusted SHA-256 for {role}")
        feature_rows = _extract_feature_rows(features_path, labels)
        labels = sorted(labels, key=lambda row: str(row["videoId"]))
        _nested_fold_plan(labels, OUTER_FOLDS, INNER_FOLDS)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _not_ready(f"invalid_or_unverified_features: {exc}", uncertain_count=uncertain_count)
    if output_dir.exists():
        return _not_ready("output_directory_already_exists", uncertain_count=uncertain_count)
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schemaVersion": "task3.reviewability-training/1.0",
        "seed": SEED,
        "outerFolds": OUTER_FOLDS,
        "innerFolds": INNER_FOLDS,
        "minimumSourceGroups": MIN_SOURCE_GROUPS,
        "minimumHumanVideos": MIN_HUMAN_VIDEOS,
        "candidateNames": list(CANDIDATE_NAMES),
        "featureProfile": FEATURE_PROFILE,
        "featureColumns": list(FEATURE_COLUMNS),
        "extractionConfig": EXTRACTION_CONFIG,
        "labelsFileSha256": _sha256(labels_path),
        "featuresFileSha256": _sha256(features_path),
        "usedSplit": "human_reviewability_only",
        "modelExport": False,
        "pythonVersion": platform.python_version(),
        "sourceCodeSha256": _sha256(Path(__file__)),
        "sourceHashes": {
            str(path.relative_to(Path(__file__).resolve().parents[1])): _sha256(path)
            for path in (Path(__file__), Path(__file__).with_name("reviewability_annotations.py"),
                         Path(__file__).with_name("extract_features.py"), Path(__file__).with_name("train_quality.py"))
        },
        "packages": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "opencv-python")},
        "labelRowCount": len(raw_labels),
        "binaryLabelCount": len(labels),
        "uncertainLabelCount": uncertain_count,
        "sourceInputHashes": source_validation["inputHashes"],
    }
    _write_exclusive(output_dir / "preregistration.json", protocol)
    fit_stats: dict[str, int] = {}
    try:
        result = evaluate_nested_oof(labels, feature_rows, model_factory=model_factory, fit_stats=fit_stats)
        result["sourceCodeSha256"] = protocol["sourceCodeSha256"]
        result["labelsFileSha256"] = protocol["labelsFileSha256"]
        result["featuresFileSha256"] = protocol["featuresFileSha256"]
        result["featureColumns"] = list(FEATURE_COLUMNS)
        result["extractionConfig"] = EXTRACTION_CONFIG
        result["labelProvenance"] = [row["provenance"] for row in labels]
        result["sourceInputHashes"] = source_validation["inputHashes"]
        for role, path in {"labels": labels_path, "features": features_path, **(source_paths or {})}.items():
            if _sha256(Path(path)) != str((expected_hashes or {})[role]).lower():
                raise ValueError(f"input changed during training: {role}")
        for row in result["oofPredictions"]:
            row["modelVersion"] = "reviewability-gate-nested-v1"
            row["predictionInputSha256"] = protocol["featuresFileSha256"]
            row["labelInputSha256"] = protocol["labelsFileSha256"]
            row["sourceManifestSha256"] = source_validation["inputHashes"]["manifest"]
            row["fitVideoSha256"] = result["outerFolds"][row["outerFold"]]["fitVideoSha256"]
        with (output_dir / "oof-scores.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for row in result["oofPredictions"]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        # Publish the success marker only after the complete OOF file is closed.
        _write_exclusive(output_dir / "report.json", result)
        return result
    except Exception as exc:
        failure = _not_ready(f"training_failed: {exc}", uncertain_count=uncertain_count)
        failure["fitCount"] = fit_stats.get("completedFits", 0)
        failure["attemptedFitCount"] = fit_stats.get("attemptedFits", 0)
        _write_exclusive(output_dir / "failure.json", failure)
        return failure


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--annotator-a", type=Path)
    parser.add_argument("--annotator-b", type=Path)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--input-sha256", action="append", default=[], metavar="ROLE=DIGEST")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    hashes = {}
    for value in args.input_sha256:
        if "=" not in value:
            raise ValueError("--input-sha256 must use ROLE=DIGEST")
        role, digest = value.split("=", 1)
        if role in hashes:
            raise ValueError("duplicate --input-sha256 role")
        hashes[role] = digest
    source_paths = {role: path for role, path in {
        "manifest": args.manifest, "annotator_a": args.annotator_a,
        "annotator_b": args.annotator_b, "adjudication": args.adjudication,
    }.items() if path is not None}
    result = run_training(args.labels, args.features, args.output_dir, source_paths=source_paths, expected_hashes=hashes)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "research_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
