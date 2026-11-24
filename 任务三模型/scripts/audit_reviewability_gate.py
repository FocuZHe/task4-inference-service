"""Audit explicit reviewability operating points without fitting or choosing a gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.train_reviewability_gate import (
        CANDIDATE_NAMES, FORBIDDEN_FEATURE_KEYS, MIN_HUMAN_VIDEOS, MIN_SOURCE_GROUPS, OUTER_FOLDS,
        _read_jsonl, _reject_disallowed_provenance, _sha256, _valid_sha,
        _validate_label_rows, validate_source_inputs,
    )
except ModuleNotFoundError:
    from train_reviewability_gate import (
        CANDIDATE_NAMES, FORBIDDEN_FEATURE_KEYS, MIN_HUMAN_VIDEOS, MIN_SOURCE_GROUPS, OUTER_FOLDS,
        _read_jsonl, _reject_disallowed_provenance, _sha256, _valid_sha,
        _validate_label_rows, validate_source_inputs,
    )


def _not_ready(reason: str, *, uncertain_count: int = 0) -> dict[str, Any]:
    return {
        "schemaVersion": "task3.reviewability-gate-audit/1.0",
        "status": "not_ready", "reason": reason,
        "automaticGateValidated": False, "thresholdPromoted": False,
        "thresholdSelectionStatus": "not_selected", "thresholds": [],
        "uncertainLabelCount": uncertain_count,
    }


def _has_engineering_marker(value: object) -> bool:
    if isinstance(value, dict):
        return any(_has_engineering_marker(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_engineering_marker(item) for item in value)
    return isinstance(value, str) and "ENGINEERING_ONLY" in value.upper()


def _check_pure_input_policy(value: object, location: str) -> None:
    """Permit explicit engineering markers only for non-exporting arithmetic calls."""
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).replace("_", "").lower()
            if normalized in {field.replace("_", "").lower() for field in FORBIDDEN_FEATURE_KEYS}:
                raise ValueError(f"{location}: forbidden supervision field {key}")
            if normalized == "split" and not isinstance(item, str):
                raise ValueError(f"{location}: split must be a string")
            if normalized in {"predictiontype", "predictionscope"} and isinstance(item, str):
                if item.lower() in {"train", "in_sample", "insample", "training"}:
                    raise ValueError(f"{location}: in-sample predictions are forbidden")
            if isinstance(item, str) and "ENGINEERING_ONLY" in item.upper():
                # Still screen the key: MOS/model outputs cannot hide behind a marker.
                _reject_disallowed_provenance({key: "engineering-fixture"}, location)
            else:
                _check_pure_input_policy(item, f"{location}.{key}")
                if not isinstance(item, (dict, list)):
                    _reject_disallowed_provenance({key: item}, location)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_pure_input_policy(item, f"{location}[{index}]")


def _thresholds(values: list[float]) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValueError("thresholds must be a non-empty list")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("thresholds must be finite numbers in [0, 1]")
        result.append(float(value))
    if len(set(result)) != len(result):
        raise ValueError("thresholds must be unique")
    return result


def _index(rows: list[dict[str, object]], role: str) -> dict[str, dict[str, object]]:
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("videoId"), str) or not str(row["videoId"]).strip():
            raise ValueError(f"{role}: videoId must be a non-empty string")
        video_id = str(row["videoId"]).strip()
        if video_id in indexed:
            raise ValueError(f"{role}: duplicate videoId {video_id}")
        indexed[video_id] = row
    return indexed


def _pair_oof(final_labels: list[dict[str, object]], predictions: list[dict[str, object]]) -> list[dict[str, Any]]:
    labels, oof = _index(final_labels, "labels"), _index(predictions, "predictions")
    if set(labels) != set(oof):
        raise ValueError("labels and predictions must cover the same videoId set")
    rows = []
    group_folds: dict[str, int] = {}
    group_labels: dict[str, str] = {}
    shared: dict[str, str] = {}
    fold_models: dict[int, tuple[str, frozenset[str]]] = {}
    for video_id in sorted(labels):
        label, prediction = labels[video_id], oof[video_id]
        digest = str(label["videoSha256"]).strip().lower()
        if not _valid_sha(prediction.get("videoSha256")) or str(prediction["videoSha256"]).strip().lower() != digest:
            raise ValueError(f"{video_id}: prediction video SHA-256 mismatch")
        if prediction.get("humanLabel") != label["label"]:
            raise ValueError(f"{video_id}: OOF humanLabel differs from final label")
        score = prediction.get("continuousScore")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"{video_id}: continuous score must be finite and in [0, 1]")
        fold = prediction.get("outerFold")
        if not isinstance(fold, int) or isinstance(fold, bool) or fold < 0:
            raise ValueError(f"{video_id}: outerFold must be a non-negative integer")
        if digest in group_folds and group_folds[digest] != fold:
            raise ValueError("one source SHA group spans multiple outer folds")
        if digest in group_labels and group_labels[digest] != label["label"]:
            raise ValueError("one source SHA group has conflicting human labels")
        group_folds[digest], group_labels[digest] = fold, str(label["label"])
        family = prediction.get("selectedFamily")
        if not isinstance(family, str) or family not in CANDIDATE_NAMES:
            raise ValueError(f"{video_id}: unknown selectedFamily")
        model_version = prediction.get("modelVersion")
        if not isinstance(model_version, str) or not model_version.strip():
            raise ValueError(f"{video_id}: modelVersion is required")
        for field in ("modelVersion", "predictionInputSha256", "labelInputSha256", "sourceManifestSha256"):
            value = prediction.get(field)
            if field != "modelVersion" and not _valid_sha(value):
                raise ValueError(f"{video_id}: {field} must be a valid SHA-256")
            normalized = str(value).strip() if field == "modelVersion" else str(value).strip().lower()
            if field in shared and shared[field] != normalized:
                raise ValueError(f"OOF {field} must be consistent across predictions")
            shared[field] = normalized
        fit_hashes = prediction.get("fitVideoSha256")
        if not isinstance(fit_hashes, list) or not fit_hashes or not all(_valid_sha(item) for item in fit_hashes):
            raise ValueError(f"{video_id}: fitVideoSha256 must contain training SHA-256 values")
        fit_set = frozenset(str(item).strip().lower() for item in fit_hashes)
        if len(fit_set) != len(fit_hashes):
            raise ValueError(f"{video_id}: duplicate fitVideoSha256")
        if digest in fit_set:
            raise ValueError(f"{video_id}: held source is included in model fit")
        identity = (family, fit_set)
        if fold in fold_models and fold_models[fold] != identity:
            raise ValueError("one outer fold has inconsistent selected family or fit sources")
        fold_models[fold] = identity
        rows.append({"videoId": video_id, "sourceGroup": digest, "label": str(label["label"]),
                     "score": float(score), "outerFold": fold, "fitGroups": fit_set})
    if len(fold_models) < 2:
        raise ValueError("OOF predictions require at least two outer folds")
    all_groups = set(group_folds)
    for row in rows:
        held = {group for group, fold in group_folds.items() if fold == row["outerFold"]}
        if row["fitGroups"] != all_groups - held:
            raise ValueError("OOF fit source groups must equal all sources outside the held fold")
    return rows


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _operating_point(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    tn = fp = fn = tp = 0
    for row in rows:
        positive, passed = row["label"] == "reviewable", row["score"] >= threshold
        tn += int(not positive and not passed)
        fp += int(not positive and passed)
        fn += int(positive and not passed)
        tp += int(positive and passed)
    return {
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "falseAcceptRate": _ratio(fp, fp + tn), "falseRejectRate": _ratio(fn, fn + tp),
        "passRate": _ratio(tp + fp, len(rows)),
        "reviewWorkloadCount": tn + fn, "reviewWorkloadRate": _ratio(tn + fn, len(rows)),
    }


def _bootstrap(rows: list[dict[str, Any]], threshold: float, samples: int, seed: int) -> dict[str, Any]:
    groups = sorted({row["sourceGroup"] for row in rows})
    buckets = {group: [row for row in rows if row["sourceGroup"] == group] for group in groups}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {field: [] for field in ("falseAcceptRate", "falseRejectRate", "passRate", "reviewWorkloadRate")}
    for _ in range(samples):
        sampled = [row for group in rng.choice(groups, len(groups), replace=True) for row in buckets[group]]
        metrics = _operating_point(sampled, threshold)
        for field in values:
            if metrics[field] is not None:
                values[field].append(metrics[field])
    return {
        "samples": samples, "seed": seed, "sourceGroupCount": len(groups),
        "validSamples": {field: len(items) for field, items in values.items()},
        "intervals": {field: [float(value) for value in np.quantile(items, [0.025, 0.975])] if items else None
                      for field, items in values.items()},
        "scope": "descriptive source-group OOF bootstrap; excludes retraining/selection uncertainty",
    }


def evaluate_gate(
    final_labels: list[dict[str, object]], oof_predictions: list[dict[str, object]], *,
    thresholds: list[float], bootstrap_samples: int = 2000, seed: int = 42,
) -> dict[str, object]:
    """Compute a development trade-off table; never select or validate a gate."""
    candidates = _thresholds(thresholds)
    if not isinstance(bootstrap_samples, int) or isinstance(bootstrap_samples, bool) or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not final_labels:
        return _not_ready("missing_final_labels")
    _index(final_labels, "labels")
    _check_pure_input_policy(final_labels, "labels")
    _check_pure_input_policy(oof_predictions, "predictions")
    uncertain = sum(row.get("label") == "uncertain" for row in final_labels)
    if uncertain:
        return _not_ready("unresolved_uncertain_labels", uncertain_count=uncertain)
    _validate_label_rows(final_labels)
    paired = _pair_oof(final_labels, oof_predictions)
    engineering = _has_engineering_marker(final_labels) or _has_engineering_marker(oof_predictions)
    rows = [{"threshold": value, **_operating_point(paired, value),
             "sourceGroupBootstrap95": _bootstrap(paired, value, bootstrap_samples, seed)}
            for value in candidates]
    return {
        "schemaVersion": "task3.reviewability-gate-audit/1.0",
        "status": "engineering_only" if engineering else "research_only",
        "automaticGateValidated": False, "thresholdPromoted": False,
        "thresholdSelectionStatus": "not_selected", "calibratedProbability": False,
        "sampleCount": len(paired), "sourceGroupCount": len({row["sourceGroup"] for row in paired}),
        "uncertainLabelCount": 0, "thresholds": rows,
        "reviewWorkloadMeaning": "videos failing the gate and requiring follow-up review",
        "limitations": ["OOF development trade-off, not independent threshold validation",
                        "business error costs and unseen real-video confirmation are required before deployment"],
    }


def run_audit(
    labels_path: Path, predictions_path: Path, *, thresholds: list[float],
    source_paths: dict[str, Path] | None = None, expected_hashes: dict[str, str] | None = None,
    bootstrap_samples: int = 2000, seed: int = 42,
) -> dict[str, Any]:
    """Verify file/source provenance before permitting a research-only audit."""
    paths = {"labels": Path(labels_path), "predictions": Path(predictions_path)}
    hashes = expected_hashes or {}
    try:
        _thresholds(thresholds)
        if not paths["labels"].is_file():
            return _not_ready("missing_final_labels")
        if not paths["predictions"].is_file():
            return _not_ready("missing_oof_predictions")
        labels, predictions = _read_jsonl(paths["labels"]), _read_jsonl(paths["predictions"])
        if _has_engineering_marker(labels) or _has_engineering_marker(predictions):
            return _not_ready("engineering_only_input")
        for role, path in paths.items():
            if not _valid_sha(hashes.get(role)) or _sha256(path) != str(hashes[role]).strip().lower():
                raise ValueError(f"missing or mismatched trusted SHA-256 for {role}")
        _reject_disallowed_provenance(predictions, "predictions")
        verified = validate_source_inputs(labels, source_paths or {}, hashes)
        accepted, uncertain = _validate_label_rows(labels)
        if uncertain:
            return _not_ready("unresolved_uncertain_labels", uncertain_count=uncertain)
        if len(accepted) < MIN_HUMAN_VIDEOS:
            return _not_ready("fewer_than_100_human_videos")
        if len({str(row["videoSha256"]).strip().lower() for row in accepted}) < MIN_SOURCE_GROUPS:
            return _not_ready("fewer_than_10_independent_source_groups")
        if not _valid_sha(hashes.get("features")):
            raise ValueError("trusted feature input SHA-256 is required")
        result = evaluate_gate(labels, predictions, thresholds=thresholds, bootstrap_samples=bootstrap_samples, seed=seed)
        if result["status"] != "research_only":
            return result
        if {row["outerFold"] for row in predictions} != set(range(OUTER_FOLDS)):
            raise ValueError("real OOF audit requires fixed five outer folds")
        expected_oof = {"labelInputSha256": hashes["labels"], "predictionInputSha256": hashes["features"],
                        "sourceManifestSha256": verified["inputHashes"]["manifest"]}
        for row in predictions:
            for field, digest in expected_oof.items():
                if str(row[field]).strip().lower() != str(digest).strip().lower():
                    raise ValueError(f"OOF {field} does not match trusted input")
        for role, path in {**paths, **(source_paths or {})}.items():
            if _sha256(Path(path)) != str(hashes[role]).strip().lower():
                raise ValueError(f"input changed during audit: {role}")
        result["inputHashes"] = {**verified["inputHashes"], **{role: _sha256(path) for role, path in paths.items()}, "features": hashes["features"]}
        result["sourceCodeSha256"] = _sha256(Path(__file__))
        return result
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return _not_ready(f"invalid_or_unverified_input: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    for flag in ("manifest", "annotator-a", "annotator-b", "adjudication"):
        parser.add_argument(f"--{flag}", type=Path)
    parser.add_argument("--input-sha256", action="append", default=[], metavar="ROLE=DIGEST")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        candidates = [float(value) for value in args.thresholds.split(",")]
        hashes = {}
        for value in args.input_sha256:
            if "=" not in value:
                raise ValueError("--input-sha256 must use ROLE=DIGEST")
            role, digest = value.split("=", 1)
            if role in hashes:
                raise ValueError("duplicate --input-sha256 role")
            hashes[role] = digest
        sources = {role: path for role, path in {
            "manifest": args.manifest, "annotator_a": args.annotator_a,
            "annotator_b": args.annotator_b, "adjudication": args.adjudication,
        }.items() if path is not None}
        result = run_audit(args.labels, args.predictions, thresholds=candidates, source_paths=sources,
                           expected_hashes=hashes, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    except (ValueError, TypeError) as exc:
        result = _not_ready(f"invalid_arguments: {exc}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 2 if args.strict and result["status"] != "research_only" else 0


if __name__ == "__main__":
    raise SystemExit(main())
