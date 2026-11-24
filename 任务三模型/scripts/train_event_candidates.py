"""Compare controlled-event classifiers on train/val only.

This candidate workflow never reads test labels and never overwrites the frozen
event-rf artifact. It reuses the repository's event interval and IoU rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from diagnose_events import (
    EVENT_CLASSES,
    TIME_POSTPROCESS_CONFIGS,
    event_metrics_at_threshold,
    merge_windows,
    read_rows,
    validate_lineage_splits,
)
from scripts.independent_event_model import IndependentEventForests
from event_validation_audit import source_group_bootstrap
from extract_anomaly_features import (
    CLASS_NAMES,
    FEATURE_COLUMNS,
    RELATIVE_SHARPNESS_FEATURE_COLUMNS,
    SHARPNESS_RUN_FEATURE_COLUMNS,
    QUANTILE_SHARPNESS_FEATURE_COLUMNS,
    NORMALIZED_SHARPNESS_FEATURE_COLUMNS,
    RELATIVE_EXPOSURE_FEATURE_COLUMNS,
    RELATIVE_EDGE_FEATURE_COLUMNS,
    GRADIENT_SHARPNESS_FEATURE_COLUMNS,
    CONTEXT_SHARPNESS_FEATURE_COLUMNS,
    WIDE_CONTEXT_SHARPNESS_FEATURE_COLUMNS,
    TEMPORAL_FEATURE_COLUMNS,
    TEMPORAL_EVIDENCE_FEATURE_COLUMNS,
    relative_sharpness_features,
    relative_sharpness_quantile_features,
    normalized_sharpness_features,
    relative_exposure_features,
    relative_edge_features,
    context_sharpness_features,
    wide_context_sharpness_features,
    temporal_features,
)


SEED = 42
EVENT_FEATURE_PROFILE_COLUMNS = {
    "event-features-v1": tuple(FEATURE_COLUMNS),
    "event-features-v2": tuple(FEATURE_COLUMNS) + tuple(TEMPORAL_FEATURE_COLUMNS),
    "event-features-v3": tuple(FEATURE_COLUMNS) + tuple(TEMPORAL_FEATURE_COLUMNS) + tuple(RELATIVE_SHARPNESS_FEATURE_COLUMNS),
    "event-features-v4": tuple(FEATURE_COLUMNS) + tuple(TEMPORAL_FEATURE_COLUMNS) + tuple(RELATIVE_SHARPNESS_FEATURE_COLUMNS) + tuple(SHARPNESS_RUN_FEATURE_COLUMNS),
    "event-features-v5": tuple(FEATURE_COLUMNS) + tuple(TEMPORAL_FEATURE_COLUMNS) + tuple(RELATIVE_SHARPNESS_FEATURE_COLUMNS) + tuple(SHARPNESS_RUN_FEATURE_COLUMNS) + tuple(QUANTILE_SHARPNESS_FEATURE_COLUMNS),
    "event-features-v6": tuple(FEATURE_COLUMNS) + tuple(TEMPORAL_FEATURE_COLUMNS) + tuple(RELATIVE_SHARPNESS_FEATURE_COLUMNS) + tuple(SHARPNESS_RUN_FEATURE_COLUMNS) + tuple(QUANTILE_SHARPNESS_FEATURE_COLUMNS) + tuple(NORMALIZED_SHARPNESS_FEATURE_COLUMNS),
    "event-features-v7": tuple(FEATURE_COLUMNS) + tuple(TEMPORAL_FEATURE_COLUMNS) + tuple(RELATIVE_SHARPNESS_FEATURE_COLUMNS) + tuple(SHARPNESS_RUN_FEATURE_COLUMNS) + tuple(QUANTILE_SHARPNESS_FEATURE_COLUMNS) + tuple(NORMALIZED_SHARPNESS_FEATURE_COLUMNS) + tuple(GRADIENT_SHARPNESS_FEATURE_COLUMNS),
    "event-features-v8": tuple(FEATURE_COLUMNS) + tuple(TEMPORAL_FEATURE_COLUMNS) + tuple(RELATIVE_SHARPNESS_FEATURE_COLUMNS) + tuple(SHARPNESS_RUN_FEATURE_COLUMNS) + tuple(QUANTILE_SHARPNESS_FEATURE_COLUMNS) + tuple(NORMALIZED_SHARPNESS_FEATURE_COLUMNS) + tuple(GRADIENT_SHARPNESS_FEATURE_COLUMNS) + tuple(RELATIVE_EXPOSURE_FEATURE_COLUMNS),
    "event-features-v9": tuple(FEATURE_COLUMNS) + tuple(TEMPORAL_FEATURE_COLUMNS) + tuple(RELATIVE_SHARPNESS_FEATURE_COLUMNS) + tuple(SHARPNESS_RUN_FEATURE_COLUMNS) + tuple(QUANTILE_SHARPNESS_FEATURE_COLUMNS) + tuple(NORMALIZED_SHARPNESS_FEATURE_COLUMNS) + tuple(GRADIENT_SHARPNESS_FEATURE_COLUMNS) + tuple(RELATIVE_EXPOSURE_FEATURE_COLUMNS) + tuple(RELATIVE_EDGE_FEATURE_COLUMNS),
    "event-features-v10": tuple(FEATURE_COLUMNS) + tuple(TEMPORAL_FEATURE_COLUMNS) + tuple(TEMPORAL_EVIDENCE_FEATURE_COLUMNS) + tuple(RELATIVE_SHARPNESS_FEATURE_COLUMNS) + tuple(SHARPNESS_RUN_FEATURE_COLUMNS) + tuple(QUANTILE_SHARPNESS_FEATURE_COLUMNS) + tuple(NORMALIZED_SHARPNESS_FEATURE_COLUMNS),
    "event-features-v11": tuple(FEATURE_COLUMNS) + tuple(TEMPORAL_FEATURE_COLUMNS) + tuple(TEMPORAL_EVIDENCE_FEATURE_COLUMNS) + tuple(RELATIVE_SHARPNESS_FEATURE_COLUMNS) + tuple(SHARPNESS_RUN_FEATURE_COLUMNS) + tuple(QUANTILE_SHARPNESS_FEATURE_COLUMNS) + tuple(NORMALIZED_SHARPNESS_FEATURE_COLUMNS) + tuple(GRADIENT_SHARPNESS_FEATURE_COLUMNS) + tuple(RELATIVE_EXPOSURE_FEATURE_COLUMNS) + tuple(RELATIVE_EDGE_FEATURE_COLUMNS),
    "event-features-v12": tuple(FEATURE_COLUMNS) + tuple(TEMPORAL_FEATURE_COLUMNS) + tuple(TEMPORAL_EVIDENCE_FEATURE_COLUMNS) + tuple(RELATIVE_SHARPNESS_FEATURE_COLUMNS) + tuple(SHARPNESS_RUN_FEATURE_COLUMNS) + tuple(QUANTILE_SHARPNESS_FEATURE_COLUMNS) + tuple(NORMALIZED_SHARPNESS_FEATURE_COLUMNS) + tuple(GRADIENT_SHARPNESS_FEATURE_COLUMNS) + tuple(RELATIVE_EXPOSURE_FEATURE_COLUMNS) + tuple(RELATIVE_EDGE_FEATURE_COLUMNS) + tuple(CONTEXT_SHARPNESS_FEATURE_COLUMNS),
    "event-features-v13": tuple(FEATURE_COLUMNS) + tuple(TEMPORAL_FEATURE_COLUMNS) + tuple(TEMPORAL_EVIDENCE_FEATURE_COLUMNS) + tuple(RELATIVE_SHARPNESS_FEATURE_COLUMNS) + tuple(SHARPNESS_RUN_FEATURE_COLUMNS) + tuple(QUANTILE_SHARPNESS_FEATURE_COLUMNS) + tuple(NORMALIZED_SHARPNESS_FEATURE_COLUMNS) + tuple(GRADIENT_SHARPNESS_FEATURE_COLUMNS) + tuple(RELATIVE_EXPOSURE_FEATURE_COLUMNS) + tuple(RELATIVE_EDGE_FEATURE_COLUMNS) + tuple(CONTEXT_SHARPNESS_FEATURE_COLUMNS) + tuple(WIDE_CONTEXT_SHARPNESS_FEATURE_COLUMNS),
}
FEATURE_PROFILE_COLUMNS = EVENT_FEATURE_PROFILE_COLUMNS
_FORBIDDEN_FEATURE_COLUMNS = {
    "label", "groundtruthevent", "issynthetic", "mos", "MOS", "event", "anomaly",
}
RF_SPECS = (
    {"n_estimators": 160, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"n_estimators": 240, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"n_estimators": 160, "min_samples_leaf": 4, "max_features": "sqrt"},
    {"n_estimators": 160, "min_samples_leaf": 2, "max_features": 0.75},
)
HGB_SPECS = (
    {"max_iter": 100, "max_leaf_nodes": 7, "learning_rate": 0.08, "l2_regularization": 1.0},
    {"max_iter": 160, "max_leaf_nodes": 7, "learning_rate": 0.05, "l2_regularization": 1.0},
)
BLUR_SAMPLE_WEIGHT_MULTIPLIERS = (2.0, 4.0)


def _model(name: str, spec: dict[str, Any], seed: int) -> Any:
    family = spec["family"]
    if family == "one_vs_rest_random_forest":
        return IndependentEventForests(
            {key: value for key, value in spec.items() if key != "family"}, seed
        )
    if family == "random_forest":
        return RandomForestClassifier(
            **{key: value for key, value in spec.items() if key not in {"family", "classWeightMultipliers"}},
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    return HistGradientBoostingClassifier(
        **{key: value for key, value in spec.items() if key not in {"family", "classWeightMultipliers"}},
        random_state=seed,
    )


def fit_candidate_model(
    model: Any,
    features: np.ndarray,
    labels: list[str],
    spec: dict[str, Any],
) -> Any:
    class_weight_multipliers = spec.get("classWeightMultipliers")
    if not class_weight_multipliers:
        return model.fit(features, labels)
    sample_weight = np.asarray(
        [float(class_weight_multipliers.get(label, 1.0)) for label in labels],
        dtype=np.float64,
    )
    return model.fit(features, labels, sample_weight=sample_weight)


def build_feature_matrix(rows: list[dict[str, Any]], feature_columns: tuple[str, ...] | list[str]) -> np.ndarray:
    columns = tuple(feature_columns)
    forbidden = [column for column in columns if column.casefold() in {name.casefold() for name in _FORBIDDEN_FEATURE_COLUMNS}]
    if forbidden:
        raise ValueError(f"label and metadata fields cannot be model features: {forbidden}")
    try:
        values = np.asarray([[float(row[column]) for column in columns] for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("feature rows do not match the selected feature profile") from exc
    if not np.isfinite(values).all():
        raise ValueError("feature table contains non-finite model features")
    return values


def enrich_temporal_features(
    rows: list[dict[str, Any]], *, relative_sharpness: bool = False, quantile_sharpness: bool = False,
    normalized_sharpness: bool = False, relative_exposure: bool = False, relative_edge: bool = False,
    context_sharpness: bool = False, wide_context_sharpness: bool = False,
) -> list[dict[str, Any]]:
    """Build v2 summaries from development rows only; never inspect test observations."""
    development = [row for row in rows if row.get("split") in {"train", "val"}]
    summaries = temporal_features(development)
    relative = relative_sharpness_features(development) if relative_sharpness else None
    quantiles = relative_sharpness_quantile_features(development) if quantile_sharpness else None
    normalized = normalized_sharpness_features(development) if normalized_sharpness else None
    exposure = relative_exposure_features(development) if relative_exposure else None
    edge = relative_edge_features(development) if relative_edge else None
    context = context_sharpness_features(development) if context_sharpness else None
    wide_context = wide_context_sharpness_features(development) if wide_context_sharpness else None
    enriched: list[dict[str, Any]] = []
    for index, row in enumerate(development):
        video_id = str(row.get("videoId") or row.get("sourceVideoId"))
        enriched.append({
            **row,
            **summaries[video_id],
            **(relative[index] if relative is not None else {}),
            **(quantiles[index] if quantiles is not None else {}),
            **(normalized[index] if normalized is not None else {}),
            **(exposure[index] if exposure is not None else {}),
            **(edge[index] if edge is not None else {}),
            **(context[index] if context is not None else {}),
            **(wide_context[index] if wide_context is not None else {}),
        })
    return enriched


def _specs() -> list[tuple[str, dict[str, Any]]]:
    specs = [(f"rf_{item['n_estimators']}_leaf_{item['min_samples_leaf']}_features_{item['max_features']}", {"family": "random_forest", **item}) for item in RF_SPECS]
    weighted_rf = RF_SPECS[0]
    specs.extend(
        (
            f"rf_{weighted_rf['n_estimators']}_leaf_{weighted_rf['min_samples_leaf']}_features_{weighted_rf['max_features']}__blur_weight_{int(multiplier)}",
            {
                "family": "random_forest",
                **weighted_rf,
                "classWeightMultipliers": {"blur": multiplier},
            },
        )
        for multiplier in BLUR_SAMPLE_WEIGHT_MULTIPLIERS
    )
    specs.extend(
        (f"hgb_{item['max_iter']}_leaf_{item['max_leaf_nodes']}_lr_{item['learning_rate']}", {"family": "hist_gradient_boosting", **item})
        for item in HGB_SPECS
    )
    specs.append((
        "ovr_rf_160_leaf_2_features_sqrt",
        {
            "family": "one_vs_rest_random_forest",
            "n_estimators": 160,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
    ))
    return specs


def select_best_candidate(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose a validation candidate without treating valid zero metrics as missing."""
    if not evaluations:
        raise ValueError("candidate evaluations are empty")

    def metric(item: dict[str, Any], name: str, missing: float) -> float:
        value = item["validationMetrics"].get(name)
        return missing if value is None else float(value)

    return max(
        evaluations,
        key=lambda item: (
            metric(item, "macroF1", -math.inf),
            metric(item, "freezeRecallAtLeast3s", -math.inf),
            -metric(item, "lowMotionNegativeFreezeFalseAlarmRate", math.inf),
        ),
    )


def _strength_descriptor(row: dict[str, Any], event_class: str) -> tuple[str, str]:
    explicit: Any = None
    for key in ("strengthLevel", "strength", "strength_level"):
        if row.get(key) is not None:
            explicit = row[key]
            break
    parameters = row.get("parameters")
    if explicit is None and isinstance(parameters, dict):
        for key in ("strengthLevel", "strength", "strength_level"):
            if parameters.get(key) is not None:
                explicit = parameters[key]
                break
        if explicit is None and len(parameters) == 1:
            parameter_name, explicit = next(iter(parameters.items()))
            if isinstance(explicit, (int, float)):
                return f"parameter_{parameter_name}_{explicit:g}", "explicit_manifest"
            return f"parameter_{parameter_name}_{str(explicit)}", "explicit_manifest"
    if explicit is not None:
        return f"strength_{explicit:g}" if isinstance(explicit, (int, float)) else f"strength_{str(explicit)}", "explicit_manifest"
    video_id = str(row.get("videoId") or "")
    match = re.search(rf"__{re.escape(event_class)}__s(\d+)$", video_id)
    if match:
        return f"strength_{match.group(1)}", "inferred_id"
    return "strength_unknown", "unknown"


def _window_truth_label(row: dict[str, Any]) -> str:
    event = row.get("groundTruthEvent")
    if event is None:
        return "normal"
    start = float(row["windowStartS"])
    end = float(row["windowEndS"])
    center = round((start + end) / 2.0, 6)
    return str(event["type"]) if float(event["startTimeS"]) <= center < float(event["endTimeS"]) else "normal"


def _validate_window_label(row: dict[str, Any]) -> str:
    actual = _window_truth_label(row)
    if row.get("label") is not None and row["label"] != actual:
        raise ValueError(f"{row.get('videoId', '<unknown>')} label disagrees with groundTruthEvent window center")
    return actual


def _select_thresholds(
    model: Any, rows: list[dict[str, Any]], feature_columns: tuple[str, ...], time_config: dict[str, Any]
) -> dict[str, float]:
    probabilities = model.predict_proba(build_feature_matrix(rows, feature_columns))
    by_class = {str(name): probabilities[:, index] for index, name in enumerate(model.classes_)}
    thresholds: dict[str, float] = {}
    for event_class in EVENT_CLASSES:
        ranked = []
        for threshold in [round(value, 2) for value in np.arange(0.15, 0.851, 0.05)]:
            metrics = event_metrics_at_threshold(rows, by_class[event_class], event_class, threshold, time_config)
            ranked.append((metrics["f1"], metrics["precision"], threshold))
        thresholds[event_class] = max(ranked, key=lambda item: (item[0], item[1], item[2]))[2]
    return thresholds


def _macro_metrics(
    rows: list[dict[str, Any]], model: Any, thresholds: dict[str, float], feature_columns: tuple[str, ...], time_config: dict[str, Any]
) -> dict[str, Any]:
    probabilities = model.predict_proba(build_feature_matrix(rows, feature_columns))
    by_class = {str(name): probabilities[:, index] for index, name in enumerate(model.classes_)}
    per_class: dict[str, dict[str, Any]] = {}
    for event_class in EVENT_CLASSES:
        metrics = event_metrics_at_threshold(rows, by_class[event_class], event_class, thresholds[event_class], time_config)
        metrics.update(_window_and_length_diagnostics(rows, by_class[event_class], event_class, thresholds[event_class], time_config))
        per_class[event_class] = metrics
    strata = _stratified_metrics(rows, by_class, thresholds, time_config)
    return {
        "macroF1": float(np.mean([value["f1"] for value in per_class.values()])),
        "freezeRecallAtLeast3s": _freeze_recall(rows, by_class["freeze_suspected"], thresholds["freeze_suspected"], time_config),
        "lowMotionNegativeFreezeFalseAlarmRate": _low_motion_false_alarm(rows, by_class["freeze_suspected"], thresholds["freeze_suspected"], time_config),
        "perClass": per_class,
        "diagnosticStrata": strata,
    }


def _stratified_metrics(
    rows: list[dict[str, Any]],
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    time_config: dict[str, Any],
) -> dict[str, Any]:
    """Report event metrics by generated strength and freeze duration on val."""
    output: dict[str, Any] = {}
    for event_class in EVENT_CLASSES:
        groups: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            event = row.get("groundTruthEvent")
            if event is None or event.get("type") != event_class:
                continue
            if event_class == "freeze_suspected":
                duration = round((float(event["endTimeS"]) - float(event["startTimeS"])) * 2) / 2
                strength_key, strength_source = _strength_descriptor(row, event_class)
                strength = f"__{strength_key}__{strength_source}"
                key = f"duration_{duration:g}s{strength}"
            else:
                strength_key, strength_source = _strength_descriptor(row, event_class)
                key = f"{strength_key}__{strength_source}"
            groups.setdefault(key, []).append(index)
        output[event_class] = {}
        for key, indices in groups.items():
            subset = [rows[index] for index in indices]
            subset_scores = probabilities[event_class][indices]
            output[event_class][key] = event_metrics_at_threshold(
                subset, subset_scores, event_class, thresholds[event_class], time_config
            )
    return output


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "meanS": None, "minS": None, "maxS": None}
    return {
        "count": len(values),
        "meanS": float(np.mean(values)),
        "minS": float(np.min(values)),
        "maxS": float(np.max(values)),
    }


def _window_and_length_diagnostics(
    rows: list[dict[str, Any]], probabilities: np.ndarray, event_class: str, threshold: float, time_config: dict[str, Any]
) -> dict[str, Any]:
    actual = np.asarray([_window_truth_label(row) == event_class for row in rows], dtype=bool)
    predicted = probabilities >= threshold
    return {
        "windowMetrics": {
            "tp": int(np.sum(actual & predicted)),
            "fp": int(np.sum(~actual & predicted)),
            "fn": int(np.sum(actual & ~predicted)),
        },
        "predictedIntervalLengthS": _summary(_predicted_lengths(rows, probabilities, event_class, threshold, time_config)),
        "truthIntervalLengthS": _summary(_truth_lengths(rows, event_class)),
    }


def _predicted_lengths(rows: list[dict[str, Any]], probabilities: np.ndarray, event_class: str, threshold: float, time_config: dict[str, Any]) -> list[float]:
    by_video: dict[str, list[dict[str, Any]]] = {}
    for row, probability in zip(rows, probabilities):
        if float(probability) >= threshold:
            by_video.setdefault(str(row["videoId"]), []).append(row)
    lengths = []
    for video_rows in by_video.values():
        lengths.extend(
            interval["endTimeS"] - interval["startTimeS"]
            for interval in merge_windows(
                video_rows,
                max_gap_s=float(time_config["maxGapS"]),
                min_duration_s=float(time_config["minimumEventDurationS"]),
                confirmation_windows=int(time_config["confirmationWindows"]),
                recovery_s=float(time_config["recoveryS"]),
            )
        )
    return [float(value) for value in lengths]


def _truth_lengths(rows: list[dict[str, Any]], event_class: str) -> list[float]:
    seen: set[str] = set()
    lengths = []
    for row in rows:
        video_id = str(row["videoId"])
        event = row.get("groundTruthEvent")
        if video_id in seen or event is None or event.get("type") != event_class:
            continue
        seen.add(video_id)
        lengths.append(float(event["endTimeS"]) - float(event["startTimeS"]))
    return lengths


def _freeze_recall(rows: list[dict[str, Any]], probabilities: np.ndarray, threshold: float, time_config: dict[str, Any]) -> float | None:
    truth: dict[str, tuple[float, float]] = {}
    for row in rows:
        event = row.get("groundTruthEvent")
        if event is not None and event.get("type") == "freeze_suspected" and float(event["endTimeS"]) - float(event["startTimeS"]) >= 3.0:
            truth[str(row["videoId"])] = (float(event["startTimeS"]), float(event["endTimeS"]))
    if not truth:
        return None
    eligible = set(truth)
    filtered = [row for row in rows if str(row["videoId"]) in eligible]
    indices = [index for index, row in enumerate(rows) if str(row["videoId"]) in eligible]
    metrics = event_metrics_at_threshold(filtered, probabilities[indices], "freeze_suspected", threshold, time_config)
    return float(metrics["recall"])


def _low_motion_false_alarm(rows: list[dict[str, Any]], probabilities: np.ndarray, threshold: float, time_config: dict[str, Any]) -> float | None:
    by_video: dict[str, list[dict[str, Any]]] = {}
    for row, probability in zip(rows, probabilities):
        if bool(row.get("isSynthetic")) or row.get("groundTruthEvent") is not None:
            continue
        by_video.setdefault(str(row["videoId"]), []).append({**row, "probability": float(probability)})
    if not by_video:
        return None
    ranked = sorted(by_video.items(), key=lambda item: float(np.mean([float(row["frame_diff"]) for row in item[1]])))[:30]
    false_alarms = 0
    for _, video_rows in ranked:
        candidates = [row for row in video_rows if row["probability"] >= threshold]
        if merge_windows(
            candidates,
            max_gap_s=float(time_config["maxGapS"]),
            min_duration_s=float(time_config["minimumEventDurationS"]),
            confirmation_windows=int(time_config["confirmationWindows"]),
            recovery_s=float(time_config["recoveryS"]),
        ):
            false_alarms += 1
    return false_alarms / len(ranked)


def validation_error_report(
    rows: list[dict[str, Any]],
    probabilities: dict[str, np.ndarray] | None = None,
    thresholds: dict[str, float] | None = None,
    time_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize event errors using validation rows only, with known metadata strata."""
    val_indices = [index for index, row in enumerate(rows) if row.get("split") == "val"]
    val_rows = []
    for index in val_indices:
        row = dict(rows[index])
        row.setdefault("videoId", row.get("recordId") or f"val-row-{index}")
        row.setdefault("windowStartS", 0.0)
        row.setdefault("windowEndS", 0.5)
        val_rows.append(row)
    if not val_rows:
        raise ValueError("validation split is empty")

    if probabilities is None:
        return {
            "split": "val",
            "predictionProvided": False,
            "coverage": {"windowCount": len(val_rows), "videoCount": len({str(row["videoId"]) for row in val_rows})},
            "testLabelsUsed": False,
            "testFeaturesUsed": False,
            "testNotUsed": True,
            "limitations": ["未提供模型概率；这里只报告验证覆盖，不生成 precision/recall/F1。"],
        }
    if thresholds is None:
        thresholds = {event_class: 0.5 for event_class in EVENT_CLASSES}

    by_class: dict[str, Any] = {}
    for event_class in EVENT_CLASSES:
        val_scores = np.asarray(probabilities[event_class])[val_indices]
        threshold = float(thresholds[event_class])
        result = event_metrics_at_threshold(val_rows, val_scores, event_class, threshold, time_config)
        grouped: dict[str, list[int]] = {}
        errors = []
        for local_index, row in enumerate(val_rows):
            event = row.get("groundTruthEvent")
            actual_label = _validate_window_label(row)
            actual = actual_label == event_class
            predicted = float(val_scores[local_index]) >= threshold
            if actual != predicted:
                errors.append(
                    {
                        "errorType": "fp" if predicted else "fn",
                        "recordId": row.get("recordId") or row["videoId"],
                        "sourceVideoId": row.get("sourceVideoId"),
                        "videoId": row["videoId"],
                        "windowStartS": float(row["windowStartS"]),
                        "windowEndS": float(row["windowEndS"]),
                        "probability": float(val_scores[local_index]),
                    }
                )
            if not actual:
                key = "negative_or_unknown"
                grouped.setdefault(key, []).append(local_index)
            else:
                video_id = str(row["videoId"])
                event_length = float(event["endTimeS"]) - float(event["startTimeS"])
                strength_key, strength_source = _strength_descriptor(row, event_class)
                if event_class == "freeze_suspected":
                    keys = [f"duration_{event_length:g}s__{strength_key}__{strength_source}", f"eventLength_{event_length:g}s"]
                else:
                    keys = [f"{strength_key}__{strength_source}", f"eventLength_{event_length:g}s"]
                for key in keys:
                    grouped.setdefault(key, []).append(local_index)

        strata: dict[str, Any] = {}
        for key, indices in grouped.items():
            subset = [val_rows[index] for index in indices]
            strata[key] = event_metrics_at_threshold(subset, val_scores[indices], event_class, threshold, time_config)
        by_class[event_class] = {**result, "strata": strata, "errors": errors}

    return {
        "split": "val",
        "testLabelsUsed": False,
        "testFeaturesUsed": False,
        "testNotUsed": True,
        "byClass": by_class,
        "limitations": ["分层优先使用 manifest 强度字段；旧记录可从 videoId 推断并标注 inferred_id，其余标注 unknown；冻结时长来自验证真值区间。"],
    }


def run_event_candidate_experiment(
    input_path: Path,
    output_dir: Path,
    *,
    seed: int = SEED,
    feature_profile: str = "event-features-v1",
    provenance_path: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if feature_profile not in EVENT_FEATURE_PROFILE_COLUMNS:
        raise ValueError(f"unsupported feature profile: {feature_profile}")
    feature_columns = EVENT_FEATURE_PROFILE_COLUMNS[feature_profile]
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    registered_input_sha256: str | None = None
    if provenance_path is not None:
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            registered_input_sha256 = provenance["candidateInputs"]["eventFeatures"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid event candidate provenance: {provenance_path}") from exc
        if registered_input_sha256 != input_sha256:
            raise ValueError(
                "event candidate input SHA-256 does not match registered provenance"
            )
    rows = read_rows(input_path)
    validate_lineage_splits(rows)
    if any(any(str(key).casefold() == "mos" for key in row) for row in rows):
        raise ValueError("event candidate input must not contain MOS labels")
    train_rows = [row for row in rows if row.get("split") == "train"]
    val_rows = [row for row in rows if row.get("split") == "val"]
    if not train_rows or not val_rows:
        raise ValueError("train and val splits are required")
    if feature_profile in {
        "event-features-v7", "event-features-v8", "event-features-v9",
        "event-features-v10", "event-features-v11", "event-features-v12", "event-features-v13",
    }:
        missing_raw_columns = sorted({
            column
            for row in train_rows + val_rows
            for column in GRADIENT_SHARPNESS_FEATURE_COLUMNS
            if column not in row
        })
        if missing_raw_columns:
            raise ValueError(
                f"{feature_profile} input is missing required raw feature columns: {missing_raw_columns}"
            )
    if any(row.get("label") not in CLASS_NAMES for row in train_rows + val_rows):
        raise ValueError("unsupported event label")
    labels = {str(row["label"]) for row in train_rows}
    if labels != set(CLASS_NAMES):
        raise ValueError("train split must contain all event classes")
    if feature_profile in {"event-features-v2", "event-features-v3", "event-features-v4", "event-features-v5", "event-features-v6", "event-features-v7", "event-features-v8", "event-features-v9", "event-features-v10", "event-features-v11", "event-features-v12", "event-features-v13"}:
        development = enrich_temporal_features(
            rows,
            relative_sharpness=feature_profile in {"event-features-v3", "event-features-v4", "event-features-v5", "event-features-v6", "event-features-v7", "event-features-v8", "event-features-v9", "event-features-v10", "event-features-v11", "event-features-v12", "event-features-v13"},
            quantile_sharpness=feature_profile in {"event-features-v5", "event-features-v6", "event-features-v7", "event-features-v8", "event-features-v9", "event-features-v10", "event-features-v11", "event-features-v12", "event-features-v13"},
            normalized_sharpness=feature_profile in {"event-features-v6", "event-features-v7", "event-features-v8", "event-features-v9", "event-features-v10", "event-features-v11", "event-features-v12", "event-features-v13"},
            relative_exposure=feature_profile in {"event-features-v8", "event-features-v9", "event-features-v11", "event-features-v12", "event-features-v13"},
            relative_edge=feature_profile in {"event-features-v9", "event-features-v11", "event-features-v12", "event-features-v13"},
            context_sharpness=feature_profile in {"event-features-v12", "event-features-v13"},
            wide_context_sharpness=feature_profile == "event-features-v13",
        )
        train_rows = [row for row in development if row.get("split") == "train"]
        val_rows = [row for row in development if row.get("split") == "val"]
    x_train = build_feature_matrix(train_rows, feature_columns)
    y_train = [str(row["label"]) for row in train_rows]
    evaluations: list[dict[str, Any]] = []
    for name, spec in _specs():
        model = _model(name, spec, seed)
        fit_candidate_model(model, x_train, y_train, spec)
        for time_name, time_config in TIME_POSTPROCESS_CONFIGS.items():
            thresholds = _select_thresholds(model, val_rows, feature_columns, time_config)
            metrics = _macro_metrics(val_rows, model, thresholds, feature_columns, time_config)
            evaluations.append(
                {
                    "name": f"{name}__{time_name}",
                    "parameters": spec,
                    "timePostprocess": dict(time_config),
                    "thresholds": thresholds,
                    "validationMetrics": metrics,
                    "model": model,
                }
            )
    selected = select_best_candidate(evaluations)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    model_file = "event-v2-candidate.joblib"
    artifact = {
        "model": selected["model"],
        "modelVersion": f"{selected['name']}-candidate-v2",
        "featureProfile": feature_profile,
        "featureColumns": feature_columns,
        "eventClasses": EVENT_CLASSES,
        "thresholds": selected["thresholds"],
        "mergeMaxGapS": float(selected["timePostprocess"]["maxGapS"]),
        "minimumEventDurationS": float(selected["timePostprocess"]["minimumEventDurationS"]),
        "timePostprocess": selected["timePostprocess"],
        "trainingSeed": seed,
        "trainedOn": "train",
        "selectedOn": "val",
        "frozen": True,
        "candidateOnly": True,
    }
    joblib.dump(artifact, output_dir / model_file)
    model_sha256 = hashlib.sha256((output_dir / model_file).read_bytes()).hexdigest()
    selected_probabilities = selected["model"].predict_proba(build_feature_matrix(val_rows, feature_columns))
    selected_class_indices = {str(name): index for index, name in enumerate(selected["model"].classes_)}
    error_report = validation_error_report(
        val_rows,
        {event_class: selected_probabilities[:, selected_class_indices[event_class]] for event_class in EVENT_CLASSES},
        selected["thresholds"],
        selected["timePostprocess"],
    )
    error_report["inputSha256"] = input_sha256
    error_report["testNotUsed"] = True
    validation_bootstrap = source_group_bootstrap(
        val_rows,
        {event_class: selected_probabilities[:, selected_class_indices[event_class]] for event_class in EVENT_CLASSES},
        selected["thresholds"],
        selected["timePostprocess"],
        samples=2000,
        seed=seed,
    )
    report = {
        "schemaVersion": "task3.event-candidate-report/2.0",
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": {"path": str(input_path), "sha256": input_sha256},
        "provenance": (
            {"path": str(provenance_path), "registeredInputSha256": registered_input_sha256}
            if provenance_path is not None
            else None
        ),
        "featureProfile": feature_profile,
        "featureColumns": list(feature_columns),
        "splitCounts": {"train": len(train_rows), "val": len(val_rows)},
        "seed": seed,
        "selectionRule": ["val event Macro-F1 descending", ">=3s freeze recall descending", "validation false positives ascending"],
        "timePostprocessCandidates": [dict(config) for config in TIME_POSTPROCESS_CONFIGS.values()],
        "selectedCandidate": selected["name"],
        "timePostprocess": selected["timePostprocess"],
        "modelFile": model_file,
        "modelSha256": model_sha256,
        "testNotUsed": True,
        "candidates": [{key: value for key, value in item.items() if key != "model"} for item in evaluations],
        "validationErrorReport": error_report,
        "validationBootstrap": validation_bootstrap,
        "limitations": ["候选报告只代表 train/val 受控合成事件；未运行冻结 test，不能作为正式验收结果。"],
    }
    (output_dir / "event_candidate_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    diagnostics = {
        "split": "val",
        "testNotUsed": True,
        "inputSha256": input_sha256,
        "candidates": {item["name"]: item["validationMetrics"] for item in evaluations},
        "selectedCandidateErrorReport": error_report,
        "selectedCandidateValidationBootstrap": validation_bootstrap,
    }
    (output_dir / "event_validation_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--feature-profile", choices=sorted(EVENT_FEATURE_PROFILE_COLUMNS), default="event-features-v1")
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("docs/task3-v2-provenance.json"),
        help="provenance JSON whose candidateInputs.eventFeatures SHA-256 must match the feature table",
    )
    args = parser.parse_args(argv)
    print(json.dumps(run_event_candidate_experiment(args.features, args.output_dir, seed=args.seed, feature_profile=args.feature_profile, provenance_path=args.provenance), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
