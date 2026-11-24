"""Train, freeze, and independently evaluate controlled Task 3 event diagnosis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

from scripts.extract_anomaly_features import (
    CLASS_NAMES,
    FEATURE_COLUMNS,
    RELATIVE_SHARPNESS_FEATURE_COLUMNS,
    SHARPNESS_RUN_FEATURE_COLUMNS,
    QUANTILE_SHARPNESS_FEATURE_COLUMNS,
    NORMALIZED_SHARPNESS_FEATURE_COLUMNS,
    GRADIENT_SHARPNESS_FEATURE_COLUMNS,
    RELATIVE_EXPOSURE_FEATURE_COLUMNS,
    RELATIVE_EDGE_FEATURE_COLUMNS,
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


EVENT_CLASSES = CLASS_NAMES[1:]
MODEL_VERSION = "task3-event-rf-v1"
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
TIME_POSTPROCESS_CONFIGS = {
    "baseline-gap-0.5-min-0.5": {
        "name": "baseline-gap-0.5-min-0.5",
        "maxGapS": 0.5,
        "confirmationWindows": 1,
        "recoveryS": 0.5,
        "minimumEventDurationS": 0.5,
    },
    "confirm-2-recover-0.5-min-0.5": {
        "name": "confirm-2-recover-0.5-min-0.5",
        "maxGapS": 0.5,
        "confirmationWindows": 2,
        "recoveryS": 0.5,
        "minimumEventDurationS": 0.5,
    },
    "confirm-2-recover-1.0-min-1.0": {
        "name": "confirm-2-recover-1.0-min-1.0",
        "maxGapS": 0.5,
        "confirmationWindows": 2,
        "recoveryS": 1.0,
        "minimumEventDurationS": 1.0,
    },
    "confirm-3-recover-1.0-min-1.0": {
        "name": "confirm-3-recover-1.0-min-1.0",
        "maxGapS": 0.5,
        "confirmationWindows": 3,
        "recoveryS": 1.0,
        "minimumEventDurationS": 1.0,
    },
}
DEFAULT_TIME_POSTPROCESS = TIME_POSTPROCESS_CONFIGS["baseline-gap-0.5-min-0.5"]
def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    raise ValueError("feature table must be .jsonl or .csv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_lineage_splits(rows: list[dict[str, Any]]) -> None:
    split_by_source: dict[str, str] = {}
    split_by_video: dict[str, str] = {}
    for row in rows:
        source_id = str(row.get("sourceVideoId") or "").strip()
        video_id = str(row.get("videoId") or "").strip()
        split = str(row.get("split") or "").strip()
        if not source_id or not video_id or split not in {"train", "val", "test"}:
            raise ValueError("every feature row requires videoId, sourceVideoId and a supported split")
        if source_id in split_by_source and split_by_source[source_id] != split:
            raise ValueError(f"source video {source_id} appears in a cross split")
        if video_id in split_by_video and split_by_video[video_id] != split:
            raise ValueError(f"video {video_id} appears in a cross split")
        split_by_source[source_id] = split
        split_by_video[video_id] = split


def interval_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def merge_windows(
    rows: list[dict[str, Any]], *, max_gap_s: float = 0.5, min_duration_s: float = 0.5,
    confirmation_windows: int = 1, recovery_s: float | None = None,
) -> list[dict[str, float]]:
    if max_gap_s < 0 or min_duration_s < 0 or confirmation_windows <= 0:
        raise ValueError("gap, duration, and confirmation thresholds must be valid")
    if recovery_s is not None and (not math.isfinite(recovery_s) or recovery_s < 0):
        raise ValueError("recovery threshold must be finite and non-negative")
    effective_gap = max_gap_s if recovery_s is None else recovery_s
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        video_id = str(row.get("videoId") or "__single_video__")
        event_class = str(row.get("eventClass") or row.get("eventType") or "__single_class__")
        groups[(video_id, event_class)].append(row)

    merged_output: list[dict[str, float]] = []
    for grouped_rows in groups.values():
        intervals: list[tuple[list[float], int]] = []
        for row in sorted(grouped_rows, key=lambda item: float(item["windowStartS"])):
            start = float(row["windowStartS"])
            end = float(row["windowEndS"])
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
                raise ValueError("window intervals must be finite and ordered")
            if intervals and start - intervals[-1][0][1] <= effective_gap:
                intervals[-1][0][1] = max(intervals[-1][0][1], end)
                intervals[-1] = (intervals[-1][0], intervals[-1][1] + 1)
            else:
                intervals.append(([start, end], 1))
        for interval, count in intervals:
            if count >= confirmation_windows and interval[1] - interval[0] >= min_duration_s:
                merged_output.append({"startTimeS": interval[0], "endTimeS": interval[1]})
    return sorted(merged_output, key=lambda item: (item["startTimeS"], item["endTimeS"]))


def event_metrics_at_threshold(
    rows: list[dict[str, Any]], scores: list[float] | np.ndarray, event_class: str, threshold: float,
    time_config: dict[str, Any] | None = None,
) -> dict[str, float | int]:
    if len(rows) != len(scores):
        raise ValueError("one event probability is required for each feature window")
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        if not math.isfinite(float(score)):
            raise ValueError("event probabilities must be finite")
        if float(score) >= threshold:
            by_video[str(row["videoId"])].append(row)

    grouped_truth: dict[str, dict[str, Any] | None] = {}
    for row in rows:
        video_id = str(row["videoId"])
        event = row.get("groundTruthEvent")
        if video_id in grouped_truth and grouped_truth[video_id] != event:
            raise ValueError(f"inconsistent event truth across windows for {video_id}")
        grouped_truth[video_id] = event

    config = DEFAULT_TIME_POSTPROCESS if time_config is None else time_config
    tp = fp = fn = 0
    for video_id, event in grouped_truth.items():
        predicted = merge_windows(
            by_video.get(video_id, []),
            max_gap_s=float(config["maxGapS"]),
            min_duration_s=float(config["minimumEventDurationS"]),
            confirmation_windows=int(config["confirmationWindows"]),
            recovery_s=float(config["recoveryS"]),
        )
        truth = (
            (float(event["startTimeS"]), float(event["endTimeS"]))
            if event is not None and event["type"] == event_class
            else None
        )
        if truth is None:
            fp += len(predicted)
            continue
        best_iou = max(
            (interval_iou(truth, (item["startTimeS"], item["endTimeS"])) for item in predicted),
            default=0.0,
        )
        if best_iou >= 0.5:
            tp += 1
            fp += len(predicted) - 1
        else:
            fn += 1
            fp += len(predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _matrix(rows: list[dict[str, Any]], feature_columns: tuple[str, ...] = tuple(FEATURE_COLUMNS)) -> np.ndarray:
    values = np.asarray([[float(row[column]) for column in feature_columns] for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("feature table contains non-finite model features")
    return values


def _probabilities(
    model: Any, rows: list[dict[str, Any]], feature_columns: tuple[str, ...] = tuple(FEATURE_COLUMNS)
) -> dict[str, np.ndarray]:
    probabilities = model.predict_proba(_matrix(rows, feature_columns))
    return {str(name): probabilities[:, index] for index, name in enumerate(model.classes_)}


def select_thresholds(
    model: Any,
    validation_rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...] = tuple(FEATURE_COLUMNS),
    time_config: dict[str, Any] | None = None,
) -> dict[str, float]:
    if not validation_rows:
        raise ValueError("validation split is empty")
    probabilities = _probabilities(model, validation_rows, feature_columns)
    candidates = [round(value, 2) for value in np.arange(0.15, 0.851, 0.05)]
    thresholds: dict[str, float] = {}
    for event_class in EVENT_CLASSES:
        if event_class not in probabilities:
            raise ValueError(f"training data did not produce class {event_class}")
        ranked = []
        for threshold in candidates:
            metrics = event_metrics_at_threshold(
                validation_rows, probabilities[event_class], event_class, threshold, time_config
            )
            ranked.append((metrics["f1"], metrics["precision"], threshold, metrics["recall"]))
        best = max(ranked, key=lambda item: (item[0], item[1], item[2]))
        thresholds[event_class] = best[2]
    return thresholds


def fit_model(rows: list[dict[str, Any]], model_path: Path, validation_report_path: Path, seed: int = 42) -> dict[str, Any]:
    validate_lineage_splits(rows)
    if any("mos" in row for row in rows):
        raise ValueError("anomaly classifier input must not contain MOS labels")
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    if not train_rows or not val_rows:
        raise ValueError("train and val splits must both be non-empty")
    labels = {str(row.get("label")) for row in train_rows}
    if not set(CLASS_NAMES).issubset(labels):
        raise ValueError("train split must contain normal and every anomaly class")
    if any(row.get("label") not in CLASS_NAMES for row in train_rows + val_rows):
        raise ValueError("feature table contains an unsupported label")

    model = RandomForestClassifier(
        n_estimators=160,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(_matrix(train_rows), [row["label"] for row in train_rows])
    thresholds = select_thresholds(model, val_rows)
    validation_probabilities = _probabilities(model, val_rows)
    validation_metrics = {}
    validation_event_metrics = {}
    for event_class in EVENT_CLASSES:
        actual = np.asarray([row["label"] == event_class for row in val_rows], dtype=bool)
        predicted = validation_probabilities[event_class] >= thresholds[event_class]
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual, predicted, average="binary", zero_division=0
        )
        validation_metrics[event_class] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "threshold": thresholds[event_class],
            "windowCount": int(actual.sum()),
        }
        validation_event_metrics[event_class] = event_metrics_at_threshold(
            val_rows, validation_probabilities[event_class], event_class, thresholds[event_class]
        )
    artifact = {
        "model": model,
        "modelVersion": MODEL_VERSION,
        "featureColumns": FEATURE_COLUMNS,
        "featureProfile": "event-features-v1",
        "eventClasses": EVENT_CLASSES,
        "thresholds": thresholds,
        "mergeMaxGapS": 0.5,
        "minimumEventDurationS": 0.5,
        "timePostprocess": dict(DEFAULT_TIME_POSTPROCESS),
        "trainingSeed": seed,
        "trainedOn": "train",
        "selectedOn": "val",
        "frozen": True,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    report = {
        "schemaVersion": "task3.event-validation/1.0",
        "modelVersion": MODEL_VERSION,
        "trainWindowCount": len(train_rows),
        "validationWindowCount": len(val_rows),
        "thresholds": thresholds,
        "validationWindowMetrics": validation_metrics,
        "validationEventMetrics": validation_event_metrics,
        "testUsedForSelection": False,
    }
    validation_report_path.parent.mkdir(parents=True, exist_ok=True)
    validation_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _artifact_feature_rows(rows: list[dict[str, Any]], artifact: dict[str, Any]) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    profile = str(artifact.get("featureProfile") or "event-features-v1")
    expected = EVENT_FEATURE_PROFILE_COLUMNS.get(profile)
    declared = tuple(artifact.get("featureColumns") or ())
    if expected is None or declared != expected:
        raise ValueError("event model must declare a supported feature profile and feature order")
    if profile == "event-features-v1":
        return [dict(row) for row in rows], expected
    summaries = temporal_features(rows)
    relative = relative_sharpness_features(rows) if profile in {"event-features-v3", "event-features-v4", "event-features-v5", "event-features-v6", "event-features-v7", "event-features-v8", "event-features-v9", "event-features-v10", "event-features-v11", "event-features-v12", "event-features-v13"} else None
    quantiles = relative_sharpness_quantile_features(rows) if profile in {"event-features-v5", "event-features-v6", "event-features-v7", "event-features-v8", "event-features-v9", "event-features-v10", "event-features-v11", "event-features-v12", "event-features-v13"} else None
    normalized = normalized_sharpness_features(rows) if profile in {"event-features-v6", "event-features-v7", "event-features-v8", "event-features-v9", "event-features-v10", "event-features-v11", "event-features-v12", "event-features-v13"} else None
    exposure = relative_exposure_features(rows) if profile in {"event-features-v8", "event-features-v9", "event-features-v11", "event-features-v12", "event-features-v13"} else None
    if profile in {"event-features-v9", "event-features-v11", "event-features-v12", "event-features-v13"}:
        exposure = relative_exposure_features(rows)
        edge = relative_edge_features(rows)
    else:
        edge = None
    context = context_sharpness_features(rows) if profile in {"event-features-v12", "event-features-v13"} else None
    wide_context = wide_context_sharpness_features(rows) if profile == "event-features-v13" else None
    enriched = []
    for index, row in enumerate(rows):
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
    return enriched, expected


def predict_intervals(rows: list[dict[str, Any]], artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if artifact.get("frozen") is not True:
        raise ValueError("event model must be frozen")
    feature_rows, feature_columns = _artifact_feature_rows(rows, artifact)
    time_config = artifact.get("timePostprocess")
    if time_config is None:
        time_config = {
            "maxGapS": float(artifact.get("mergeMaxGapS", 0.5)),
            "confirmationWindows": 1,
            "recoveryS": float(artifact.get("mergeMaxGapS", 0.5)),
            "minimumEventDurationS": float(artifact.get("minimumEventDurationS", 0.5)),
        }
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        by_video[str(row["videoId"])].append(row)
    result: dict[str, dict[str, Any]] = {}
    for video_id, video_rows in by_video.items():
        started = time.perf_counter()
        probabilities = _probabilities(artifact["model"], video_rows, feature_columns)
        result[video_id] = {"__processingTimeMs": 0}
        for event_class in EVENT_CLASSES:
            threshold = float(artifact["thresholds"][event_class])
            candidates = [
                {**row, "eventClass": event_class, "probability": float(probabilities[event_class][index])}
                for index, row in enumerate(video_rows)
                if probabilities[event_class][index] >= threshold
            ]
            merged = merge_windows(
                candidates,
                max_gap_s=float(time_config["maxGapS"]),
                min_duration_s=float(time_config["minimumEventDurationS"]),
                confirmation_windows=int(time_config["confirmationWindows"]),
                recovery_s=float(time_config["recoveryS"]),
            )
            result[video_id][event_class] = merged
        result[video_id]["__processingTimeMs"] = max(1, round((time.perf_counter() - started) * 1000))
    return result


def _event_counts(rows: list[dict[str, Any]], predictions: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        metadata.setdefault(
            str(row["videoId"]),
            {"split": row["split"], "isSynthetic": bool(row["isSynthetic"]), "event": row.get("groundTruthEvent"), "motion": []},
        )["motion"].append(float(row["frame_diff"]))

    counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in EVENT_CLASSES}
    freeze_long_tp = 0
    freeze_long_fn = 0
    by_video_gt = {
        video_id: meta["event"]
        for video_id, meta in metadata.items()
        if meta["split"] == "test" and meta["event"] is not None
    }
    for video_id, event in by_video_gt.items():
        event_class = event["type"]
        truth = (float(event["startTimeS"]), float(event["endTimeS"]))
        candidates = predictions.get(video_id, {}).get(event_class, [])
        best_index = max(
            range(len(candidates)),
            key=lambda index: interval_iou(
                truth, (candidates[index]["startTimeS"], candidates[index]["endTimeS"])
            ),
            default=None,
        )
        best_iou = 0.0 if best_index is None else interval_iou(
            truth, (candidates[best_index]["startTimeS"], candidates[best_index]["endTimeS"])
        )
        if best_iou >= 0.5:
            counts[event_class]["tp"] += 1
            if event_class == "freeze_suspected" and truth[1] - truth[0] >= 3.0:
                freeze_long_tp += 1
            extra = len(candidates) - 1
        else:
            counts[event_class]["fn"] += 1
            if event_class == "freeze_suspected" and truth[1] - truth[0] >= 3.0:
                freeze_long_fn += 1
            extra = len(candidates)
        counts[event_class]["fp"] += max(0, extra)

    for video_id, meta in metadata.items():
        if meta["split"] != "test":
            continue
        event = meta["event"]
        for event_class in EVENT_CLASSES:
            if event is not None and event["type"] == event_class:
                continue
            counts[event_class]["fp"] += len(predictions.get(video_id, {}).get(event_class, []))

    per_class = {}
    for event_class, values in counts.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[event_class] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    macro_f1 = float(np.mean([metric["f1"] for metric in per_class.values()]))

    original_negatives = [
        (video_id, float(np.mean(meta["motion"])))
        for video_id, meta in metadata.items()
        if meta["split"] == "test" and not meta["isSynthetic"] and meta["event"] is None
    ]
    low_motion = [video_id for video_id, _ in sorted(original_negatives, key=lambda item: item[1])[:30]]
    if len(low_motion) < 30:
        raise ValueError("test split must include at least 30 original negative videos for low-motion audit")
    low_motion_false_alarms = sum(
        bool(predictions.get(video_id, {}).get("freeze_suspected")) for video_id in low_motion
    )
    return {
        "perClassEventMetrics": per_class,
        "macroF1": macro_f1,
        "freezeRecallAtLeast3s": (
            freeze_long_tp / (freeze_long_tp + freeze_long_fn) if freeze_long_tp + freeze_long_fn else None
        ),
        "freezeEventsAtLeast3s": freeze_long_tp + freeze_long_fn,
        "lowMotionNegativeVideoCount": len(low_motion),
        "lowMotionNegativeFreezeFalseAlarms": low_motion_false_alarms,
        "lowMotionNegativeFreezeFalseAlarmRate": low_motion_false_alarms / len(low_motion),
        "lowMotionNegativeVideoIds": low_motion,
    }


def build_contract_results(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
    quality_artifact: dict[str, Any],
    event_model_version: str,
) -> list[dict[str, Any]]:
    from shared_contracts.common import DataQuality, Provenance, SourceModality, TimeQuality
    from shared_contracts.quality import AnomalyInterval, VideoQualityResult

    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "test":
            by_video[str(row["videoId"])].append(row)
    quality_model = quality_artifact["model"]
    quality_started = time.perf_counter()
    quality_features = []
    video_ids = list(by_video)
    for video_id in video_ids:
        video_rows = by_video[video_id]
        means = {name: float(np.mean([float(row[name]) for row in video_rows])) for name in FEATURE_COLUMNS}
        quality_features.append(
            [
                means["lap_var"],
                means["luma_mean"],
                means["dark_ratio"],
                means["bright_ratio"],
                float(np.mean([float(row["contrast"]) for row in video_rows])),
                means["frame_diff"],
            ]
        )
    raw_scores = quality_model.predict(np.asarray(quality_features, dtype=np.float64))
    quality_ms_per_video = max(0, round((time.perf_counter() - quality_started) * 1000 / max(1, len(video_ids))))
    normalization = quality_artifact["qualityScoreNormalization"]
    spread = float(normalization["mosMax"]) - float(normalization["mosMin"])
    normalized = np.clip((raw_scores - float(normalization["mosMin"])) / spread, 0.0, 1.0)
    timestamp = datetime.now(timezone.utc)
    stamp = timestamp.isoformat().replace("+00:00", "Z")
    output = []
    for video_id, score in zip(video_ids, normalized):
        video_rows = by_video[video_id]
        first = video_rows[0]
        anomalies = []
        for event_class in EVENT_CLASSES:
            for interval in predictions.get(video_id, {}).get(event_class, []):
                anomalies.append(
                    AnomalyInterval(
                        type=event_class,
                        startTimeS=interval["startTimeS"],
                        endTimeS=interval["endTimeS"],
                        status="suspected",
                    )
                )
        result = VideoQualityResult(
            jobId=f"offline-{video_id}",
            videoId=video_id,
            provenance=Provenance(
                source=SourceModality.EO,
                deviceId="konvid-1k-offline",
                observedAt=stamp,
                receivedAt=stamp,
                dataQuality=DataQuality.DEGRADED if anomalies else DataQuality.OK,
                timeQuality=TimeQuality.UNKNOWN,
                modelVersion=f"{event_model_version}+{quality_artifact.get('modelVersion', 'quality-model')}",
            ),
            channelType="visible",
            isSynthetic=bool(first["isSynthetic"]),
            limitations=[
                "KoNViD 原始采集时间不可用；observedAt 记录离线评估时间，不代表采集时间。",
                "离线处理基于预提取特征；processingTimeMs 不含解码和特征提取，不能用于实时性能验收。",
                "受控合成异常结果不能替代真实设备故障验收。",
            ],
            scoreStatus="scored",
            qualityScore=float(score),
            anomalies=anomalies,
            processingTimeMs=int(predictions.get(video_id, {}).get("__processingTimeMs", 1)) + quality_ms_per_video,
        )
        output.append(result.model_dump(mode="json", by_alias=True))
    return output


def evaluate_test(
    rows: list[dict[str, Any]],
    artifact: dict[str, Any],
    quality_artifact: dict[str, Any],
    *,
    input_sha256: str | None = None,
    model_sha256: str | None = None,
    quality_model_sha256: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validate_lineage_splits(rows)
    if artifact.get("frozen") is not True or artifact.get("selectedOn") != "val":
        raise ValueError("test evaluation requires a frozen model selected on val")
    test_rows = [row for row in rows if row["split"] == "test"]
    if not test_rows:
        raise ValueError("feature table contains no test split")
    if any(row.get("label") not in CLASS_NAMES for row in test_rows):
        raise ValueError("test feature table has an unknown label")
    predictions = predict_intervals(test_rows, artifact)
    event_metrics = _event_counts(test_rows, predictions)
    contract_results = build_contract_results(test_rows, predictions, quality_artifact, artifact["modelVersion"])
    report = {
        "schemaVersion": "task3.controlled-event-test/1.0",
        "dataset": "KoNViD-1k-controlled-distortions",
        "modelVersion": artifact["modelVersion"],
        "trainUsedForFit": True,
        "validationUsedForSelection": True,
        "testUsedForSelection": False,
        "testWindowCount": len(test_rows),
        "testVideoCount": len({row["videoId"] for row in test_rows}),
        "inputSha256": input_sha256,
        "modelSha256": model_sha256,
        "qualityModelSha256": quality_model_sha256,
        **event_metrics,
        "acceptanceTargets": {
            "macroF1": 0.90,
            "freezeRecallAtLeast3s": 0.95,
            "lowMotionNegativeFreezeFalseAlarmRateMaximum": 0.05,
        },
        "limitations": [
            "结果只代表基于 KoNViD 可见光视频生成的受控异常，不代表真实摄像机故障表现。",
            "低运动负例由测试 split 原始视频的帧差排序选出 30 段，未经人工静止场景标注。",
        ],
    }
    return report, contract_results


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit", help="fit on train, select thresholds on val, and freeze the model")
    fit_parser.add_argument("--features", type=Path, required=True)
    fit_parser.add_argument("--model-out", type=Path, required=True)
    fit_parser.add_argument("--validation-report", type=Path, required=True)
    fit_parser.add_argument("--seed", type=int, default=42)
    eval_parser = subparsers.add_parser("evaluate", help="evaluate one frozen model on test")
    eval_parser.add_argument("--features", type=Path, required=True)
    eval_parser.add_argument("--model", type=Path, required=True)
    eval_parser.add_argument("--quality-model", type=Path, required=True)
    eval_parser.add_argument("--report", type=Path, required=True)
    eval_parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = read_rows(args.features)
    if args.command == "fit":
        report = fit_model(rows, args.model_out, args.validation_report, args.seed)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.report.exists() or args.results.exists():
        parser.error("refusing to overwrite test evaluation outputs")
    artifact = joblib.load(args.model)
    quality_artifact = joblib.load(args.quality_model)
    report, results = evaluate_test(
        rows,
        artifact,
        quality_artifact,
        input_sha256=_sha256(args.features),
        model_sha256=_sha256(args.model),
        quality_model_sha256=_sha256(args.quality_model),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_jsonl(results, args.results)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
