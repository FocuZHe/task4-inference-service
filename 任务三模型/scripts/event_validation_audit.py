"""Validation-only, source-grouped bootstrap statistics for event candidates.

The bootstrap unit is ``sourceVideoId`` rather than an individual window or
synthetic variant.  This keeps all variants derived from one official source
video together and prevents validation intervals from overstating the amount
of independent evidence.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from diagnose_events import (
    EVENT_CLASSES,
    event_metrics_at_threshold,
    merge_windows,
)


def _finite_probability_vector(
    probabilities: Mapping[str, Sequence[float] | np.ndarray],
    event_class: str,
    indices: list[int],
) -> np.ndarray:
    if event_class not in probabilities:
        raise ValueError(f"missing probability vector for {event_class}")
    values = np.asarray(probabilities[event_class], dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"probabilities for {event_class} must be one-dimensional")
    if not np.isfinite(values[indices]).all():
        raise ValueError("event probabilities must be finite")
    return values


def _validate_inputs(
    rows: list[dict[str, Any]],
    probabilities: Mapping[str, Sequence[float] | np.ndarray],
    thresholds: Mapping[str, float],
    samples: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, list[int]]]:
    if not isinstance(samples, int) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    val_indices = [index for index, row in enumerate(rows) if row.get("split") == "val"]
    if not val_indices:
        raise ValueError("validation split is empty")

    val_rows: list[dict[str, Any]] = []
    for index in val_indices:
        row = rows[index]
        source_id = str(row.get("sourceVideoId") or "").strip()
        video_id = str(row.get("videoId") or "").strip()
        if not source_id or not video_id:
            raise ValueError("validation rows require sourceVideoId and videoId")
        val_rows.append(dict(row))

    val_probabilities: dict[str, np.ndarray] = {}
    for event_class in EVENT_CLASSES:
        values = _finite_probability_vector(probabilities, event_class, val_indices)
        if len(values) != len(rows):
            raise ValueError(f"probabilities for {event_class} must match rows")
        threshold = thresholds.get(event_class)
        if threshold is None or not math.isfinite(float(threshold)):
            raise ValueError(f"threshold for {event_class} must be finite")
        val_probabilities[event_class] = values[val_indices]

    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(val_rows):
        groups[str(row["sourceVideoId"])].append(index)
    return val_rows, val_probabilities, dict(groups)


def _metrics(
    rows: list[dict[str, Any]],
    probabilities: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
    time_config: Mapping[str, Any],
) -> tuple[dict[str, dict[str, int]], float, float | None, float | None, int]:
    event_counts: dict[str, dict[str, int]] = {}
    f1_values: list[float] = []
    for event_class in EVENT_CLASSES:
        result = event_metrics_at_threshold(
            rows,
            probabilities[event_class],
            event_class,
            float(thresholds[event_class]),
            dict(time_config),
        )
        event_counts[event_class] = {
            "tp": int(result["tp"]),
            "fp": int(result["fp"]),
            "fn": int(result["fn"]),
        }
        f1_values.append(float(result["f1"]))
    macro_f1 = float(np.mean(f1_values))

    freeze_rows: list[dict[str, Any]] = []
    freeze_indices: list[int] = []
    for index, row in enumerate(rows):
        event = row.get("groundTruthEvent")
        if (
            isinstance(event, dict)
            and event.get("type") == "freeze_suspected"
            and float(event["endTimeS"]) - float(event["startTimeS"]) >= 3.0
        ):
            freeze_rows.append(row)
            freeze_indices.append(index)
    freeze_recall: float | None = None
    if freeze_rows:
        freeze_result = event_metrics_at_threshold(
            freeze_rows,
            probabilities["freeze_suspected"][freeze_indices],
            "freeze_suspected",
            float(thresholds["freeze_suspected"]),
            dict(time_config),
        )
        freeze_recall = float(freeze_result["recall"])

    negatives: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if bool(row.get("isSynthetic")) or row.get("groundTruthEvent") is not None:
            continue
        candidate = dict(row)
        candidate["probability"] = float(probabilities["freeze_suspected"][index])
        negatives[str(row["videoId"])].append(candidate)
    ranked = sorted(
        negatives.items(),
        key=lambda item: float(np.mean([float(row["frame_diff"]) for row in item[1]])),
    )[:30]
    false_alarms = 0
    for _, video_rows in ranked:
        predicted = [
            {**row, "eventClass": "freeze_suspected"}
            for row in video_rows
            if row["probability"] >= float(thresholds["freeze_suspected"])
        ]
        merged = merge_windows(
            predicted,
            max_gap_s=float(time_config["maxGapS"]),
            min_duration_s=float(time_config["minimumEventDurationS"]),
            confirmation_windows=int(time_config["confirmationWindows"]),
            recovery_s=float(time_config["recoveryS"]),
        )
        false_alarms += bool(merged)
    low_motion_rate = false_alarms / len(ranked) if ranked else None
    return event_counts, macro_f1, freeze_recall, low_motion_rate, len(ranked)


def _copy_group_sample(
    rows: list[dict[str, Any]],
    probabilities: Mapping[str, np.ndarray],
    group_indices: list[int],
    replicate: int,
) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    copied_rows: list[dict[str, Any]] = []
    copied_probabilities = {event_class: [] for event_class in EVENT_CLASSES}
    for index in group_indices:
        row = dict(rows[index])
        # Duplicate source groups need distinct video IDs so event matching
        # counts each bootstrap draw instead of merging repeated draws.
        row["videoId"] = f"{row['videoId']}__bootstrap_{replicate}"
        copied_rows.append(row)
        for event_class in EVENT_CLASSES:
            copied_probabilities[event_class].append(float(probabilities[event_class][index]))
    return copied_rows, copied_probabilities


def source_group_bootstrap(
    rows: list[dict[str, Any]],
    probabilities: Mapping[str, Sequence[float] | np.ndarray],
    thresholds: Mapping[str, float],
    time_config: Mapping[str, Any],
    *,
    samples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Return point estimates and percentile intervals using val source groups.

    Test rows are filtered before probability validation and are therefore
    ignored even when their features or probabilities are non-finite.
    """
    val_rows, val_probabilities, groups = _validate_inputs(rows, probabilities, thresholds, samples)
    source_ids = sorted(groups)
    point_counts, point_macro, point_freeze, point_low_motion, negative_count = _metrics(
        val_rows, val_probabilities, thresholds, time_config
    )

    rng = np.random.default_rng(seed)
    macro_values: list[float] = []
    freeze_values: list[float] = []
    low_motion_values: list[float] = []
    for draw in range(samples):
        selected_sources = rng.choice(source_ids, size=len(source_ids), replace=True)
        sampled_rows: list[dict[str, Any]] = []
        sampled_probabilities = {event_class: [] for event_class in EVENT_CLASSES}
        for replicate, source_id in enumerate(selected_sources):
            group_rows, group_probabilities = _copy_group_sample(
                val_rows,
                val_probabilities,
                groups[str(source_id)],
                replicate,
            )
            sampled_rows.extend(group_rows)
            for event_class in EVENT_CLASSES:
                sampled_probabilities[event_class].extend(group_probabilities[event_class])
        _, macro, freeze, low_motion, _ = _metrics(
            sampled_rows,
            {name: np.asarray(values, dtype=np.float64) for name, values in sampled_probabilities.items()},
            thresholds,
            time_config,
        )
        macro_values.append(macro)
        if freeze is not None:
            freeze_values.append(freeze)
        if low_motion is not None:
            low_motion_values.append(low_motion)

    def interval(values: list[float]) -> list[float | None]:
        if not values:
            return [None, None]
        low, high = np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.975])
        return [float(low), float(high)]

    return {
        "sourceGroupCount": len(source_ids),
        "videoCount": len({str(row["videoId"]) for row in val_rows}),
        "resamplingUnit": "sourceVideoId",
        "seed": seed,
        "samples": samples,
        "eventCounts": point_counts,
        "pointEstimates": {
            "macroF1": point_macro,
            "freezeRecallAtLeast3s": point_freeze,
            "lowMotionNegativeFreezeFalseAlarmRate": point_low_motion,
            "lowMotionNegativeVideoCount": negative_count,
        },
        "confidenceIntervals": {
            "level": 0.95,
            "method": "percentile_bootstrap_by_sourceVideoId",
            "macroF1": interval(macro_values),
            "freezeRecallAtLeast3s": interval(freeze_values),
            "lowMotionNegativeFreezeFalseAlarmRate": interval(low_motion_values),
        },
    }
