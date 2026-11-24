"""Extract short-window visual features and labels for controlled anomaly experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

cv2.setNumThreads(1)


FEATURE_COLUMNS = (
    "luma_mean",
    "luma_std",
    "dark_ratio",
    "bright_ratio",
    "lap_var",
    "frame_diff",
)
CLASS_NAMES = ("normal", "blur", "dark", "overexposed", "freeze_suspected")
TEMPORAL_FEATURE_COLUMNS = (
    "frame_diff_delta_mean",
    "frame_diff_median",
    "frame_diff_p10",
    "low_frame_diff_candidate_duration_s",
    "luma_delta_mean",
)
TEMPORAL_EVIDENCE_FEATURE_COLUMNS = (
    "frame_diff_low_fraction",
    "frame_diff_zero_fraction",
    "frame_diff_low_run_count",
)
RELATIVE_SHARPNESS_FEATURE_COLUMNS = (
    "lap_log_relative_median",
    "lap_log_drop_previous",
    "lap_log_rise_next",
)
SHARPNESS_RUN_FEATURE_COLUMNS = (
    "lap_low_run_duration_s_025",
    "lap_low_run_duration_s_050",
)
QUANTILE_SHARPNESS_FEATURE_COLUMNS = (
    "lap_log_relative_p75",
    "lap_log_relative_p90",
    "lap_percentile_rank",
    "lap_low_rank_run_duration_s",
)
NORMALIZED_SHARPNESS_FEATURE_COLUMNS = (
    "lap_log_contrast_normalized",
)
RELATIVE_EXPOSURE_FEATURE_COLUMNS = (
    "luma_log_relative_median",
    "luma_percentile_rank",
    "contrast_log_relative_median",
    "contrast_percentile_rank",
    "luma_low_run_duration_s",
    "luma_high_run_duration_s",
)
RELATIVE_EDGE_FEATURE_COLUMNS = (
    "gradient_log_relative_median",
    "gradient_percentile_rank",
    "lap_abs_log_relative_median",
    "lap_abs_percentile_rank",
)
GRADIENT_SHARPNESS_FEATURE_COLUMNS = (
    "gradient_energy",
    "gradient_p90",
    "lap_abs_mean",
    "lap_abs_contrast_ratio",
    "high_frequency_energy_ratio",
)
CONTEXT_SHARPNESS_FEATURE_COLUMNS = (
    "lap_log_context_drop",
    "lap_log_context_rise",
    "gradient_log_context_drop",
    "gradient_log_context_rise",
)
WIDE_CONTEXT_SHARPNESS_FEATURE_COLUMNS = (
    "lap_log_wide_context_drop",
    "lap_log_wide_context_rise",
    "gradient_log_wide_context_drop",
    "gradient_log_wide_context_rise",
)
LOW_FRAME_DIFF_THRESHOLD = 0.5


def frame_sharpness_features(gray: np.ndarray) -> dict[str, float]:
    """Measure frame-level edge and high-frequency energy without using labels."""
    image = np.asarray(gray)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("frame sharpness input must be a non-empty grayscale image")
    image_float = image.astype(np.float32, copy=False)
    if not np.isfinite(image_float).all():
        raise ValueError("frame sharpness input contains non-finite values")
    grad_x = cv2.Sobel(image_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_float, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    laplacian = cv2.Laplacian(image_float, cv2.CV_32F)
    lap_abs_mean = float(np.mean(np.abs(laplacian)))
    contrast = float(np.std(image_float))
    values = {
        "gradient_energy": float(np.mean(grad_x * grad_x + grad_y * grad_y)),
        "gradient_p90": float(np.percentile(gradient_magnitude, 90)),
        "lap_abs_mean": lap_abs_mean,
        "lap_abs_contrast_ratio": float(lap_abs_mean / (contrast + 1e-6)),
    }
    high_energy = float(np.mean(np.square(laplacian, dtype=np.float64)))
    image_energy = float(np.mean(np.square(image_float, dtype=np.float64)))
    values["high_frequency_energy_ratio"] = float(high_energy / (high_energy + image_energy + 1e-12))
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("frame sharpness calculation produced non-finite values")
    return values


def select_manifest_records(
    records: list[dict[str, Any]], split: str | None = None, *,
    part_index: int | None = None, part_count: int = 1,
) -> list[dict[str, Any]]:
    """Select records by declared split, preserving order and lineage."""
    if split is not None and split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, test, or None")
    if part_count < 1 or part_index is not None and not 0 <= part_index < part_count:
        raise ValueError("part_index must be within part_count and part_count must be positive")
    for record in records:
        record_split = record.get("split")
        if record_split not in {"train", "val", "test"}:
            raise ValueError(f"manifest record has unsupported split: {record_split!r}")
    selected = records if split is None else [record for record in records if record["split"] == split]
    if part_index is None:
        return selected
    return [record for index, record in enumerate(selected) if index % part_count == part_index]


def temporal_features(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Summarize time-ordered motion and luma changes independently per video."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    metadata: dict[str, tuple[str, str]] = {}
    for row in rows:
        video_id = str(row.get("videoId") or "").strip()
        if not video_id:
            raise ValueError("temporal rows require videoId")
        source_id = str(row.get("sourceVideoId") or "")
        split = str(row.get("split") or "")
        if video_id in metadata:
            previous_source, previous_split = metadata[video_id]
            if source_id != previous_source:
                raise ValueError(f"video {video_id} has conflicting sourceVideoId")
            if split != previous_split:
                raise ValueError(f"video {video_id} has conflicting split")
        else:
            metadata[video_id] = (source_id, split)
        grouped.setdefault(video_id, []).append(row)

    result: dict[str, dict[str, float]] = {}
    for video_id, video_rows in grouped.items():
        try:
            ordered = sorted(video_rows, key=lambda row: float(row["windowStartS"]))
            frame_diffs = np.asarray([float(row["frame_diff"]) for row in ordered], dtype=np.float64)
            lumas = np.asarray([float(row.get("luma_mean", 0.0)) for row in ordered], dtype=np.float64)
            timestamps = np.asarray(
                [float(row.get("windowEndS", row["windowStartS"])) for row in ordered], dtype=np.float64
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("temporal rows require numeric windowStartS, frame_diff, and windowEndS") from exc
        starts = np.asarray([float(row["windowStartS"]) for row in ordered], dtype=np.float64)
        if not np.isfinite(starts).all():
            raise ValueError("temporal rows contain non-finite windowStartS")
        if len(starts) > 1 and np.any(np.diff(starts) == 0):
            raise ValueError(f"video {video_id} has duplicate windowStartS")
        if not np.isfinite(frame_diffs).all() or not np.isfinite(lumas).all() or not np.isfinite(timestamps).all():
            raise ValueError("temporal rows contain non-finite values")
        if any(float(row["windowEndS"]) < float(row["windowStartS"]) for row in ordered if "windowEndS" in row):
            raise ValueError("temporal window intervals must be ordered")
        frame_delta = np.abs(np.diff(frame_diffs))
        luma_delta = np.abs(np.diff(lumas))

        longest_low_run = 0.0
        low_run_count = 0
        run_start: float | None = None
        run_end = 0.0
        for row, frame_diff in zip(ordered, frame_diffs):
            start = float(row["windowStartS"])
            end = float(row.get("windowEndS", start))
            if frame_diff <= LOW_FRAME_DIFF_THRESHOLD:
                if run_start is None:
                    low_run_count += 1
                    run_start = start
                    run_end = end
                elif start <= run_end:
                    run_end = max(run_end, end)
                else:
                    longest_low_run = max(longest_low_run, run_end - run_start)
                    run_start, run_end = start, end
            elif run_start is not None:
                longest_low_run = max(longest_low_run, run_end - run_start)
                run_start = None
        if run_start is not None:
            longest_low_run = max(longest_low_run, run_end - run_start)

        result[video_id] = {
            "frame_diff_delta_mean": float(frame_delta.mean()) if frame_delta.size else 0.0,
            "frame_diff_median": float(np.median(frame_diffs)),
            "frame_diff_p10": float(np.percentile(frame_diffs, 10)),
            "low_frame_diff_candidate_duration_s": float(longest_low_run),
            "luma_delta_mean": float(luma_delta.mean()) if luma_delta.size else 0.0,
            "frame_diff_low_fraction": float(np.mean(frame_diffs <= LOW_FRAME_DIFF_THRESHOLD)),
            "frame_diff_zero_fraction": float(np.mean(frame_diffs == 0.0)),
            "frame_diff_low_run_count": float(low_run_count),
        }
    return result


def relative_sharpness_features(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Add per-window sharpness changes relative to each video's own baseline.

    Calculations are grouped by ``videoId`` and returned in the input order.
    Only timing and ``lap_var`` are used; labels and event metadata are ignored.
    """
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    metadata: dict[str, tuple[str, str]] = {}
    for index, row in enumerate(rows):
        video_id = str(row.get("videoId") or "").strip()
        if not video_id:
            raise ValueError("relative sharpness rows require videoId")
        try:
            start = float(row["windowStartS"])
            lap_var = float(row["lap_var"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("relative sharpness rows require numeric windowStartS and lap_var") from exc
        if not math.isfinite(start) or not math.isfinite(lap_var):
            raise ValueError("relative sharpness rows contain non-finite values")
        if lap_var < 0:
            raise ValueError("relative sharpness lap_var must be non-negative")
        source_id = str(row.get("sourceVideoId") or "")
        split = str(row.get("split") or "")
        if video_id in metadata and metadata[video_id] != (source_id, split):
            if metadata[video_id][0] != source_id:
                raise ValueError(f"video {video_id} has conflicting sourceVideoId")
            raise ValueError(f"video {video_id} has conflicting split")
        metadata[video_id] = (source_id, split)
        grouped.setdefault(video_id, []).append((index, row))

    epsilon = 1e-6
    output = [
        {name: 0.0 for name in RELATIVE_SHARPNESS_FEATURE_COLUMNS + SHARPNESS_RUN_FEATURE_COLUMNS}
        for _ in rows
    ]
    for video_id, indexed_rows in grouped.items():
        ordered = sorted(indexed_rows, key=lambda item: float(item[1]["windowStartS"]))
        starts = [float(row["windowStartS"]) for _, row in ordered]
        if len(starts) > 1 and any(right == left for left, right in zip(starts, starts[1:])):
            raise ValueError(f"video {video_id} has duplicate windowStartS")
        values = np.asarray([float(row["lap_var"]) for _, row in ordered], dtype=np.float64)
        baseline = float(np.median(values))
        log_values = np.log(values + epsilon)
        log_baseline = math.log(baseline + epsilon)
        for position, (original_index, _) in enumerate(ordered):
            previous = log_values[position - 1] if position else log_values[position]
            following = log_values[position + 1] if position + 1 < len(log_values) else log_values[position]
            output[original_index].update({
                "lap_log_relative_median": float(log_values[position] - log_baseline),
                "lap_log_drop_previous": float(max(0.0, previous - log_values[position])),
                "lap_log_rise_next": float(max(0.0, following - log_values[position])),
            })
        for ratio, feature_name in ((0.25, "lap_low_run_duration_s_025"), (0.50, "lap_low_run_duration_s_050")):
            active = values <= (baseline * ratio)
            runs: list[tuple[int, int, float]] = []
            run_start_index: int | None = None
            run_end_s = 0.0
            for position, (original_index, row) in enumerate(ordered):
                start_s = float(row["windowStartS"])
                end_s = float(row.get("windowEndS", start_s + 0.5))
                if active[position]:
                    if run_start_index is None:
                        run_start_index = position
                        run_end_s = end_s
                    elif start_s <= run_end_s:
                        run_end_s = max(run_end_s, end_s)
                    else:
                        runs.append((run_start_index, position - 1, run_end_s))
                        run_start_index = position
                        run_end_s = end_s
                elif run_start_index is not None:
                    runs.append((run_start_index, position - 1, run_end_s))
                    run_start_index = None
            if run_start_index is not None:
                runs.append((run_start_index, len(ordered) - 1, run_end_s))
            for first, last, end_s in runs:
                start_s = float(ordered[first][1]["windowStartS"])
                duration_s = max(0.0, end_s - start_s)
                for position in range(first, last + 1):
                    output[ordered[position][0]][feature_name] = duration_s
    return output


def context_sharpness_features(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Compare each window with up to two neighbouring windows in its video.

    The comparison is deliberately local and label-free.  Grouping by video and
    validating lineage prevents a neighbouring window from another recording or
    split from becoming an accidental reference.
    """
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    metadata: dict[str, tuple[str, str]] = {}
    for index, row in enumerate(rows):
        video_id = str(row.get("videoId") or "").strip()
        if not video_id:
            raise ValueError("context sharpness rows require videoId")
        try:
            start = float(row["windowStartS"])
            lap_var = float(row["lap_var"])
            gradient_energy = float(row["gradient_energy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "context sharpness rows require numeric windowStartS, lap_var, and gradient_energy"
            ) from exc
        if not all(math.isfinite(value) for value in (start, lap_var, gradient_energy)):
            raise ValueError("context sharpness rows contain non-finite values")
        if lap_var < 0 or gradient_energy < 0:
            raise ValueError("context sharpness energies must be non-negative")
        lineage = (str(row.get("sourceVideoId") or ""), str(row.get("split") or ""))
        if video_id in metadata:
            previous_source, previous_split = metadata[video_id]
            if lineage[0] != previous_source:
                raise ValueError(f"video {video_id} has conflicting sourceVideoId")
            if lineage[1] != previous_split:
                raise ValueError(f"video {video_id} has conflicting split")
        else:
            metadata[video_id] = lineage
        grouped.setdefault(video_id, []).append((index, row))

    output = [{name: 0.0 for name in CONTEXT_SHARPNESS_FEATURE_COLUMNS} for _ in rows]
    for video_id, indexed_rows in grouped.items():
        ordered = sorted(indexed_rows, key=lambda item: float(item[1]["windowStartS"]))
        starts = [float(row["windowStartS"]) for _, row in ordered]
        if len(starts) > 1 and any(right == left for left, right in zip(starts, starts[1:])):
            raise ValueError(f"video {video_id} has duplicate windowStartS")
        lap_logs = np.log1p(np.asarray([float(row["lap_var"]) for _, row in ordered], dtype=np.float64))
        gradient_logs = np.log1p(np.asarray([float(row["gradient_energy"]) for _, row in ordered], dtype=np.float64))
        for position, (original_index, _) in enumerate(ordered):
            previous = slice(max(0, position - 2), position)
            following = slice(position + 1, min(len(ordered), position + 3))
            current_lap = float(lap_logs[position])
            current_gradient = float(gradient_logs[position])
            previous_lap = float(np.median(lap_logs[previous])) if position else current_lap
            following_lap = float(np.median(lap_logs[following])) if position + 1 < len(ordered) else current_lap
            previous_gradient = float(np.median(gradient_logs[previous])) if position else current_gradient
            following_gradient = float(np.median(gradient_logs[following])) if position + 1 < len(ordered) else current_gradient
            output[original_index].update({
                "lap_log_context_drop": max(0.0, previous_lap - current_lap),
                "lap_log_context_rise": max(0.0, following_lap - current_lap),
                "gradient_log_context_drop": max(0.0, previous_gradient - current_gradient),
                "gradient_log_context_rise": max(0.0, following_gradient - current_gradient),
            })
    return output


def wide_context_sharpness_features(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Compare each window with sharpness outside its two nearest neighbours."""
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    metadata: dict[str, tuple[str, str]] = {}
    for index, row in enumerate(rows):
        video_id = str(row.get("videoId") or "").strip()
        if not video_id:
            raise ValueError("wide context sharpness rows require videoId")
        try:
            start = float(row["windowStartS"])
            lap_var = float(row["lap_var"])
            gradient_energy = float(row["gradient_energy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "wide context sharpness rows require numeric windowStartS, lap_var, and gradient_energy"
            ) from exc
        if not all(math.isfinite(value) for value in (start, lap_var, gradient_energy)):
            raise ValueError("wide context sharpness rows contain non-finite values")
        if lap_var < 0 or gradient_energy < 0:
            raise ValueError("wide context sharpness energies must be non-negative")
        lineage = (str(row.get("sourceVideoId") or ""), str(row.get("split") or ""))
        if video_id in metadata and metadata[video_id] != lineage:
            if metadata[video_id][0] != lineage[0]:
                raise ValueError(f"video {video_id} has conflicting sourceVideoId")
            raise ValueError(f"video {video_id} has conflicting split")
        metadata[video_id] = lineage
        grouped.setdefault(video_id, []).append((index, row))

    output = [{name: 0.0 for name in WIDE_CONTEXT_SHARPNESS_FEATURE_COLUMNS} for _ in rows]
    for video_id, indexed_rows in grouped.items():
        ordered = sorted(indexed_rows, key=lambda item: float(item[1]["windowStartS"]))
        starts = [float(row["windowStartS"]) for _, row in ordered]
        if len(starts) > 1 and any(right == left for left, right in zip(starts, starts[1:])):
            raise ValueError(f"video {video_id} has duplicate windowStartS")
        lap_logs = np.log1p(np.asarray([float(row["lap_var"]) for _, row in ordered], dtype=np.float64))
        gradient_logs = np.log1p(
            np.asarray([float(row["gradient_energy"]) for _, row in ordered], dtype=np.float64)
        )
        for position, (original_index, _) in enumerate(ordered):
            previous = slice(max(0, position - 5), max(0, position - 2))
            following = slice(min(len(ordered), position + 3), min(len(ordered), position + 6))
            current_lap = float(lap_logs[position])
            current_gradient = float(gradient_logs[position])
            previous_lap = float(np.median(lap_logs[previous])) if previous.start < previous.stop else current_lap
            following_lap = float(np.median(lap_logs[following])) if following.start < following.stop else current_lap
            previous_gradient = float(np.median(gradient_logs[previous])) if previous.start < previous.stop else current_gradient
            following_gradient = float(np.median(gradient_logs[following])) if following.start < following.stop else current_gradient
            output[original_index].update({
                "lap_log_wide_context_drop": max(0.0, previous_lap - current_lap),
                "lap_log_wide_context_rise": max(0.0, following_lap - current_lap),
                "gradient_log_wide_context_drop": max(0.0, previous_gradient - current_gradient),
                "gradient_log_wide_context_rise": max(0.0, following_gradient - current_gradient),
            })
    return output


def relative_sharpness_quantile_features(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Add robust within-video sharpness quantile and low-rank-run features.

    The calculation uses only ``videoId``, timing, and ``lap_var``.  Mid-ranks
    make tied low-sharpness windows deterministic without consulting labels.
    """
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    metadata: dict[str, tuple[str, str]] = {}
    for index, row in enumerate(rows):
        video_id = str(row.get("videoId") or "").strip()
        if not video_id:
            raise ValueError("quantile sharpness rows require videoId")
        try:
            start = float(row["windowStartS"])
            lap_var = float(row["lap_var"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("quantile sharpness rows require numeric windowStartS and lap_var") from exc
        if not math.isfinite(start) or not math.isfinite(lap_var):
            raise ValueError("quantile sharpness rows contain non-finite values")
        if lap_var < 0:
            raise ValueError("quantile sharpness lap_var must be non-negative")
        lineage = (str(row.get("sourceVideoId") or ""), str(row.get("split") or ""))
        if video_id in metadata:
            previous_source, previous_split = metadata[video_id]
            if lineage[0] != previous_source:
                raise ValueError(f"video {video_id} has conflicting sourceVideoId")
            if lineage[1] != previous_split:
                raise ValueError(f"video {video_id} has conflicting split")
        else:
            metadata[video_id] = lineage
        grouped.setdefault(video_id, []).append((index, row))

    epsilon = 1e-6
    output = [{name: 0.0 for name in QUANTILE_SHARPNESS_FEATURE_COLUMNS} for _ in rows]
    for video_id, indexed_rows in grouped.items():
        ordered = sorted(indexed_rows, key=lambda item: float(item[1]["windowStartS"]))
        starts = [float(row["windowStartS"]) for _, row in ordered]
        if len(starts) > 1 and any(right == left for left, right in zip(starts, starts[1:])):
            raise ValueError(f"video {video_id} has duplicate windowStartS")
        values = np.asarray([float(row["lap_var"]) for _, row in ordered], dtype=np.float64)
        sorted_values = np.sort(values)
        count = len(values)
        p75 = float(np.percentile(values, 75))
        p90 = float(np.percentile(values, 90))
        lower = np.searchsorted(sorted_values, values, side="left")
        upper = np.searchsorted(sorted_values, values, side="right")
        ranks = (lower + upper) / (2.0 * count)

        active = ranks <= 0.25
        run_lengths = np.zeros(count, dtype=np.float64)
        run_start: int | None = None
        run_end_s = 0.0
        runs: list[tuple[int, int, float]] = []
        for position, (_, row) in enumerate(ordered):
            start_s = float(row["windowStartS"])
            end_s = float(row.get("windowEndS", start_s + 0.5))
            if active[position]:
                if run_start is None:
                    run_start, run_end_s = position, end_s
                elif start_s <= run_end_s:
                    run_end_s = max(run_end_s, end_s)
                else:
                    runs.append((run_start, position - 1, run_end_s))
                    run_start, run_end_s = position, end_s
            elif run_start is not None:
                runs.append((run_start, position - 1, run_end_s))
                run_start = None
        if run_start is not None:
            runs.append((run_start, count - 1, run_end_s))
        for first, last, end_s in runs:
            duration_s = max(0.0, end_s - float(ordered[first][1]["windowStartS"]))
            run_lengths[first : last + 1] = duration_s

        for position, (original_index, _) in enumerate(ordered):
            value = values[position]
            output[original_index] = {
                "lap_log_relative_p75": float(math.log(value + epsilon) - math.log(p75 + epsilon)),
                "lap_log_relative_p90": float(math.log(value + epsilon) - math.log(p90 + epsilon)),
                "lap_percentile_rank": float(ranks[position]),
                "lap_low_rank_run_duration_s": float(run_lengths[position]),
            }
    return output


def normalized_sharpness_features(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Normalize Laplacian energy by local contrast for brightness invariance.

    A uniform brightness scale changes both contrast and Laplacian variance by
    approximately the same squared factor.  The log ratio therefore gives the
    classifier a sharpness signal that is less coupled to exposure level.
    Only measured frame statistics are used; labels and event metadata are
    intentionally ignored.
    """
    output: list[dict[str, float]] = []
    epsilon = 1e-6
    for row in rows:
        try:
            lap_var = float(row["lap_var"])
            contrast = float(row["contrast"] if "contrast" in row else row["luma_std"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("normalized sharpness rows require numeric lap_var and contrast") from exc
        if not math.isfinite(lap_var) or not math.isfinite(contrast):
            raise ValueError("normalized sharpness rows contain non-finite values")
        if lap_var < 0 or contrast < 0:
            raise ValueError("normalized sharpness values must be non-negative")
        output.append({
            "lap_log_contrast_normalized": float(math.log(lap_var + epsilon) - 2.0 * math.log(contrast + epsilon)),
        })
    return output


def relative_exposure_features(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Describe exposure changes relative to each video's own measured baseline."""
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    metadata: dict[str, tuple[str, str]] = {}
    for index, row in enumerate(rows):
        video_id = str(row.get("videoId") or "").strip()
        if not video_id:
            raise ValueError("relative exposure rows require videoId")
        try:
            start = float(row["windowStartS"])
            luma = float(row["luma_mean"])
            contrast = float(row["contrast"] if "contrast" in row else row["luma_std"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("relative exposure rows require numeric windowStartS, luma_mean, and contrast") from exc
        if not all(math.isfinite(value) for value in (start, luma, contrast)):
            raise ValueError("relative exposure rows contain non-finite values")
        if luma < 0 or contrast < 0:
            raise ValueError("relative exposure values must be non-negative")
        lineage = (str(row.get("sourceVideoId") or ""), str(row.get("split") or ""))
        if video_id in metadata:
            previous_source, previous_split = metadata[video_id]
            if lineage[0] != previous_source:
                raise ValueError(f"video {video_id} has conflicting sourceVideoId")
            if lineage[1] != previous_split:
                raise ValueError(f"video {video_id} has conflicting split")
        else:
            metadata[video_id] = lineage
        grouped.setdefault(video_id, []).append((index, row))

    epsilon = 1e-6
    output = [{name: 0.0 for name in RELATIVE_EXPOSURE_FEATURE_COLUMNS} for _ in rows]
    for video_id, indexed_rows in grouped.items():
        ordered = sorted(indexed_rows, key=lambda item: float(item[1]["windowStartS"]))
        starts = np.asarray([float(row["windowStartS"]) for _, row in ordered], dtype=np.float64)
        if len(starts) > 1 and np.any(np.diff(starts) == 0):
            raise ValueError(f"video {video_id} has duplicate windowStartS")
        luma = np.asarray([float(row["luma_mean"]) for _, row in ordered], dtype=np.float64)
        contrast = np.asarray(
            [float(row["contrast"] if "contrast" in row else row["luma_std"]) for _, row in ordered],
            dtype=np.float64,
        )
        luma_baseline = float(np.median(luma))
        contrast_baseline = float(np.median(contrast))
        luma_sorted = np.sort(luma)
        contrast_sorted = np.sort(contrast)
        count = len(ordered)
        luma_ranks = (
            np.searchsorted(luma_sorted, luma, side="left")
            + np.searchsorted(luma_sorted, luma, side="right")
        ) / (2.0 * count)
        contrast_ranks = (
            np.searchsorted(contrast_sorted, contrast, side="left")
            + np.searchsorted(contrast_sorted, contrast, side="right")
        ) / (2.0 * count)

        def run_durations(active: np.ndarray) -> np.ndarray:
            durations = np.zeros(count, dtype=np.float64)
            run_start: int | None = None
            run_end_s = 0.0
            runs: list[tuple[int, int, float]] = []
            for position, (_, row) in enumerate(ordered):
                start_s = float(row["windowStartS"])
                end_s = float(row.get("windowEndS", start_s + 0.5))
                if active[position]:
                    if run_start is None:
                        run_start, run_end_s = position, end_s
                    elif start_s <= run_end_s:
                        run_end_s = max(run_end_s, end_s)
                    else:
                        runs.append((run_start, position - 1, run_end_s))
                        run_start, run_end_s = position, end_s
                elif run_start is not None:
                    runs.append((run_start, position - 1, run_end_s))
                    run_start = None
            if run_start is not None:
                runs.append((run_start, count - 1, run_end_s))
            for first, last, end_s in runs:
                duration = max(0.0, end_s - float(ordered[first][1]["windowStartS"]))
                durations[first : last + 1] = duration
            return durations

        low_runs = run_durations(luma_ranks <= 0.25)
        high_runs = run_durations(luma_ranks >= 0.75)
        for position, (original_index, _) in enumerate(ordered):
            output[original_index] = {
                "luma_log_relative_median": float(math.log(luma[position] + epsilon) - math.log(luma_baseline + epsilon)),
                "luma_percentile_rank": float(luma_ranks[position]),
                "contrast_log_relative_median": float(math.log(contrast[position] + epsilon) - math.log(contrast_baseline + epsilon)),
                "contrast_percentile_rank": float(contrast_ranks[position]),
                "luma_low_run_duration_s": float(low_runs[position]),
                "luma_high_run_duration_s": float(high_runs[position]),
            }
    return output


def relative_edge_features(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Describe edge-energy changes relative to each video's own baseline."""
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    metadata: dict[str, tuple[str, str]] = {}
    for index, row in enumerate(rows):
        video_id = str(row.get("videoId") or "").strip()
        if not video_id:
            raise ValueError("relative edge rows require videoId")
        try:
            start = float(row["windowStartS"])
            gradient = float(row["gradient_energy"])
            lap_abs = float(row["lap_abs_mean"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("relative edge rows require numeric windowStartS, gradient_energy, and lap_abs_mean") from exc
        if not all(math.isfinite(value) for value in (start, gradient, lap_abs)):
            raise ValueError("relative edge rows contain non-finite values")
        if gradient < 0 or lap_abs < 0:
            raise ValueError("relative edge values must be non-negative")
        lineage = (str(row.get("sourceVideoId") or ""), str(row.get("split") or ""))
        if video_id in metadata:
            previous_source, previous_split = metadata[video_id]
            if lineage[0] != previous_source:
                raise ValueError(f"video {video_id} has conflicting sourceVideoId")
            if lineage[1] != previous_split:
                raise ValueError(f"video {video_id} has conflicting split")
        else:
            metadata[video_id] = lineage
        grouped.setdefault(video_id, []).append((index, row))

    epsilon = 1e-6
    output = [{name: 0.0 for name in RELATIVE_EDGE_FEATURE_COLUMNS} for _ in rows]
    for video_id, indexed_rows in grouped.items():
        ordered = sorted(indexed_rows, key=lambda item: float(item[1]["windowStartS"]))
        starts = np.asarray([float(row["windowStartS"]) for _, row in ordered], dtype=np.float64)
        if len(starts) > 1 and np.any(np.diff(starts) == 0):
            raise ValueError(f"video {video_id} has duplicate windowStartS")
        gradient = np.asarray([float(row["gradient_energy"]) for _, row in ordered], dtype=np.float64)
        lap_abs = np.asarray([float(row["lap_abs_mean"]) for _, row in ordered], dtype=np.float64)
        gradient_sorted = np.sort(gradient)
        lap_sorted = np.sort(lap_abs)
        count = len(ordered)
        gradient_ranks = (
            np.searchsorted(gradient_sorted, gradient, side="left")
            + np.searchsorted(gradient_sorted, gradient, side="right")
        ) / (2.0 * count)
        lap_ranks = (
            np.searchsorted(lap_sorted, lap_abs, side="left")
            + np.searchsorted(lap_sorted, lap_abs, side="right")
        ) / (2.0 * count)
        gradient_baseline = float(np.median(gradient))
        lap_baseline = float(np.median(lap_abs))
        for position, (original_index, _) in enumerate(ordered):
            output[original_index] = {
                "gradient_log_relative_median": float(math.log(gradient[position] + epsilon) - math.log(gradient_baseline + epsilon)),
                "gradient_percentile_rank": float(gradient_ranks[position]),
                "lap_abs_log_relative_median": float(math.log(lap_abs[position] + epsilon) - math.log(lap_baseline + epsilon)),
                "lap_abs_percentile_rank": float(lap_ranks[position]),
            }
    return output


def label_window(start_s: float, end_s: float, event: dict[str, Any] | None) -> str:
    if event is None:
        return "normal"
    # Match the six-decimal timestamps serialized in feature rows so an exact
    # frame-boundary center is not shifted by binary floating-point noise.
    center = round((start_s + end_s) / 2.0, 6)
    return str(event["type"]) if float(event["startTimeS"]) <= center < float(event["endTimeS"]) else "normal"


def event_ground_truth(record: dict[str, Any]) -> dict[str, Any] | None:
    event = record.get("anomaly")
    return None if event is None else dict(event)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_record(record: dict[str, Any], sample_period_s: float = 0.5) -> list[dict[str, Any]]:
    if sample_period_s <= 0 or not math.isfinite(sample_period_s):
        raise ValueError("sample_period_s must be finite and positive")
    path = Path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if _sha256(path) != record.get("videoSha256"):
        raise ValueError(f"video SHA-256 mismatch: {record.get('videoId')}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot decode video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError(f"video has missing timestamp or frame-count metadata: {path}")
    duration_s = frame_count / fps
    sample_every = max(1, round(fps * sample_period_s))
    actual_period = sample_every / fps
    rows: list[dict[str, Any]] = []
    previous_gray: np.ndarray | None = None
    index = 0
    try:
        while True:
            if index % sample_every == 0:
                ok, frame = capture.read()
            else:
                ok, frame = capture.grab(), None
            if not ok:
                break
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                gray_small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
                sharpness_gray = cv2.resize(gray_small, (64, 36), interpolation=cv2.INTER_AREA)
                height, width = gray.shape[:2]
                long_side = max(height, width)
                if long_side > 640:
                    scale = 640 / long_side
                    quality_gray = cv2.resize(
                        gray,
                        (max(1, round(width * scale)), max(1, round(height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    quality_gray = gray
                gray_float = quality_gray.astype(np.float32)
                gradient_features = frame_sharpness_features(sharpness_gray)
                start_s = index / fps
                end_s = min(duration_s, start_s + actual_period)
                event = record.get("anomaly")
                row = {
                    "recordId": record["recordId"],
                    "videoId": record["videoId"],
                    "sourceVideoId": record["sourceVideoId"],
                    "sourceVideoSha256": record["sourceVideoSha256"],
                    "split": record["split"],
                    "isSynthetic": bool(record["isSynthetic"]),
                    "groundTruthEvent": event_ground_truth(record),
                    "windowStartS": round(start_s, 6),
                    "windowEndS": round(end_s, 6),
                    "label": label_window(start_s, end_s, event),
                    "luma_mean": float(gray_float.mean()),
                    "luma_std": float(gray_float.std()),
                    "contrast": float(gray_float.std()),
                    "dark_ratio": float(np.mean(quality_gray <= 16)),
                    "bright_ratio": float(np.mean(quality_gray >= 239)),
                    "lap_var": float(cv2.Laplacian(quality_gray, cv2.CV_64F).var()),
                    "frame_diff": 0.0 if previous_gray is None else float(cv2.absdiff(gray_small, previous_gray).mean()),
                    "sharpnessFeatureSize": [64, 36],
                    **gradient_features,
                }
                for metadata_key in ("strengthLevel", "strength", "parameters"):
                    if metadata_key in record:
                        row[metadata_key] = record[metadata_key]
                rows.append(row)
                previous_gray = gray_small
            index += 1
    finally:
        capture.release()
    if not rows:
        raise ValueError(f"video decoded no feature windows: {path}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/konvid-1k/synthetic/manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/konvid-1k/synthetic/event-features.jsonl"))
    parser.add_argument("--sample-period-s", type=float, default=0.5)
    parser.add_argument("--split", choices=("train", "val", "test"), help="only decode records in this split")
    parser.add_argument("--part-index", type=int, help="zero-based manifest part for parallel extraction")
    parser.add_argument("--part-count", type=int, default=1, help="number of parallel manifest parts")
    parser.add_argument("--limit", type=int, help="process first N manifest records for smoke tests")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite existing feature table: {args.output}")
    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = select_manifest_records(records, args.split, part_index=args.part_index, part_count=args.part_count)
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("limit must be positive")
        records = records[:args.limit]
    if not records:
        parser.error("manifest contains no records")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        count = 0
        for record_index, record in enumerate(records, start=1):
            for row in extract_record(record, args.sample_period_s):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
            handle.flush()
            if record_index == 1 or record_index % 100 == 0 or record_index == len(records):
                print(f"checkpointed video {record_index}/{len(records)}", flush=True)
    print(f"wrote {count} short-window rows for {len(records)} videos to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
