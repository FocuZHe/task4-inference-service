"""Nested source-group validation for controlled-event candidates using train only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from diagnose_events import (
    DEFAULT_TIME_POSTPROCESS,
    EVENT_CLASSES,
    TIME_POSTPROCESS_CONFIGS,
    event_metrics_at_threshold,
    read_rows,
    validate_lineage_splits,
)


def _partition_sources(sources: Sequence[str], count: int, seed: int) -> list[list[str]]:
    if count < 2 or len(sources) < count:
        raise ValueError("source group count must be at least the number of folds")
    values = np.asarray(sorted(sources), dtype=object)
    np.random.default_rng(seed).shuffle(values)
    return [part.tolist() for part in np.array_split(values, count)]


def _feature_paths(input_path: Path | Sequence[Path]) -> list[Path]:
    paths = [input_path] if isinstance(input_path, Path) else list(input_path)
    if not paths:
        raise ValueError("at least one event feature input is required")
    return paths


def _ordered_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_registered_input(input_path: Path | Sequence[Path], provenance_path: Path) -> str:
    paths = _feature_paths(input_path)
    digest = _ordered_sha256(paths)
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        expected = provenance["candidateInputs"]["eventFeatures"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid event candidate provenance: {provenance_path}") from exc
    if expected != digest:
        raise ValueError("event feature input SHA-256 does not match registered provenance")
    return digest


def _candidate_model(name: str, spec: dict[str, Any], seed: int) -> Any:
    if spec.get("family") == "extra_trees":
        return ExtraTreesClassifier(
            **{key: value for key, value in spec.items() if key != "family"},
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    from train_event_candidates import _model

    return _model(name, spec, seed)


def _calibration_sources(sources: Sequence[str], fraction: float, seed: int) -> tuple[list[str], list[str]]:
    if not 0.0 < fraction < 0.5:
        raise ValueError("calibration_fraction must be greater than 0 and below 0.5")
    ordered = np.asarray(sorted(sources), dtype=object)
    if len(ordered) < 2:
        raise ValueError("at least two source groups are needed for fit/calibration isolation")
    np.random.default_rng(seed).shuffle(ordered)
    calibration_count = max(1, int(math.ceil(len(ordered) * fraction)))
    calibration = ordered[:calibration_count].tolist()
    fit = ordered[calibration_count:].tolist()
    return fit, calibration


def _rows_for_sources(rows: list[dict[str, Any]], sources: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["sourceVideoId"]) in sources]


def _thresholds_for_calibration(
    rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    model_classes: Sequence[str],
    time_config: Mapping[str, Any],
    *,
    strategy: str = "f1",
) -> dict[str, float]:
    if strategy not in {"f1", "joint_gate", "strict_joint_gate"}:
        raise ValueError("threshold strategy must be f1, joint_gate, or strict_joint_gate")
    by_class = {str(name): probabilities[:, index] for index, name in enumerate(model_classes)}
    thresholds: dict[str, float] = {}
    if strategy in {"joint_gate", "strict_joint_gate"}:
        from train_event_candidates import _freeze_recall, _low_motion_false_alarm

    for event_class in EVENT_CLASSES:
        choices = []
        for threshold in np.round(np.arange(0.15, 0.851, 0.05), 2):
            metrics = event_metrics_at_threshold(rows, by_class[event_class], event_class, float(threshold), time_config)
            if event_class == "freeze_suspected" and strategy in {"joint_gate", "strict_joint_gate"}:
                freeze_recall = _freeze_recall(rows, by_class[event_class], float(threshold), dict(time_config))
                low_motion_rate = _low_motion_false_alarm(rows, by_class[event_class], float(threshold), dict(time_config))
                recall_ok = freeze_recall is not None and freeze_recall >= 0.95
                false_alarm_limit = 0.0 if strategy == "strict_joint_gate" else 0.05
                false_alarm_ok = low_motion_rate is not None and low_motion_rate <= false_alarm_limit
                choices.append((
                    int(recall_ok and false_alarm_ok),
                    int(false_alarm_ok),
                    int(recall_ok),
                    float(metrics["f1"]),
                    float(metrics["precision"]),
                    float(threshold),
                ))
            else:
                choices.append((1, 1, 1, float(metrics["f1"]), float(metrics["precision"]), float(threshold)))
        thresholds[event_class] = max(choices)[-1]
    return thresholds


def _score(
    rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    model_classes: Sequence[str],
    thresholds: Mapping[str, float],
    time_config: Mapping[str, Any],
) -> dict[str, Any]:
    by_class = {str(name): probabilities[:, index] for index, name in enumerate(model_classes)}
    per_class = {
        event_class: event_metrics_at_threshold(rows, by_class[event_class], event_class, thresholds[event_class], time_config)
        for event_class in EVENT_CLASSES
    }
    macro_f1 = float(np.mean([float(metrics["f1"]) for metrics in per_class.values()]))
    from train_event_candidates import _freeze_recall, _low_motion_false_alarm

    freeze_recall = _freeze_recall(rows, by_class["freeze_suspected"], thresholds["freeze_suspected"], dict(time_config))
    false_alarm = _low_motion_false_alarm(rows, by_class["freeze_suspected"], thresholds["freeze_suspected"], dict(time_config))
    return {
        "macroF1": macro_f1,
        "freezeRecallAtLeast3s": freeze_recall,
        "lowMotionNegativeFreezeFalseAlarmRate": false_alarm,
        "perClass": per_class,
    }


def _rank(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    macro_f1 = metrics.get("macroF1")
    freeze_recall = metrics.get("freezeRecallAtLeast3s")
    false_alarm = metrics.get("lowMotionNegativeFreezeFalseAlarmRate")
    return (
        -math.inf if macro_f1 is None else float(macro_f1),
        -math.inf if freeze_recall is None else float(freeze_recall),
        -math.inf if false_alarm is None else -float(false_alarm),
    )


def _candidate_metrics(
    candidate_rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    model_classes: Sequence[str],
    time_config: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    thresholds = _thresholds_for_calibration(candidate_rows, probabilities, model_classes, time_config)
    metrics = _score(candidate_rows, probabilities, model_classes, thresholds, time_config)
    return thresholds, metrics


def nested_source_group_evaluation(
    rows: list[dict[str, Any]],
    *,
    feature_columns: Sequence[str],
    candidate_specs: Sequence[tuple[str, dict[str, Any]]],
    time_configs: Mapping[str, Mapping[str, Any]] | None = None,
    outer_splits: int = 5,
    inner_splits: int = 3,
    calibration_fraction: float = 0.2,
    seed: int = 42,
    threshold_strategy: str = "f1",
) -> dict[str, Any]:
    """Select candidates inside each outer fold and score only held-out source groups.

    Each fit/calibration pair has disjoint source IDs. Inner validation selects a
    candidate; outer validation estimates its performance using a newly fitted
    model and thresholds calibrated on a separate subset of outer-train sources.
    """
    if not rows:
        raise ValueError("training rows are empty")
    if any(row.get("split") == "test" for row in rows):
        raise ValueError("nested validation input must not contain test rows")
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("nested validation accepts train rows only")
    if not candidate_specs:
        raise ValueError("at least one candidate specification is required")
    if threshold_strategy not in {"f1", "joint_gate", "strict_joint_gate"}:
        raise ValueError("threshold_strategy must be f1, joint_gate, or strict_joint_gate")
    validate_lineage_splits(rows)
    source_groups = sorted({str(row["sourceVideoId"]) for row in rows})
    outer_parts = _partition_sources(source_groups, outer_splits, seed)
    configs = dict(time_configs or TIME_POSTPROCESS_CONFIGS)
    if not configs:
        raise ValueError("at least one time postprocess configuration is required")

    from train_event_candidates import (
        _macro_metrics,
        _validate_window_label,
        build_feature_matrix,
        fit_candidate_model,
    )

    for row in rows:
        _validate_window_label(row)

    outer_folds: list[dict[str, Any]] = []
    for outer_index, outer_holdout_list in enumerate(outer_parts):
        outer_holdout_sources = set(outer_holdout_list)
        outer_train_sources = set(source_groups) - outer_holdout_sources
        outer_train_rows = _rows_for_sources(rows, outer_train_sources)
        outer_holdout_rows = _rows_for_sources(rows, outer_holdout_sources)
        inner_parts = _partition_sources(
            sorted(outer_train_sources), inner_splits, seed + 1000 + outer_index
        )
        inner_results: dict[str, list[dict[str, Any]]] = {
            f"{name}__{time_name}": []
            for name, _ in candidate_specs
            for time_name in configs
        }

        for inner_index, inner_val_list in enumerate(inner_parts):
            inner_val_sources = set(inner_val_list)
            inner_train_sources = outer_train_sources - inner_val_sources
            fit_sources, calibration_sources = _calibration_sources(
                sorted(inner_train_sources), calibration_fraction,
                seed + 10_000 + outer_index * 100 + inner_index,
            )
            fit_set, calibration_set = set(fit_sources), set(calibration_sources)
            fit_rows = _rows_for_sources(outer_train_rows, fit_set)
            calibration_rows = _rows_for_sources(outer_train_rows, calibration_set)
            inner_val_rows = _rows_for_sources(outer_train_rows, inner_val_sources)
            x_fit = build_feature_matrix(fit_rows, tuple(feature_columns))
            y_fit = [str(row["label"]) for row in fit_rows]
            x_calibration = build_feature_matrix(calibration_rows, tuple(feature_columns))
            x_inner_val = build_feature_matrix(inner_val_rows, tuple(feature_columns))

            for candidate_name, spec in candidate_specs:
                model = _candidate_model(candidate_name, spec, seed + outer_index * 100 + inner_index)
                fit_candidate_model(model, x_fit, y_fit, spec)
                calibration_probabilities = model.predict_proba(x_calibration)
                validation_probabilities = model.predict_proba(x_inner_val)
                for time_name, config in configs.items():
                    thresholds = _thresholds_for_calibration(
                        calibration_rows, calibration_probabilities, model.classes_, config,
                        strategy=threshold_strategy,
                    )
                    metrics = _score(inner_val_rows, validation_probabilities, model.classes_, thresholds, config)
                    inner_results[f"{candidate_name}__{time_name}"].append(metrics)

        inner_summary = {
            name: {
                metric: float(np.mean([float(result[metric]) for result in results if result.get(metric) is not None]))
                if any(result.get(metric) is not None for result in results)
                else None
                for metric in ("macroF1", "freezeRecallAtLeast3s", "lowMotionNegativeFreezeFalseAlarmRate")
            }
            for name, results in inner_results.items()
        }
        selected_key = max(inner_summary, key=lambda name: _rank(inner_summary[name]))
        candidate_name, time_name = selected_key.rsplit("__", 1)
        selected_spec = next(spec for name, spec in candidate_specs if name == candidate_name)
        selected_config = configs[time_name]

        outer_fit_sources, outer_calibration_sources = _calibration_sources(
            sorted(outer_train_sources), calibration_fraction, seed + 20_000 + outer_index
        )
        outer_fit_set, outer_calibration_set = set(outer_fit_sources), set(outer_calibration_sources)
        outer_fit_rows = _rows_for_sources(outer_train_rows, outer_fit_set)
        outer_calibration_rows = _rows_for_sources(outer_train_rows, outer_calibration_set)
        model = _candidate_model(candidate_name, selected_spec, seed + 30_000 + outer_index)
        fit_candidate_model(
            model,
            build_feature_matrix(outer_fit_rows, tuple(feature_columns)),
            [str(row["label"]) for row in outer_fit_rows],
            selected_spec,
        )
        calibration_probabilities = model.predict_proba(
            build_feature_matrix(outer_calibration_rows, tuple(feature_columns))
        )
        thresholds = _thresholds_for_calibration(
            outer_calibration_rows, calibration_probabilities, model.classes_, selected_config,
            strategy=threshold_strategy,
        )
        outer_probabilities = model.predict_proba(build_feature_matrix(outer_holdout_rows, tuple(feature_columns)))
        outer_metrics = _score(outer_holdout_rows, outer_probabilities, model.classes_, thresholds, selected_config)
        outer_folds.append({
            "fold": outer_index,
            "selectedCandidate": selected_key,
            "selectedModel": candidate_name,
            "selectedTimePostprocess": time_name,
            "thresholds": thresholds,
            "outerTrainSourceCount": len(outer_train_sources),
            "outerScoringSourceCount": len(outer_holdout_sources),
            "fitSourceCount": len(outer_fit_set),
            "calibrationSourceCount": len(outer_calibration_set),
            "outerTrainSources": sorted(outer_train_sources),
            "outerScoringSources": sorted(outer_holdout_sources),
            "fitSources": sorted(outer_fit_set),
            "calibrationSources": sorted(outer_calibration_set),
            "innerCandidateMeanMetrics": inner_summary,
            "outerMetrics": outer_metrics,
        })

    metric_names = ("macroF1", "freezeRecallAtLeast3s", "lowMotionNegativeFreezeFalseAlarmRate")
    outer_summary = {
        metric: float(np.mean([float(fold["outerMetrics"][metric]) for fold in outer_folds if fold["outerMetrics"].get(metric) is not None]))
        if any(fold["outerMetrics"].get(metric) is not None for fold in outer_folds)
        else None
        for metric in metric_names
    }
    return {
        "schemaVersion": "task3.nested-event-validation/1.0",
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "split": "train-only",
        "groupKey": "sourceVideoId",
        "sourceGroupCount": len(source_groups),
        "videoCount": len({str(row["videoId"]) for row in rows}),
        "outerSplitCount": outer_splits,
        "innerSplitCount": inner_splits,
        "calibrationFraction": calibration_fraction,
        "seed": seed,
        "thresholdSelectionStrategy": threshold_strategy,
        "featureColumns": list(feature_columns),
        "candidateNames": [name for name, _ in candidate_specs],
        "timePostprocessCandidates": sorted(configs),
        "testNotUsed": True,
        "outerSummary": outer_summary,
        "outerFolds": outer_folds,
        "limitations": [
            "指标来自 train 内嵌套源视频交叉验证，不是固定 KoNViD test 验收结果。",
            "受控异常由官方源视频生成，不能替代真实设备故障标注。",
        ],
    }


def _default_specs() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("rf160_leaf2", {"family": "random_forest", "n_estimators": 160, "min_samples_leaf": 2, "max_features": "sqrt"}),
        ("rf160_leaf2_blur2", {"family": "random_forest", "n_estimators": 160, "min_samples_leaf": 2, "max_features": "sqrt", "classWeightMultipliers": {"blur": 2.0}}),
        ("rf160_leaf4_blur2", {"family": "random_forest", "n_estimators": 160, "min_samples_leaf": 4, "max_features": "sqrt", "classWeightMultipliers": {"blur": 2.0}}),
        ("extra_trees160_leaf2", {"family": "extra_trees", "n_estimators": 160, "min_samples_leaf": 2, "max_features": "sqrt"}),
        ("hgb100_leaf7", {"family": "hist_gradient_boosting", "max_iter": 100, "max_leaf_nodes": 7, "learning_rate": 0.08, "l2_regularization": 1.0}),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, default=Path("docs/task3-v2-provenance.json"))
    parser.add_argument("--feature-profile", choices=("event-features-v1", "event-features-v2", "event-features-v3", "event-features-v4", "event-features-v5", "event-features-v6", "event-features-v7", "event-features-v8", "event-features-v9", "event-features-v10", "event-features-v11", "event-features-v12", "event-features-v13"), default="event-features-v5")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-strategy", choices=("f1", "joint_gate", "strict_joint_gate"), default="f1")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    input_digest = validate_registered_input(args.features, args.provenance)

    from train_event_candidates import (
        EVENT_FEATURE_PROFILE_COLUMNS,
        enrich_temporal_features,
    )

    rows = []
    for feature_path in args.features:
        rows.extend(read_rows(feature_path))
    rows = [row for row in rows if row.get("split") == "train"]
    profile = args.feature_profile
    if profile != "event-features-v1":
        rows = enrich_temporal_features(
            rows,
            relative_sharpness=profile in {"event-features-v3", "event-features-v4", "event-features-v5", "event-features-v6", "event-features-v7", "event-features-v8", "event-features-v9", "event-features-v10", "event-features-v11", "event-features-v12", "event-features-v13"},
            quantile_sharpness=profile in {"event-features-v5", "event-features-v6", "event-features-v7", "event-features-v8", "event-features-v9", "event-features-v10", "event-features-v11", "event-features-v12", "event-features-v13"},
            normalized_sharpness=profile in {"event-features-v6", "event-features-v7", "event-features-v8", "event-features-v9", "event-features-v10", "event-features-v11", "event-features-v12", "event-features-v13"},
            relative_exposure=profile in {"event-features-v8", "event-features-v9", "event-features-v11", "event-features-v12", "event-features-v13"},
            relative_edge=profile in {"event-features-v9", "event-features-v11", "event-features-v12", "event-features-v13"},
            context_sharpness=profile in {"event-features-v12", "event-features-v13"},
            wide_context_sharpness=profile == "event-features-v13",
        )
    report = nested_source_group_evaluation(
        rows,
        feature_columns=EVENT_FEATURE_PROFILE_COLUMNS[profile],
        candidate_specs=_default_specs(),
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
        threshold_strategy=args.threshold_strategy,
    )
    report["featureProfile"] = profile
    report["input"] = {
        "path": [str(path) for path in args.features],
        "sha256": input_digest,
        "provenancePath": str(args.provenance),
        "trainingRowsUsed": len(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["outerSummary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
