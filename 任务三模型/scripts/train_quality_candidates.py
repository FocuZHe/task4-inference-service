"""Compare train-only grouped-CV MOS regression candidates; never use test labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quality_experiment import refuse_existing_outputs
from train_quality import FEATURE_COLUMNS, FRAME_STATISTICS_COLUMNS, aggregate_video_features


SEED = 42
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
HISTOGRAM_PARAMETERS = (
    {"max_iter": 100, "max_leaf_nodes": 7, "l2_regularization": 1.0},
    {"max_iter": 200, "max_leaf_nodes": 7, "l2_regularization": 1.0},
    {"max_iter": 200, "max_leaf_nodes": 15, "l2_regularization": 1.0},
    {"max_iter": 200, "max_leaf_nodes": 15, "l2_regularization": 10.0},
)
RANDOM_FOREST_PARAMETERS = {
    "n_estimators": 160,
    "min_samples_leaf": 2,
    "max_features": 1.0,
    "n_jobs": 4,
}
STABLE_FOLD_WIN_FRACTION = 0.75

FEATURE_PROFILE_COLUMNS = {
    "quality-features-v1": tuple(FEATURE_COLUMNS),
    "quality-features-v2": tuple(
        f"{base}_{suffix}"
        for base in ("lap_var", "luma", "dark_ratio", "bright_ratio", "contrast", "frame_diff")
        for suffix in ("mean", "std", "p10", "p50", "p90")
    ),
    "quality-frame-statistics-v2": FRAME_STATISTICS_COLUMNS,
}


def _load_train_val(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        split = str(row.get("split") or "").strip()
        if split not in {"train", "val", "test", "external_test"}:
            raise ValueError(f"unsupported split {split!r}")
        if split in {"train", "val"}:
            (train if split == "train" else val).append(row)
    return train, val


def _validate_source_profiles(rows: list[dict[str, Any]]) -> str:
    profiles = {str(row["feature_profile"]).strip() for row in rows if row.get("feature_profile")}
    missing_profile = any(not str(row.get("feature_profile") or "").strip() for row in rows)
    if len(profiles) > 1 or (profiles and missing_profile):
        raise ValueError(f"mixed feature profiles are not allowed: {sorted(profiles)}")
    configs = {
        int(row["extraction_config"]["max_long_side"])
        for row in rows
        if isinstance(row.get("extraction_config"), dict)
        and row["extraction_config"].get("max_long_side") is not None
    }
    missing_config = any(
        not isinstance(row.get("extraction_config"), dict)
        or row["extraction_config"].get("max_long_side") is None
        for row in rows
    )
    if len(configs) > 1 or (configs and missing_config):
        raise ValueError(f"mixed max_long_side extraction configurations are not allowed: {sorted(configs)}")
    # Rows from the original extractor predate provenance metadata. Keep them
    # usable, but make the legacy/unverified status explicit in reports.
    return next(iter(profiles)) if profiles else "legacy640"


def _videos(
    rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...],
    *,
    frame_statistics: bool = False,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    splits: dict[str, str] = {}
    scores: dict[str, float] = {}
    for row in rows:
        video_id = str(row.get("video_id") or "").strip()
        if not video_id:
            raise ValueError("every train/val row requires video_id")
        split = str(row["split"])
        if video_id in splits and splits[video_id] != split:
            raise ValueError(f"{video_id}: video appears in multiple splits")
        splits[video_id] = split
        try:
            mos = float(row["mos"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{video_id}: invalid mos") from exc
        if not math.isfinite(mos) or (video_id in scores and scores[video_id] != mos):
            raise ValueError(f"{video_id}: invalid or inconsistent mos")
        scores[video_id] = mos
        grouped.setdefault(video_id, []).append(row)
    if frame_statistics:
        aggregated: list[dict[str, Any]] = []
        for video_id, windows in grouped.items():
            values_by_column: dict[str, float] = {}
            for column in feature_columns:
                values = np.asarray([float(window[column]) for window in windows], dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError(f"{video_id}: invalid {column}")
                values_by_column[column] = float(np.mean(values))
            aggregated.append(
                {
                    "video_id": video_id,
                    "split": splits[video_id],
                    "mos": scores[video_id],
                    **values_by_column,
                }
            )
    else:
        aggregated = aggregate_video_features(rows, FEATURE_COLUMNS)
    return [
        {
            "video_id": row["video_id"],
            "split": row["split"],
            "mos": row["mos"],
            **{column: row[column] for column in feature_columns},
        }
        for row in aggregated
    ]


def build_feature_matrix(rows: list[dict[str, Any]], feature_columns: tuple[str, ...]) -> np.ndarray:
    """Build the exact numeric matrix used for fitting and scoring."""
    return np.asarray([[row[column] for column in feature_columns] for row in rows], dtype=np.float64)


def build_group_folds(
    rows: list[dict[str, Any]], n_splits: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray([row["video_id"] for row in rows])
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(np.zeros((len(rows), 1)), groups=groups))


def _candidate_specs() -> list[tuple[str, dict[str, Any]]]:
    specs = [(f"ridge_alpha_{alpha:g}", {"family": "ridge", "alpha": alpha}) for alpha in RIDGE_ALPHAS]
    for params in HISTOGRAM_PARAMETERS:
        name = f"hist_iter_{params['max_iter']}_leaf_{params['max_leaf_nodes']}_l2_{params['l2_regularization']:g}"
        specs.append((name, {"family": "hist_gradient_boosting", **params}))
    specs.append(("random_forest_fixed", {"family": "random_forest", **RANDOM_FOREST_PARAMETERS}))
    return specs


def _make_model(spec: dict[str, Any], seed: int) -> Any:
    imputer = SimpleImputer(strategy="median")
    if spec["family"] == "ridge":
        return make_pipeline(imputer, StandardScaler(), Ridge(alpha=spec["alpha"]))
    if spec["family"] == "random_forest":
        return make_pipeline(
            imputer,
            RandomForestRegressor(
                n_estimators=spec["n_estimators"],
                min_samples_leaf=spec["min_samples_leaf"],
                max_features=spec["max_features"],
                n_jobs=min(int(spec["n_jobs"]), 4),
                random_state=seed,
            ),
        )
    return make_pipeline(
        imputer,
        HistGradientBoostingRegressor(
            max_iter=spec["max_iter"],
            max_leaf_nodes=spec["max_leaf_nodes"],
            l2_regularization=spec["l2_regularization"],
            random_state=seed,
        ),
    )


def _srcc(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    if len(actual) < 2 or np.ptp(actual) == 0 or np.ptp(predicted) == 0:
        return None
    result = float(spearmanr(actual, predicted).statistic)
    return result if math.isfinite(result) else None


def _ordered(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        -(metrics["cvSrcc"] if metrics["cvSrcc"] is not None else -math.inf),
        metrics["cvMae"],
        -(metrics["valSrcc"] if metrics["valSrcc"] is not None else -math.inf),
        metrics["valMae"],
    )


def _bootstrap_interval(
    actual: np.ndarray,
    predicted: np.ndarray,
    metric: str,
    *,
    seed: int,
    samples: int = 2000,
) -> list[float] | None:
    """Return a deterministic percentile bootstrap interval for validation metrics."""
    if len(actual) < 2:
        return None
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(actual), size=len(actual))
        sampled_actual = actual[indices]
        sampled_predicted = predicted[indices]
        if metric == "srcc":
            score = _srcc(sampled_actual, sampled_predicted)
        elif metric == "mae":
            score = float(np.mean(np.abs(sampled_actual - sampled_predicted)))
        else:
            raise ValueError(f"unsupported bootstrap metric: {metric}")
        if score is not None and math.isfinite(score):
            values.append(float(score))
    if not values:
        return None
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def run_candidate_experiment(
    input_path: Path,
    output_dir: Path,
    *,
    seed: int = SEED,
    n_splits: int = 5,
    feature_profile: str = "quality-features-v1",
) -> dict[str, Any]:
    if feature_profile not in FEATURE_PROFILE_COLUMNS:
        raise ValueError(f"unsupported feature profile: {feature_profile}")
    feature_columns = FEATURE_PROFILE_COLUMNS[feature_profile]
    output_dir = refuse_existing_outputs(output_dir, ("quality_candidate.joblib", "candidate_report.json"))
    train_rows, val_rows = _load_train_val(input_path)
    if not train_rows or not val_rows:
        raise ValueError("features must contain train and val rows")
    source_rows = train_rows + val_rows
    input_feature_profile = _validate_source_profiles(source_rows)
    max_long_sides = {
        int(row["extraction_config"]["max_long_side"])
        for row in source_rows
        if isinstance(row.get("extraction_config"), dict)
        and row["extraction_config"].get("max_long_side") is not None
    }
    input_max_long_side = next(iter(max_long_sides)) if len(max_long_sides) == 1 else None
    frame_statistics = feature_profile == "quality-frame-statistics-v2"
    train = _videos(train_rows, feature_columns, frame_statistics=frame_statistics)
    val = _videos(val_rows, feature_columns, frame_statistics=frame_statistics)
    train_ids = {row["video_id"] for row in train}
    val_ids = {row["video_id"] for row in val}
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(f"{sorted(overlap)[0]}: video appears in multiple splits")
    if len(train) < n_splits:
        raise ValueError("train video count must be at least n_splits")
    x_train = build_feature_matrix(train, feature_columns)
    y_train = np.asarray([row["mos"] for row in train], dtype=np.float64)
    groups = np.asarray([row["video_id"] for row in train])
    x_val = build_feature_matrix(val, feature_columns)
    y_val = np.asarray([row["mos"] for row in val], dtype=np.float64)
    folds = list(GroupKFold(n_splits=n_splits).split(x_train, y_train, groups))

    evaluations: list[dict[str, Any]] = []
    for name, spec in _candidate_specs():
        fold_srcc: list[float | None] = []
        fold_mae: list[float] = []
        for fold_train, fold_val in folds:
            model = _make_model(spec, seed)
            model.fit(x_train[fold_train], y_train[fold_train])
            predicted = np.asarray(model.predict(x_train[fold_val]), dtype=np.float64)
            actual = y_train[fold_val]
            fold_srcc.append(_srcc(actual, predicted))
            fold_mae.append(float(np.mean(np.abs(actual - predicted))))
        final_model = _make_model(spec, seed)
        final_model.fit(x_train, y_train)
        val_predicted = np.asarray(final_model.predict(x_val), dtype=np.float64)
        if not np.isfinite(val_predicted).all():
            raise ValueError(f"{name}: non-finite validation predictions")
        candidate_index = len(evaluations)
        evaluations.append({
            "name": name,
            "parameters": spec,
            "cvSrcc": (
                float(np.mean([score for score in fold_srcc if score is not None]))
                if any(score is not None for score in fold_srcc)
                else None
            ),
            "cvMae": float(np.mean(fold_mae)),
            "foldSrcc": fold_srcc,
            "foldMae": fold_mae,
            "valSrcc": _srcc(y_val, val_predicted),
            "valMae": float(np.mean(np.abs(y_val - val_predicted))),
            "valBootstrap95": {
                "srcc": _bootstrap_interval(y_val, val_predicted, "srcc", seed=seed + candidate_index * 2),
                "mae": _bootstrap_interval(y_val, val_predicted, "mae", seed=seed + candidate_index * 2 + 1),
                "samples": 2000,
            },
            "model": final_model,
        })

    baseline = next(item for item in evaluations if item["name"] == "ridge_alpha_1")
    baseline_fold_scores = baseline["foldSrcc"]
    candidates = [baseline]
    for item in evaluations:
        if item is baseline:
            continue
        wins = sum(
            score is not None and base is not None and score > base
            for score, base in zip(item["foldSrcc"], baseline_fold_scores)
        )
        required_wins = math.ceil(n_splits * STABLE_FOLD_WIN_FRACTION)
        if item["cvSrcc"] is not None and baseline["cvSrcc"] is not None and item["cvSrcc"] > baseline["cvSrcc"] and wins >= required_wins:
            candidates.append(item)
    selected = sorted(candidates, key=_ordered)[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = "quality_candidate.joblib"
    joblib.dump({
        "model": selected["model"],
        "features": list(feature_columns),
        "featureProfile": feature_profile,
        "inputFeatureProfile": input_feature_profile,
        "inputFeatureProfileStatus": (
            "legacy_unverified" if input_feature_profile == "legacy640" else "metadata_present"
        ),
        "inputExtractionConfig": {
            "maxLongSide": input_max_long_side,
            "mixedMaxLongSideRejected": len(max_long_sides) > 1,
        },
        "modelVersion": f"{selected['name']}-candidate-v2",
        "candidateOnly": True,
        "target": "mos",
        "qualityScoreNormalization": {
            "method": "train_min_max_clipped",
            "mosMin": float(np.min(y_train)),
            "mosMax": float(np.max(y_train)),
            "direction": "higher_is_better",
        },
    }, output_dir / model_file)
    model_sha256 = hashlib.sha256((output_dir / model_file).read_bytes()).hexdigest()
    report = {
        "schemaVersion": "task3.quality-candidate-report/2.0",
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": {"path": str(input_path), "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()},
        "features": list(feature_columns),
        "featureProfile": feature_profile,
        "inputFeatureProfile": input_feature_profile,
        "inputFeatureProfileStatus": (
            "legacy_unverified" if input_feature_profile == "legacy640" else "metadata_present"
        ),
        "inputExtractionConfig": {
            "maxLongSide": input_max_long_side,
            "mixedMaxLongSideRejected": len(max_long_sides) > 1,
        },
        "splitCounts": {"train": len(train), "val": len(val)},
        "candidateCount": len(evaluations),
        "seed": seed,
        "nSplits": n_splits,
        "selectionRule": {
            "ordering": ["meanCvSrcc_desc", "meanCvMae_asc", "valSrcc_desc", "valMae_asc"],
            "stableImprovement": f"candidate CV SRCC must exceed baseline and beat baseline fold SRCC in at least {STABLE_FOLD_WIN_FRACTION:.0%} of folds",
            "fallback": "retain ridge_alpha_1 when no candidate meets the stable-improvement condition",
        },
        "baselineCandidate": baseline["name"],
        "selectedCandidate": selected["name"],
        "modelFile": model_file,
        "modelSha256": model_sha256,
        "testNotUsed": True,
        "candidates": [
            {key: value for key, value in item.items() if key != "model"}
            for item in evaluations
        ],
    }
    (output_dir / "candidate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--feature-profile", choices=sorted(FEATURE_PROFILE_COLUMNS), default="quality-features-v1")
    args = parser.parse_args(argv)
    report = run_candidate_experiment(
        args.features,
        args.output_dir,
        seed=args.seed,
        n_splits=args.n_splits,
        feature_profile=args.feature_profile,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
