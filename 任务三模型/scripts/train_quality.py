"""Train baseline and histogram-gradient models for MOS quality regression."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scripts.quality_experiment import refuse_existing_outputs
except ModuleNotFoundError:
    from quality_experiment import refuse_existing_outputs


FEATURE_COLUMNS = [
    "lap_var_mean",
    "luma_mean",
    "dark_ratio_mean",
    "bright_ratio_mean",
    "contrast_mean",
    "frame_diff_mean",
]
FRAME_STATISTICS_COLUMNS = tuple(
    f"{base}_{suffix}"
    for base in ("lap_var", "luma", "dark_ratio", "bright_ratio", "contrast", "frame_diff")
    for suffix in ("mean", "std", "p05", "p50", "p95")
)
ALLOWED_SPLITS = {"train", "val", "test", "external_test"}


def load_feature_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    raise ValueError("feature table must be .jsonl or .csv")


def _finite_float(value: Any, field: str, video_id: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{video_id}: invalid {field}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{video_id}: non-finite {field}")
    return result


def aggregate_video_features(
    rows: list[dict[str, Any]],
    feature_columns: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Collapse window rows to video-level temporal summary features.

    Window feature names ending in ``_mean`` are treated as measurements of
    the corresponding base feature, so ``lap_var_mean`` produces the public
    ``lap_var_mean/std/p10/p50/p90`` columns.  Other names retain their own
    base name.  Labels and split assignments are metadata, never features.
    """
    columns = tuple(FEATURE_COLUMNS if feature_columns is None else feature_columns)
    grouped: dict[str, list[dict[str, Any]]] = {}
    split_by_id: dict[str, str] = {}
    mos_by_id: dict[str, float | None] = {}
    for row in rows:
        video_id = str(row.get("video_id") or "").strip()
        if not video_id:
            raise ValueError("every row requires video_id")
        split = str(row.get("split") or "").strip()
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"{video_id}: unsupported split {split!r}")
        if video_id in split_by_id and split_by_id[video_id] != split:
            raise ValueError(f"{video_id}: video appears in multiple splits")
        split_by_id[video_id] = split
        mos_value = row.get("mos")
        mos = None if mos_value is None or str(mos_value).strip() == "" else _finite_float(mos_value, "mos", video_id)
        if video_id in mos_by_id and mos_by_id[video_id] != mos:
            raise ValueError(f"{video_id}: inconsistent MOS across windows")
        mos_by_id[video_id] = mos
        grouped.setdefault(video_id, []).append(row)

    videos: list[dict[str, Any]] = []
    for video_id, windows in grouped.items():
        mos = mos_by_id[video_id]
        split = split_by_id[video_id]
        if split in {"train", "val", "test"} and mos is None:
            raise ValueError(f"{video_id}: MOS is required for {split}")
        features: dict[str, float] = {}
        for name in columns:
            base_name = name[:-5] if name.endswith("_mean") else name
            values = np.asarray(
                [_finite_float(window.get(name), name, video_id) for window in windows],
                dtype=np.float64,
            )
            features.update(
                {
                    f"{base_name}_mean": float(np.mean(values)),
                    f"{base_name}_std": float(np.std(values)),
                    f"{base_name}_p10": float(np.percentile(values, 10)),
                    f"{base_name}_p50": float(np.percentile(values, 50)),
                    f"{base_name}_p90": float(np.percentile(values, 90)),
                }
            )
        videos.append({"video_id": video_id, "split": split, "mos": mos, **features})
    return videos


def _video_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compatibility wrapper retaining the six-column baseline feature vector."""
    aggregated = aggregate_video_features(rows, FEATURE_COLUMNS)
    return [
        {
            "video_id": row["video_id"],
            "split": row["split"],
            "mos": row["mos"],
            **{column: row[column] for column in FEATURE_COLUMNS},
        }
        for row in aggregated
    ]


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    correlation = (
        float(spearmanr(actual, predicted).statistic)
        if len(actual) > 1 and np.ptp(actual) > 0 and np.ptp(predicted) > 0
        else float("nan")
    )
    if not math.isfinite(correlation):
        correlation = None
    return {
        "sampleCount": int(len(actual)),
        "srcc": correlation,
        "maeMos": float(np.mean(np.abs(actual - predicted))),
    }


def _make_models() -> dict[str, Any]:
    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    histogram = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=200,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    )
    return {"ridge": ridge, "hist_gradient_boosting": histogram}


def train_models(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir = refuse_existing_outputs(
        output_dir,
        ("quality_ridge.joblib", "quality_hist_gradient_boosting.joblib", "training_report.json"),
    )
    videos = _video_rows(rows)
    partitions = {name: [row for row in videos if row["split"] == name] for name in ("train", "val", "test")}
    if not partitions["train"] or not partitions["val"] or not partitions["test"]:
        raise ValueError("train, val, and test splits must each contain at least one MOS-labeled video")

    train_rows = partitions["train"]
    val_rows = partitions["val"]
    x_train = np.asarray([[row[column] for column in FEATURE_COLUMNS] for row in train_rows], dtype=np.float64)
    y_train = np.asarray([row["mos"] for row in train_rows], dtype=np.float64)
    x_val = np.asarray([[row[column] for column in FEATURE_COLUMNS] for row in val_rows], dtype=np.float64)
    y_val = np.asarray([row["mos"] for row in val_rows], dtype=np.float64)

    mos_min = float(np.min(y_train))
    mos_max = float(np.max(y_train))
    normalization = {
        "method": "train_min_max_clipped",
        "mosMin": mos_min,
        "mosMax": mos_max,
        "direction": "higher_is_better",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    models = _make_models()
    validation: dict[str, Any] = {}
    filenames = {
        "ridge": "quality_ridge.joblib",
        "hist_gradient_boosting": "quality_hist_gradient_boosting.joblib",
    }
    for name, model in models.items():
        model.fit(x_train, y_train)
        predicted = model.predict(x_val)
        validation[name] = _metrics(y_val, predicted)
        joblib.dump(
            {
                "model": model,
                "features": FEATURE_COLUMNS,
                "modelVersion": f"{name}-v1",
                "target": "mos",
                "qualityScoreNormalization": normalization,
            },
            output_dir / filenames[name],
        )

    report = {
        "schemaVersion": "task3.quality-training-report/1.0",
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "features": FEATURE_COLUMNS,
        "models": filenames,
        "sampleCounts": {name: len(partitions[name]) for name in ("train", "val", "test")},
        "validation": validation,
        "qualityScoreNormalization": normalization,
        "testUsedForSelection": False,
        "testEvaluation": "not_run_until_models_and_thresholds_are_frozen",
        "limitations": [
            "MOS regression is not an anomaly-class classifier.",
            "Quality scores are not drone-detection probabilities.",
            "Training split alone fits model preprocessing and MOS normalization.",
        ],
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="Windowed features (.jsonl or .csv)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = train_models(load_feature_table(args.features), args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
