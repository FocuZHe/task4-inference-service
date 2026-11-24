"""One preregistered train-only study of eight fixed extraction configurations."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from quality_config_data import (
    FEATURE_COLUMNS, FIXED_CONFIGS, extract_configurations, load_official_train,
    sha256, verify_sources, write_jsonl_row, write_new,
)
from train_quality_candidates import _make_model, _srcc, build_feature_matrix

ROOT = Path(__file__).resolve().parents[1]
FROZEN_FEATURES_RELATIVE = Path("data/raw/konvid-1k/quality-experiments/v2-multiscale-2026-11-11/features-native.jsonl")
OUTPUT_RELATIVE = Path("data/raw/konvid-1k/quality-experiments/reviewability-config-2026-11-17-r1")
INPUT_HASHES = {"manifest": "930c64f1635c6d225155d1f090eb50a391231f1d3331783e5b7f779b86cc27e8",
                "features": "60707a82545598dacc7a7542c37e1d6967d4ccca6032e31c4eee650d118a618f"}
MODEL_SPEC = {"family": "random_forest", "n_estimators": 160, "min_samples_leaf": 2,
              "max_features": 1.0, "n_jobs": 4}
SEED, INNER_SEED, OUTER_FOLDS, INNER_FOLDS, FIT_BUDGET = 42, 142, 5, 3, 130


def make_fixed_rf():
    return _make_model(dict(MODEL_SPEC), SEED)


def source_folds(rows: list[dict[str, Any]], count: int, seed: int):
    groups = sorted({row["source_group"] for row in rows})
    if count < 2 or len(groups) < count:
        raise ValueError("insufficient source groups for fixed folds")
    np.random.default_rng(seed).shuffle(groups)
    for part in np.array_split(groups, count):
        held = set(part.tolist())
        yield (np.asarray([i for i, row in enumerate(rows) if row["source_group"] not in held], dtype=int),
               np.asarray([i for i, row in enumerate(rows) if row["source_group"] in held], dtype=int))


def metrics(y: Any, prediction: Any) -> dict[str, Any]:
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or y.shape != prediction.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("invalid metric input/predictions")
    return {"srcc": _srcc(y, prediction), "mae": float(np.mean(np.abs(y - prediction)))}


def _ordering(item: tuple[str, dict[str, Any]]):
    key, score = item
    return (math.inf if score["srcc"] is None else -score["srcc"], score["mae"], key)


def paired_interval(records: list[dict[str, Any]], samples: int = 2000) -> dict[str, Any]:
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
        raise ValueError("bootstrap samples must be positive integer")
    groups = sorted({row["sourceGroup"] for row in records})
    buckets = {group: [row for row in records if row["sourceGroup"] == group] for group in groups}
    differences: dict[str, list[float]] = {"srcc": [], "mae": []}
    rng = np.random.default_rng(SEED)
    for _ in range(samples):
        sampled = [row for group in rng.choice(groups, len(groups), replace=True) for row in buckets[group]]
        y = [row["mos"] for row in sampled]
        reference = metrics(y, [row["baseline"] for row in sampled])
        selected = metrics(y, [row["selected"] for row in sampled])
        if reference["srcc"] is not None and selected["srcc"] is not None:
            differences["srcc"].append(selected["srcc"] - reference["srcc"])
        differences["mae"].append(selected["mae"] - reference["mae"])
    def interval(values):
        return [float(v) for v in np.quantile(values, [.025, .975])] if values else None
    return {"samples": samples, "seed": SEED, "validSrccSamples": len(differences["srcc"]),
            "deltaSrcc95": interval(differences["srcc"]), "deltaMae95": interval(differences["mae"]),
            "scope": "paired source-group OOF descriptive resampling; excludes retraining uncertainty"}


def _validate_config_rows(configurations: dict[str, list[dict[str, Any]]]):
    if set(configurations) != set(FIXED_CONFIGS):
        raise ValueError("exactly the eight fixed configurations are required")
    reference = sorted(configurations["C0"], key=lambda row: row["video_id"])
    if not reference or len({row["video_id"] for row in reference}) != len(reference):
        raise ValueError("unique nonempty video rows required")
    by_source: dict[str, float] = {}
    for key in FIXED_CONFIGS:
        current = sorted(configurations[key], key=lambda row: row["video_id"])
        if [r["video_id"] for r in current] != [r["video_id"] for r in reference]:
            raise ValueError("configuration video coverage mismatch")
        for row, expected in zip(current, reference):
            if row.get("split") != "train" or row.get("source") != "konvid-1k":
                raise ValueError("only train official rows may enter evaluation")
            if row.get("source_group") != expected.get("source_group"):
                raise ValueError("configuration source mismatch")
            if row.get("mos") != expected.get("mos"):
                raise ValueError("configuration MOS mismatch")
            if row["source_group"] in by_source and by_source[row["source_group"]] != row["mos"]:
                raise ValueError("same source has conflicting MOS")
            by_source[row["source_group"]] = row["mos"]
        configurations = {**configurations, key: current}
    matrices = {key: build_feature_matrix(configurations[key], FEATURE_COLUMNS) for key in FIXED_CONFIGS}
    y = np.asarray([row["mos"] for row in reference], dtype=float)
    if not np.isfinite(y).all() or any(not np.isfinite(matrix).all() for matrix in matrices.values()):
        raise ValueError("non-finite train matrix/MOS")
    # Preflight nested group availability before a factory can run.
    for train, _ in source_folds(reference, OUTER_FOLDS, SEED):
        list(source_folds([reference[i] for i in train], INNER_FOLDS, INNER_SEED))
    return reference, matrices, y


def evaluate_configs(
    configurations: dict[str, list[dict[str, Any]]], *, factory: Callable[[], Any] = make_fixed_rf,
    fit_stats: dict[str, int] | None = None, bootstrap_samples: int = 2000,
    progress: Callable[[str], None] = print,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pure 130-fit grouped nested procedure; does not write or export weights."""
    rows, matrix, y = _validate_config_rows(configurations)
    stats = fit_stats if fit_stats is not None else {}
    stats.update(completedFits=0, attemptedFits=0)
    def fit_predict(config_id, fit, held):
        model = factory()
        stats["attemptedFits"] += 1
        model.fit(matrix[config_id][fit], y[fit])
        stats["completedFits"] += 1
        predicted = np.asarray(model.predict(matrix[config_id][held]), dtype=float)
        metrics(y[held], predicted)
        return predicted
    predictions = np.full((len(rows), 2), np.nan)
    fold_ids = np.full(len(rows), -1, dtype=int)
    selected_ids: dict[int, str] = {}
    fold_reports = []
    for fold, (outer_train, outer_held) in enumerate(source_folds(rows, OUTER_FOLDS, SEED)):
        progress(f"fit outer fold {fold + 1}/{OUTER_FOLDS}")
        train_rows = [rows[index] for index in outer_train]
        inner_oof = {key: np.full(len(rows), np.nan) for key in FIXED_CONFIGS}
        inner_source_folds = []
        for inner_fit_local, inner_held_local in source_folds(train_rows, INNER_FOLDS, INNER_SEED):
            inner_fit, inner_held = outer_train[inner_fit_local], outer_train[inner_held_local]
            inner_source_folds.append({"trainSourceGroups": sorted({rows[i]["source_group"] for i in inner_fit}),
                                       "heldSourceGroups": sorted({rows[i]["source_group"] for i in inner_held})})
            for key in FIXED_CONFIGS:
                inner_oof[key][inner_held] = fit_predict(key, inner_fit, inner_held)
        inner_scores = {key: metrics(y[outer_train], inner_oof[key][outer_train]) for key in FIXED_CONFIGS}
        selected = min(inner_scores.items(), key=_ordering)[0]
        baseline_pred = fit_predict("C0", outer_train, outer_held)
        selected_pred = fit_predict(selected, outer_train, outer_held)
        if np.isfinite(predictions[outer_held]).any():
            raise ValueError("outer OOF prediction written twice")
        predictions[outer_held, 0], predictions[outer_held, 1] = baseline_pred, selected_pred
        fold_ids[outer_held] = fold
        selected_ids[fold] = selected
        fold_reports.append({"fold": fold, "selectedConfig": selected, "innerScores": inner_scores,
                             "innerSourceFolds": inner_source_folds,
                             "trainSourceGroups": sorted({rows[i]["source_group"] for i in outer_train}),
                             "heldSourceGroups": sorted({rows[i]["source_group"] for i in outer_held}),
                             "baseline": metrics(y[outer_held], baseline_pred), "selected": metrics(y[outer_held], selected_pred)})
    if not np.isfinite(predictions).all() or (fold_ids < 0).any() or stats["completedFits"] != FIT_BUDGET:
        raise ValueError("incomplete OOF or fixed fit-budget mismatch")
    records = [{"videoId": row["video_id"], "sourceGroup": row["source_group"], "outerFold": int(fold_ids[i]),
                "mos": float(y[i]), "baseline": float(predictions[i, 0]), "selected": float(predictions[i, 1]),
                "selectedConfig": selected_ids[int(fold_ids[i])]} for i, row in enumerate(rows)]
    baseline, selected = metrics(y, predictions[:, 0]), metrics(y, predictions[:, 1])
    delta = None if baseline["srcc"] is None or selected["srcc"] is None else selected["srcc"] - baseline["srcc"]
    wins = sum(fold["baseline"]["srcc"] is not None and fold["selected"]["srcc"] is not None
               and fold["selected"]["srcc"] > fold["baseline"]["srcc"] for fold in fold_reports)
    promising = delta is not None and delta >= .01 and selected["mae"] <= baseline["mae"] and wins >= 4
    return {"schemaVersion": "task3.quality-config-study/1.0", "scope": "official-train-only-nested-config-selection",
            "candidateOnly": True, "automaticGateValidated": False, "independentExternalEvidence": False,
            "baseline": baseline, "selectedProcedure": selected, "deltaSrcc": delta,
            "deltaMae": selected["mae"] - baseline["mae"], "foldSrccWins": wins,
            "developmentPromising": bool(promising), "conclusion": "development_promising" if promising else "no_stable_improvement",
            "fitCount": stats["completedFits"], "attemptedFitCount": stats["attemptedFits"], "folds": fold_reports,
            "bootstrap": paired_interval(records, bootstrap_samples),
            "limitations": ["historically used development data", "SHA grouping is not device/scene grouping",
                            "MOS proxy research does not validate human reviewability", "no model exported or threshold promoted"]}, records


def _git_snapshot():
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    return {"branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"),
            "statusPorcelain": git("status", "--porcelain")}


def _source_hashes():
    paths = [Path(__file__).resolve(), ROOT / "scripts/quality_config_data.py", ROOT / "scripts/extract_features.py",
             ROOT / "scripts/train_quality_candidates.py", ROOT / "scripts/train_quality.py"]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def verify_caches(output: Path, summaries: dict[str, Any]) -> None:
    for config_id, summary in summaries.items():
        for filename, digest in summary["cacheSha256"].items():
            if sha256(output / "features" / config_id / filename) != digest:
                raise ValueError(f"cache hash changed: {config_id}/{filename}")


def run_study(manifest: Path, video_root: Path, output: Path) -> dict[str, Any]:
    """Official one-shot entry; no candidate overrides or incomplete-cache resume."""
    manifest, video_root, output = Path(manifest).resolve(), Path(video_root).resolve(), Path(output).resolve()
    if output != (ROOT / OUTPUT_RELATIVE).resolve():
        raise ValueError("output must be the fixed new study directory")
    if output.exists():
        raise FileExistsError(output)
    frozen = ROOT / FROZEN_FEATURES_RELATIVE
    if not manifest.is_file() or sha256(manifest) != INPUT_HASHES["manifest"] or not frozen.is_file() or sha256(frozen) != INPUT_HASHES["features"]:
        raise ValueError("frozen input hash mismatch; investigate before extraction/fitting")
    rows = load_official_train(manifest, video_root)
    if len({row["source_group"] for row in rows}) != 840:
        raise ValueError("expected 840 distinct official train source groups")
    code_hashes = _source_hashes()
    protocol = {
        "schemaVersion": "task3.quality-config-preregistration/1.0", "createdAt": datetime.now(timezone.utc).isoformat(),
        "inputHashes": dict(INPUT_HASHES), "trainVideoHashes": {row["video_id"]: row["source_group"] for row in rows},
        "manifestPath": str(manifest), "frozenFeaturePath": str(frozen), "videoRoot": str(video_root),
        "configurations": FIXED_CONFIGS, "featureColumns": list(FEATURE_COLUMNS), "modelFeatureProfile": "quality-frame-statistics-v2",
        "modelSpec": MODEL_SPEC, "seed": SEED, "innerSeed": INNER_SEED,
        "outerFolds": OUTER_FOLDS, "innerFolds": INNER_FOLDS, "fitBudget": FIT_BUDGET,
        "selectionRule": "inner pooled OOF SRCC descending (null last), MAE ascending, configuration ID ascending",
        "screeningRule": "delta OOF SRCC >=0.01; delta OOF MAE <=0; at least4/5 strict outer SRCC wins",
        "usedSplits": ["train"], "trainVideos": len(rows), "trainSourceGroups": 840,
        "modelExport": False, "candidateOnly": True, "automaticGateValidated": False,
        "sourceHashes": code_hashes, "git": _git_snapshot(), "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "opencv-python", "joblib")},
        "opencvThreads": 1, "rfJobs": 4, "aggregation": "equal mean of each window statistic; unweighted",
        "plannedVideoDecodes": 840 * 8,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "preregistration.json", protocol)
    stats: dict[str, int] = {"completedFits": 0, "attemptedFits": 0}
    extraction_stats: dict[str, Any] = {}
    old_threads = cv2.getNumThreads()
    try:
        cv2.setNumThreads(1)
        configurations, summaries = extract_configurations(rows, output, extraction_stats=extraction_stats)
        verify_sources(rows)
        verify_caches(output, summaries)
        if sha256(manifest) != INPUT_HASHES["manifest"] or sha256(frozen) != INPUT_HASHES["features"] or _source_hashes() != code_hashes:
            raise ValueError("input/source changed before fitting")
        report, records = evaluate_configs(configurations, fit_stats=stats)
        verify_sources(rows)
        verify_caches(output, summaries)
        if sha256(manifest) != INPUT_HASHES["manifest"] or sha256(frozen) != INPUT_HASHES["features"] or _source_hashes() != code_hashes:
            raise ValueError("input/source changed during research")
        report.update(extraction=summaries, preregistrationSha256=sha256(output / "preregistration.json"),
                      inputHashes=dict(INPUT_HASHES), sourceHashes=code_hashes)
        with (output / "oof-predictions.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for record in records:
                write_jsonl_row(handle, record)
        report["predictionsSha256"] = sha256(output / "oof-predictions.jsonl")
        artifacts = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}
        write_new(output / "artifact-hashes.json", artifacts)
        report["artifactHashesSha256"] = sha256(output / "artifact-hashes.json")
        write_new(output / "report.json", report)
        return report
    except BaseException as exc:
        write_new(output / "failure.json", {"status": "failed", "errorType": type(exc).__name__, "message": str(exc),
                                           "completedFits": stats["completedFits"], "attemptedFits": stats["attemptedFits"],
                                           "extraction": extraction_stats,
                                           "automaticGateValidated": False, "resumeAllowed": False})
        raise
    finally:
        cv2.setNumThreads(old_threads)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_study(args.manifest, args.video_root, args.output_dir)
    print(json.dumps({"output": str(args.output_dir), "developmentPromising": result["developmentPromising"], "fitCount": result["fitCount"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
