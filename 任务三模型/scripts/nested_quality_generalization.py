from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np

from quality_generalization_data import load_train
from train_quality_candidates import FRAME_STATISTICS_COLUMNS, _make_model, _srcc, build_feature_matrix

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/raw/konvid-1k/manifest-official.jsonl"
FEATURES = ROOT / "data/raw/konvid-1k/quality-experiments/v2-multiscale-2026-11-11/features-native.jsonl"
INPUT_HASHES = {
    "manifest": "930c64f1635c6d225155d1f090eb50a391231f1d3331783e5b7f779b86cc27e8",
    "features": "60707a82545598dacc7a7542c37e1d6967d4ccca6032e31c4eee650d118a618f",
}
SPECS = [
    {"family": "random_forest", "n_estimators": 160, "min_samples_leaf": leaf,
     "max_features": fraction, "n_jobs": 4}
    for leaf, fraction in ((2, 1.0), (5, 0.7), (10, 0.7))
]


def folds(rows, count, seed):
    groups = sorted({row["source_group"] for row in rows})
    if len(groups) < count or count < 2:
        raise ValueError("insufficient groups")
    np.random.default_rng(seed).shuffle(groups)
    for part in np.array_split(groups, count):
        held = set(part.tolist())
        train = np.asarray([i for i, row in enumerate(rows) if row["source_group"] not in held])
        valid = np.asarray([i for i, row in enumerate(rows) if row["source_group"] in held])
        yield train, valid


def metrics(y, predicted):
    y, predicted = np.asarray(y, dtype=float), np.asarray(predicted, dtype=float)
    if y.shape != predicted.shape or not len(y) or not np.isfinite(predicted).all():
        raise ValueError("invalid predictions")
    return {"srcc": _srcc(y, predicted), "mae": float(np.mean(np.abs(y - predicted)))}


def ordering(item):
    index, score = item
    return (-(score["srcc"] if score["srcc"] is not None else -math.inf), score["mae"], index)


def paired_interval(records, *, samples=2000):
    groups = sorted({row["sourceGroup"] for row in records})
    buckets = {group: [row for row in records if row["sourceGroup"] == group] for group in groups}
    rng = np.random.default_rng(42)
    srcc, mae = [], []
    for _ in range(samples):
        sampled = [row for group in rng.choice(groups, len(groups), replace=True) for row in buckets[group]]
        y = [row["mos"] for row in sampled]
        base = metrics(y, [row["baseline"] for row in sampled])
        selected = metrics(y, [row["selected"] for row in sampled])
        if base["srcc"] is not None and selected["srcc"] is not None:
            srcc.append(selected["srcc"] - base["srcc"])
        mae.append(selected["mae"] - base["mae"])
    def interval(values):
        return [float(v) for v in np.quantile(values, [0.025, 0.975])] if values else None
    return {"samples": samples, "validSrccSamples": len(srcc),
            "deltaSrcc95": interval(srcc), "deltaMae95": interval(mae)}


def evaluate(rows, *, factory=_make_model, outer=5, inner=3):
    x = build_feature_matrix(rows, FRAME_STATISTICS_COLUMNS)
    y = np.asarray([row["mos"] for row in rows], dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("non-finite train input")
    records, fold_reports, fit_count = [], [], 0
    for fold_id, (train, held) in enumerate(folds(rows, outer, 42)):
        train_rows = [rows[i] for i in train]
        inner_results = []
        for spec_id, spec in enumerate(SPECS):
            actual, predicted = [], []
            for fit, valid in folds(train_rows, inner, 142 + fold_id):
                model = factory(spec, 42)
                model.fit(x[train[fit]], y[train[fit]])
                fit_count += 1
                values = np.asarray(model.predict(x[train[valid]]), dtype=float)
                metrics(y[train[valid]], values)
                actual.extend(y[train[valid]].tolist())
                predicted.extend(values.tolist())
            inner_results.append((spec_id, metrics(actual, predicted)))
        selected_id = min(inner_results, key=ordering)[0]
        outer_predictions = []
        for spec_id in (0, selected_id):
            model = factory(SPECS[spec_id], 42)
            model.fit(x[train], y[train])
            fit_count += 1
            values = np.asarray(model.predict(x[held]), dtype=float)
            metrics(y[held], values)
            outer_predictions.append(values)
        baseline, selected = outer_predictions
        fold_reports.append({
            "fold": fold_id, "selectedSpec": selected_id,
            "innerScores": [{"spec": i, **score} for i, score in inner_results],
            "trainSourceGroups": sorted({rows[i]["source_group"] for i in train}),
            "heldSourceGroups": sorted({rows[i]["source_group"] for i in held}),
            "baseline": metrics(y[held], baseline), "selected": metrics(y[held], selected),
        })
        for pos, i in enumerate(held):
            records.append({"videoId": rows[i]["video_id"], "sourceGroup": rows[i]["source_group"],
                            "outerFold": fold_id, "mos": float(y[i]),
                            "baseline": float(baseline[pos]), "selected": float(selected[pos])})
    if len(records) != len(rows) or len({r["videoId"] for r in records}) != len(rows):
        raise ValueError("incomplete or duplicate OOF predictions")
    baseline = metrics([r["mos"] for r in records], [r["baseline"] for r in records])
    selected = metrics([r["mos"] for r in records], [r["selected"] for r in records])
    delta = None if baseline["srcc"] is None or selected["srcc"] is None else selected["srcc"] - baseline["srcc"]
    wins = sum(f["baseline"]["srcc"] is not None and f["selected"]["srcc"] is not None
               and f["selected"]["srcc"] > f["baseline"]["srcc"] for f in fold_reports)
    report = {
        "schemaVersion": "task3.quality-generalization/1.0", "candidateOnly": True,
        "scope": "train-only-nested-selection-procedure", "independentExternalEvidence": False,
        "baseline": baseline, "selectedProcedure": selected, "deltaSrcc": delta,
        "deltaMae": selected["mae"] - baseline["mae"], "foldSrccWins": wins,
        "developmentPromising": bool(delta is not None and delta >= 0.01
                                     and selected["mae"] <= baseline["mae"] and wins >= math.ceil(outer * 0.8)),
        "fitCount": fit_count, "folds": fold_reports, "bootstrap": paired_interval(records),
        "limitations": ["historically used development data", "not device/scene grouped",
                        "paired OOF bootstrap does not include retraining uncertainty",
                        "no final model exported; no independent external improvement claim"],
    }
    return report, sorted(records, key=lambda row: row["videoId"])

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, document):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def reserve_output(path):
    path = Path(path).resolve()
    allowed = (ROOT / "data/raw/konvid-1k/quality-experiments").resolve()
    if not path.is_relative_to(allowed) or path == allowed:
        raise ValueError("output must be a new quality-experiments child directory")
    path.mkdir(parents=True, exist_ok=False)
    return path


def run(output):
    actual = {"manifest": sha256(MANIFEST), "features": sha256(FEATURES)}
    if actual != INPUT_HASHES:
        raise ValueError("frozen input hash changed; investigate before fitting")
    rows = load_train(MANIFEST, FEATURES, expected_count=840)
    output = reserve_output(output)
    source_files = [Path(__file__), ROOT / "scripts/quality_generalization_data.py",
                    ROOT / "scripts/train_quality_candidates.py", ROOT / "scripts/train_quality.py"]
    protocol = {
        "seed": 42, "outerFolds": 5, "innerFolds": 3, "fitBudget": 55,
        "trainVideos": len(rows), "trainSourceGroups": len({r["source_group"] for r in rows}),
        "inputHashes": actual, "specs": SPECS, "featureColumns": list(FRAME_STATISTICS_COLUMNS),
        "sourceHashes": {str(p.relative_to(ROOT)): sha256(p) for p in source_files},
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib")},
        "selectionRule": "inner pooled OOF SRCC descending, MAE ascending, spec index ascending",
        "screeningRule": "delta OOF SRCC >=0.01; delta MAE <=0; >=4/5 outer SRCC wins",
        "usedSplits": ["train"], "candidateOnly": True, "modelExport": False,
        "bootstrapScope": "paired source-group OOF resampling, descriptive only",
    }
    write_new(output / "preregistration.json", protocol)
    try:
        report, records = evaluate(rows)
        if report["fitCount"] != 55:
            raise ValueError("fit budget mismatch")
        if actual != {"manifest": sha256(MANIFEST), "features": sha256(FEATURES)}:
            raise ValueError("input changed during research")
        if protocol["sourceHashes"] != {str(p.relative_to(ROOT)): sha256(p) for p in source_files}:
            raise ValueError("source changed during research")
        with (output / "oof-predictions.jsonl").open("x", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        report.update(protocolSha256=sha256(output / "preregistration.json"),
                      predictionsSha256=sha256(output / "oof-predictions.jsonl"))
        write_new(output / "report.json", report)
    except Exception as exc:
        write_new(output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description="Train-only nested quality research; never evaluate used test")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output_dir)
    print(json.dumps({"output": str(args.output_dir), "developmentPromising": report["developmentPromising"]}))


if __name__ == "__main__":
    main()

