"""Evaluate a frozen video-quality model on KoNViD test or LIVE-VQC external_test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import spearmanr

from quality_experiment import refuse_existing_output
from train_quality import FEATURE_COLUMNS, aggregate_video_features, load_feature_table
from train_quality_candidates import FEATURE_PROFILE_COLUMNS


BASELINE_PROFILE = "quality-features-v1"
FRAME_STATISTICS_PROFILE = "quality-frame-statistics-v2"


def _finite_float(value: Any, field: str, video_id: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{video_id}: invalid {field}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{video_id}: invalid non-finite {field}")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _artifact_profile(artifact: dict[str, Any]) -> tuple[str, tuple[str, ...], bool]:
    raw_features = artifact.get("features")
    if not isinstance(raw_features, (list, tuple)):
        raise ValueError("model artifact requires a feature list")
    features = tuple(str(feature) for feature in raw_features)
    raw_profile = artifact.get("featureProfile")
    if raw_profile is None:
        if features != tuple(FEATURE_COLUMNS):
            raise ValueError("model feature list does not match the supported feature order")
        return BASELINE_PROFILE, features, True
    profile = str(raw_profile).strip()
    expected = FEATURE_PROFILE_COLUMNS.get(profile)
    if expected is None:
        raise ValueError(f"unsupported model feature profile: {profile!r}")
    if features != tuple(expected):
        raise ValueError(f"model feature list does not match profile {profile!r}")
    return profile, features, False


def _validate_input_metadata(
    rows: list[dict[str, Any]],
    artifact: dict[str, Any],
    *,
    legacy_baseline: bool,
) -> dict[str, str]:
    """Check extractor provenance before any test feature value is consumed."""
    if legacy_baseline:
        if artifact.get("candidateOnly"):
            raise ValueError("candidate artifact requires a metadata-backed input feature profile")
        with_metadata = [row for row in rows if row.get("feature_profile") or row.get("extraction_config")]
        if with_metadata:
            raise ValueError("legacy640 baseline requires feature rows without extraction metadata")
        return {
            "inputFeatureProfile": "legacy640",
            "inputFeatureProfileStatus": "legacy_unverified",
            "inputProvenanceAudit": "legacy640 input has no extraction metadata; allowed only for baseline evaluation",
        }

    expected_profile = str(artifact.get("inputFeatureProfile") or "").strip()
    if not expected_profile or expected_profile == "legacy640":
        raise ValueError("candidate artifact requires a metadata-backed input feature profile")
    expected_config = artifact.get("inputExtractionConfig")
    if not isinstance(expected_config, dict):
        raise ValueError("model artifact requires inputExtractionConfig")
    expected_max = expected_config.get("maxLongSide")
    if expected_max is not None:
        try:
            expected_max = int(expected_max)
        except (TypeError, ValueError) as exc:
            raise ValueError("model artifact has invalid inputExtractionConfig.maxLongSide") from exc

    seen_profiles: set[str] = set()
    seen_max: set[int] = set()
    missing_metadata = False
    for row in rows:
        profile = str(row.get("feature_profile") or "").strip()
        config = row.get("extraction_config")
        if not profile or not isinstance(config, dict):
            missing_metadata = True
            continue
        if profile != expected_profile:
            raise ValueError("input feature profile does not match model artifact")
        seen_profiles.add(profile)
        raw_max = config.get("max_long_side")
        if raw_max is not None:
            try:
                seen_max.add(int(raw_max))
            except (TypeError, ValueError) as exc:
                raise ValueError("input extraction configuration has invalid max_long_side") from exc
        if expected_max is not None and raw_max is None:
            raise ValueError("input extraction configuration does not match model artifact")
        if expected_max is not None and raw_max is not None and int(raw_max) != expected_max:
            raise ValueError("input extraction configuration does not match model artifact")
        if expected_max is None and raw_max is not None:
            raise ValueError("input extraction configuration does not match model artifact")
    if missing_metadata:
        raise ValueError("candidate input rows require feature_profile and extraction_config metadata")
    if seen_profiles != {expected_profile}:
        raise ValueError("input feature profile metadata is incomplete")
    if len(seen_max) > 1:
        raise ValueError("mixed input extraction configurations are not allowed")
    return {
        "inputFeatureProfile": expected_profile,
        "inputFeatureProfileStatus": "metadata_present",
    }


def _video_rows_for_split(
    rows: list[dict[str, Any]],
    profile: str = BASELINE_PROFILE,
    feature_columns: tuple[str, ...] | None = None,
    *,
    target_split: str,
    required_source: str,
    strict_target_only: bool = False,
) -> list[dict[str, Any]]:
    """Aggregate one evaluation split using the same recipe as candidate training."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    split_by_id: dict[str, str] = {}
    mos_by_id: dict[str, float | None] = {}
    selected_rows: list[dict[str, Any]] = []
    for row in rows:
        video_id = str(row.get("video_id") or "").strip()
        if not video_id:
            raise ValueError("every feature row requires video_id")
        split = str(row.get("split") or "").strip()
        if split not in {"train", "val", "test", "external_test"}:
            raise ValueError(f"{video_id}: unsupported split {split!r}")
        if strict_target_only and split != target_split:
            raise ValueError(f"external evaluation requires split=external_test, got {split!r}")
        if video_id in split_by_id and split_by_id[video_id] != split:
            raise ValueError(f"{video_id}: video appears in multiple splits")
        split_by_id[video_id] = split
        if split != target_split:
            continue
        raw_mos = row.get("mos")
        mos = None if raw_mos is None or str(raw_mos).strip() == "" else _finite_float(raw_mos, "mos", video_id)
        if video_id in mos_by_id and mos_by_id[video_id] != mos:
            raise ValueError(f"{video_id}: inconsistent MOS across windows")
        mos_by_id[video_id] = mos
        if row.get("source") != required_source:
            raise ValueError(f"{video_id}: {target_split} feature source must be {required_source}")
        if target_split == "external_test":
            if row.get("dataset") != "LIVE-VQC":
                raise ValueError(f"{video_id}: external_test feature dataset must be LIVE-VQC")
            if row.get("externalTestOnly") is not True:
                raise ValueError(f"{video_id}: externalTestOnly must be true")
            if row.get("trainingAllowed") is not False:
                raise ValueError(f"{video_id}: trainingAllowed must be false")
            if row.get("selectionAllowed") is not False:
                raise ValueError(f"{video_id}: selectionAllowed must be false")
        grouped.setdefault(video_id, []).append(row)
        selected_rows.append(row)

    if not grouped:
        raise ValueError(f"feature table contains no {target_split} videos")
    columns = tuple(feature_columns or FEATURE_PROFILE_COLUMNS[profile])
    for video_id, mos in mos_by_id.items():
        if mos is None:
            raise ValueError(f"{video_id}: MOS is required for {target_split} evaluation")

    if profile == FRAME_STATISTICS_PROFILE:
        videos: list[dict[str, Any]] = []
        for video_id, windows in grouped.items():
            values = {
                name: float(np.mean([_finite_float(window.get(name), name, video_id) for window in windows]))
                for name in columns
            }
            videos.append({"video_id": video_id, "mos": mos_by_id[video_id], **values})
        return videos

    # This is deliberately the training helper: it computes the window-level
    # temporal summary columns once, rather than recomputing an incompatible
    # statistic from already-aggregated values.
    aggregated = aggregate_video_features(selected_rows, FEATURE_COLUMNS)
    return [
        {
            "video_id": video["video_id"],
            "mos": video["mos"],
            **{name: video[name] for name in columns},
        }
        for video in aggregated
    ]


def _test_video_rows(
    rows: list[dict[str, Any]],
    profile: str = BASELINE_PROFILE,
    feature_columns: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    return _video_rows_for_split(
        rows,
        profile,
        feature_columns,
        target_split="test",
        required_source="konvid-1k",
    )


def _external_video_rows(
    rows: list[dict[str, Any]],
    profile: str = BASELINE_PROFILE,
    feature_columns: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    return _video_rows_for_split(
        rows,
        profile,
        feature_columns,
        target_split="external_test",
        required_source="LIVE-VQC",
        strict_target_only=True,
    )


def _srcc(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    if len(actual) < 2 or np.ptp(actual) == 0 or np.ptp(predicted) == 0:
        return None
    value = float(spearmanr(actual, predicted).statistic)
    return value if math.isfinite(value) else None


def _validate_provenance(provenance: dict[str, Any]) -> None:
    digest = provenance.get("packageSha256")
    if (
        provenance.get("dataset") != "KoNViD-1k"
        or provenance.get("annotationSource") != "official KoNViD-1k metadata package"
        or provenance.get("scoreTable") != "KoNViD_1k_mos.csv (10-decimal video-level MOS)"
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest.lower())
    ):
        raise ValueError("annotation provenance does not identify the official KoNViD MOS source and package hash")
    match = provenance.get("videoMatch")
    if not isinstance(match, dict) or match.get("unmatched") != 0:
        raise ValueError("official annotation provenance must confirm exact video filename matching")


_LIVE_VQC_OFFICIAL_URL_PREFIXES = (
    "https://www.colorado.edu/lab/live/",
    "https://live.ece.utexas.edu/research/LIVEVQC/",
)


def _validate_external_provenance(provenance: dict[str, Any], expected_video_count: int) -> None:
    digest = provenance.get("packageSha256")
    official_url = str(provenance.get("officialUrl") or "")
    license_note = str(provenance.get("license") or "").strip()
    if provenance.get("dataset") != "LIVE-VQC":
        raise ValueError("external provenance must identify LIVE-VQC")
    if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest.lower()):
        raise ValueError("external provenance must contain a valid package SHA-256")
    if not any(official_url.startswith(prefix) for prefix in _LIVE_VQC_OFFICIAL_URL_PREFIXES):
        raise ValueError("external provenance must contain a recognized LIVE-VQC official URL")
    if not license_note:
        raise ValueError("external provenance must contain a non-empty license")
    split_counts = provenance.get("split", {}).get("counts", {})
    if split_counts.get("external_test") != expected_video_count:
        raise ValueError("external provenance video count does not match expected external_test count")


def _validate_external_row_provenance(rows: list[dict[str, Any]], provenance: dict[str, Any]) -> None:
    expected_digest = str(provenance["packageSha256"]).lower()
    expected_url = str(provenance["officialUrl"])
    expected_license = str(provenance["license"]).strip()
    video_hashes: dict[str, str] = {}
    for row in rows:
        video_id = str(row.get("video_id") or "").strip()
        record_id = str(row.get("recordId") or "").strip()
        video_hash = str(row.get("videoSha256") or "").lower()
        if not record_id:
            raise ValueError("external row recordId is required")
        if not _valid_digest(video_hash):
            raise ValueError("external row videoSha256 must be a valid SHA-256")
        if video_id in video_hashes and video_hashes[video_id] != video_hash:
            raise ValueError("external row videoSha256 is inconsistent within a video")
        video_hashes[video_id] = video_hash
        if str(row.get("packageSha256") or "").lower() != expected_digest:
            raise ValueError("external row packageSha256 does not match provenance")
        if str(row.get("officialUrl") or "") != expected_url:
            raise ValueError("external row officialUrl does not match provenance")
        if str(row.get("license") or "").strip() != expected_license:
            raise ValueError("external row license does not match provenance")


def _validate_selection_record(
    path: Path,
    *,
    model_sha256: str,
    features_sha256: str,
    profile: str,
    artifact: dict[str, Any],
    match_features: bool = True,
) -> dict[str, str]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid selection record: {path}") from exc
    if not isinstance(record, dict) or record.get("frozen") is not True:
        raise ValueError("selection record must be frozen")
    if record.get("testUsedForSelection") is not False:
        raise ValueError("selection record must declare testUsedForSelection=false")
    if record.get("profile") != profile:
        raise ValueError("selection record profile does not match model artifact")
    if record.get("modelSha256") != model_sha256:
        raise ValueError("selection record modelSha256 does not match model")
    if match_features and record.get("featuresSha256") != features_sha256:
        raise ValueError("selection record featuresSha256 does not match features")
    for field in ("modelSha256", "featuresSha256"):
        if not _valid_digest(record.get(field)):
            raise ValueError(f"selection record {field} is invalid")
    if record["modelSha256"].lower() != model_sha256:
        raise ValueError("selection record modelSha256 is invalid")
    if artifact.get("modelSha256") and artifact.get("modelSha256") != model_sha256:
        raise ValueError("model artifact modelSha256 does not match model")
    if artifact.get("featuresSha256") and artifact.get("featuresSha256") != record["featuresSha256"]:
        raise ValueError("model artifact featuresSha256 does not match selection record")
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _evaluate_video_rows(
    videos: list[dict[str, Any]],
    artifact: dict[str, Any],
    provenance: dict[str, Any],
    feature_columns: tuple[str, ...],
    *,
    split_name: str,
    schema_version: str,
    count_key: str,
    bootstrap_samples: int,
    seed: int,
    model_sha256: str | None,
    features_sha256: str | None,
    selection_record: dict[str, str] | None,
    external_only: bool = False,
) -> dict[str, Any]:
    split_counts = provenance.get("split", {}).get("counts", {})
    expected_count = split_counts.get(count_key)
    if expected_count != len(videos):
        raise ValueError(f"{split_name} video count does not match provenance record")
    model = artifact.get("model")
    model_version = artifact.get("modelVersion")
    if model is None or not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("model artifact requires model and modelVersion")

    matrix = np.asarray([[video[name] for name in feature_columns] for video in videos], dtype=np.float64)
    actual = np.asarray([video["mos"] for video in videos], dtype=np.float64)
    predicted = np.asarray(model.predict(matrix), dtype=np.float64)
    if predicted.shape != actual.shape or not np.isfinite(predicted).all():
        raise ValueError(f"model returned invalid {split_name} predictions")

    srcc = _srcc(actual, predicted)
    if srcc is None:
        raise ValueError(f"{split_name} SRCC is undefined for constant or insufficient values")
    mae = float(np.mean(np.abs(actual - predicted)))

    rng = np.random.default_rng(seed)
    boot_srcc: list[float] = []
    boot_mae = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        sample = rng.integers(0, len(actual), size=len(actual))
        sampled_actual = actual[sample]
        sampled_predicted = predicted[sample]
        correlation = _srcc(sampled_actual, sampled_predicted)
        if correlation is not None:
            boot_srcc.append(correlation)
        boot_mae[index] = np.mean(np.abs(sampled_actual - sampled_predicted))
    if not boot_srcc:
        raise ValueError("no valid SRCC bootstrap replicates")

    srcc_low, srcc_high = np.quantile(boot_srcc, [0.025, 0.975])
    mae_low, mae_high = np.quantile(boot_mae, [0.025, 0.975])
    report = {
        "schemaVersion": schema_version,
        "dataset": str(provenance.get("dataset") or "unknown"),
        "modelVersion": model_version,
        "modelSha256": model_sha256 or artifact.get("modelSha256"),
        "featuresSha256": features_sha256 or artifact.get("featuresSha256"),
        "profile": str(artifact.get("featureProfile") or BASELINE_PROFILE),
        "split": split_name,
        "metrics": {"sampleCount": len(videos), "srcc": srcc, "maeMos": mae},
        "confidenceIntervals": {
            "level": 0.95,
            "method": "paired_percentile_bootstrap_by_video",
            "srcc": [float(srcc_low), float(srcc_high)],
            "maeMos": [float(mae_low), float(mae_high)],
        },
        "bootstrap": {
            "requestedSamples": bootstrap_samples,
            "validSrccSamples": len(boot_srcc),
            "seed": seed,
        },
        "testUsedForSelection": False,
        "selectionRecord": selection_record,
        "selectionRecordPath": selection_record.get("path") if selection_record else None,
        "selectionRecordSha256": selection_record.get("sha256") if selection_record else None,
        "annotationProvenance": provenance,
    }
    if external_only:
        report.update({
            "externalTestOnly": True,
            "trainingAllowed": False,
            "selectionAllowed": False,
            "limitations": [
                "LIVE-VQC 只作为冻结模型的 external_test，不参与训练、特征选择或阈值选择。",
                "LIVE-VQC 与 KoNViD 的 MOS 量表和采集条件不同，跨库结论优先解释 SRCC；MAE 仅作原始量表内记录。",
            ],
        })
    return report


def evaluate_rows(
    rows: list[dict[str, Any]],
    artifact: dict[str, Any],
    provenance: dict[str, Any],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
    model_sha256: str | None = None,
    features_sha256: str | None = None,
    selection_record: dict[str, str] | None = None,
) -> dict[str, Any]:
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    _validate_provenance(provenance)
    profile, feature_columns, legacy_baseline = _artifact_profile(artifact)
    # A pre-metadata baseline may still declare its six-column profile. Treat
    # that artifact as legacy only when it has no candidate marker or input
    # provenance fields at all.
    legacy_baseline = legacy_baseline or (
        not artifact.get("candidateOnly")
        and not artifact.get("inputFeatureProfile")
        and not artifact.get("inputExtractionConfig")
    )
    metadata = _validate_input_metadata(rows, artifact, legacy_baseline=legacy_baseline)
    videos = _test_video_rows(rows, profile, feature_columns)
    report = _evaluate_video_rows(
        videos,
        artifact,
        provenance,
        feature_columns,
        split_name="test",
        schema_version="task3.quality-test-evaluation/1.0",
        count_key="test",
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        model_sha256=model_sha256,
        features_sha256=features_sha256,
        selection_record=selection_record,
    )
    report.update(metadata)
    return report


def evaluate_external_rows(
    rows: list[dict[str, Any]],
    artifact: dict[str, Any],
    provenance: dict[str, Any],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
    model_sha256: str | None = None,
    features_sha256: str | None = None,
    selection_record: dict[str, str] | None = None,
    expected_video_count: int = 585,
) -> dict[str, Any]:
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    if expected_video_count < 1:
        raise ValueError("expected_video_count must be positive")
    _validate_external_provenance(provenance, expected_video_count)
    profile, feature_columns, legacy_baseline = _artifact_profile(artifact)
    legacy_baseline = legacy_baseline or (
        not artifact.get("candidateOnly")
        and not artifact.get("inputFeatureProfile")
        and not artifact.get("inputExtractionConfig")
    )
    metadata = _validate_input_metadata(rows, artifact, legacy_baseline=legacy_baseline)
    _validate_external_row_provenance(rows, provenance)
    videos = _external_video_rows(rows, profile, feature_columns)
    report = _evaluate_video_rows(
        videos,
        artifact,
        provenance,
        feature_columns,
        split_name="external_test",
        schema_version="task3.quality-external-evaluation/1.0",
        count_key="external_test",
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        model_sha256=model_sha256,
        features_sha256=features_sha256,
        selection_record=selection_record,
        external_only=True,
    )
    report.update(metadata)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="Windowed features (.jsonl or .csv)")
    parser.add_argument("--model", type=Path, required=True, help="Frozen joblib model artifact")
    parser.add_argument("--provenance", type=Path, required=True, help="Official annotation provenance JSON")
    parser.add_argument("--output", type=Path, required=True, help="Evaluation report JSON")
    parser.add_argument("--selection-record", type=Path, help="Frozen candidate-selection JSON")
    parser.add_argument(
        "--evaluation-split",
        choices=("test", "external_test"),
        default="test",
        help="Evaluate KoNViD test or an authorized LIVE-VQC external_test manifest",
    )
    parser.add_argument("--expected-external-count", type=int, default=585)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    # Refuse an existing report before opening any input, especially before a
    # test feature table can be read or a model can be asked for predictions.
    args.output = refuse_existing_output(args.output)
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    model_sha256 = _sha256_file(args.model)
    features_sha256 = _sha256_file(args.features)
    artifact = joblib.load(args.model)
    if not isinstance(artifact, dict):
        raise ValueError("model artifact must be a dictionary")
    profile, _, _ = _artifact_profile(artifact)
    selection_info: dict[str, str] | None = None
    if artifact.get("candidateOnly"):
        if args.selection_record is None:
            raise ValueError("candidate artifact requires --selection-record")
        selection_info = _validate_selection_record(
            args.selection_record,
            model_sha256=model_sha256,
            features_sha256=features_sha256,
            profile=profile,
            artifact=artifact,
            match_features=args.evaluation_split != "external_test",
        )
    elif args.selection_record is not None:
        selection_info = _validate_selection_record(
            args.selection_record,
            model_sha256=model_sha256,
            features_sha256=features_sha256,
            profile=profile,
            artifact=artifact,
            match_features=args.evaluation_split != "external_test",
        )

    feature_rows = load_feature_table(args.features)
    if args.evaluation_split == "external_test":
        report = evaluate_external_rows(
            feature_rows,
            artifact,
            provenance,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            model_sha256=model_sha256,
            features_sha256=features_sha256,
            selection_record=selection_info,
            expected_video_count=args.expected_external_count,
        )
    else:
        report = evaluate_rows(
            feature_rows,
            artifact,
            provenance,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            model_sha256=model_sha256,
            features_sha256=features_sha256,
            selection_record=selection_info,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
