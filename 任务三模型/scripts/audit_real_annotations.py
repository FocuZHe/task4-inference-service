"""Audit readiness of a real-video double-annotation batch.

This module is read-only. It never creates labels, repairs annotation rows, or
replaces missing real data with synthetic or public-dataset records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from merge_annotations import merge_annotations
from prepare_annotation import validate_annotation_row


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".json":
        values = json.loads(path.read_text(encoding="utf-8"))
    elif suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            values = list(csv.DictReader(handle))
    else:
        raise ValueError(f"unsupported record format: {path.suffix}")
    if not isinstance(values, list) or not all(isinstance(value, dict) for value in values):
        raise ValueError(f"{path} must contain a list of objects")
    return [dict(value) for value in values]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _issue(issues: list[dict[str, Any]], code: str, message: str, **details: Any) -> None:
    item: dict[str, Any] = {"code": code, "message": message}
    item.update(details)
    issues.append(item)


def _manifest_path(value: Any, manifest_path: Path) -> Path:
    candidate = Path(str(value or ""))
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    return candidate.resolve()


def audit_real_annotations(
    manifest_path: Path,
    annotator_a_path: Path,
    annotator_b_path: Path,
    *,
    arbitration_path: Path | None = None,
    minimum_videos: int = 100,
) -> dict[str, Any]:
    """Return a deterministic readiness report without mutating any input."""

    issues: list[dict[str, Any]] = []
    checks = {
        "minimumManifestCount": False,
        "manifestFilesAndHashes": False,
        "independentAnnotators": False,
        "validAnnotationRows": False,
        "completePairedAnnotations": False,
        "conflictsArbitrated": False,
    }
    summary = {
        "manifestVideoCount": 0,
        "annotatorARowCount": 0,
        "annotatorBRowCount": 0,
        "annotatorAVideoCount": 0,
        "annotatorBVideoCount": 0,
        "annotatedVideoCount": 0,
        "conflictCount": 0,
        "arbitrationRecordCount": 0,
        "videoHashMismatchCount": 0,
    }
    if not isinstance(minimum_videos, int) or isinstance(minimum_videos, bool) or minimum_videos < 1:
        raise ValueError("minimum_videos must be a positive integer")

    manifest_rows: list[dict[str, Any]] = []
    try:
        manifest_rows = _read_records(manifest_path)
    except FileNotFoundError:
        _issue(issues, "missing_input", "manifest file does not exist", path=str(manifest_path))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _issue(issues, "invalid_input", f"cannot read manifest: {exc}", path=str(manifest_path))

    manifest_by_id: dict[str, dict[str, Any]] = {}
    manifest_file_hashes_ok = bool(manifest_rows)
    for index, row in enumerate(manifest_rows):
        video_id = str(row.get("videoId") or "").strip()
        if not video_id:
            manifest_file_hashes_ok = False
            _issue(issues, "invalid_manifest_row", "manifest row has no videoId", rowIndex=index)
            continue
        if video_id in manifest_by_id:
            manifest_file_hashes_ok = False
            _issue(issues, "duplicate_manifest_video", "manifest contains duplicate videoId", videoId=video_id)
            continue
        manifest_by_id[video_id] = row
        source_url = str(row.get("sourceUrl") or "").strip()
        license_text = str(row.get("license") or "").strip()
        digest = str(row.get("videoSha256") or "").lower()
        duration = row.get("durationS", row.get("videoDurationS"))
        if not source_url or not license_text or not _valid_sha256(digest):
            manifest_file_hashes_ok = False
            _issue(issues, "invalid_manifest_provenance", "manifest requires sourceUrl, license and SHA-256", videoId=video_id)
        try:
            duration_value = float(duration)
        except (TypeError, ValueError):
            duration_value = float("nan")
        if not math.isfinite(duration_value) or duration_value <= 0:
            manifest_file_hashes_ok = False
            _issue(issues, "invalid_manifest_duration", "manifest durationS must be finite and positive", videoId=video_id)
        path = _manifest_path(row.get("path"), manifest_path)
        if not path.is_file():
            manifest_file_hashes_ok = False
            _issue(issues, "missing_video_file", "manifest video file does not exist", videoId=video_id, path=str(path))
        elif _valid_sha256(digest):
            try:
                actual = _sha256_file(path)
            except OSError as exc:
                manifest_file_hashes_ok = False
                _issue(issues, "unreadable_video_file", f"cannot read video file: {exc}", videoId=video_id, path=str(path))
                continue
            if actual != digest:
                manifest_file_hashes_ok = False
                summary["videoHashMismatchCount"] += 1
                _issue(issues, "manifest_hash_mismatch", "video SHA-256 does not match manifest", videoId=video_id)
    summary["manifestVideoCount"] = len(manifest_by_id)
    checks["minimumManifestCount"] = len(manifest_by_id) >= minimum_videos
    if not checks["minimumManifestCount"]:
        _issue(issues, "insufficient_manifest", f"need at least {minimum_videos} manifest videos", actual=len(manifest_by_id))
    checks["manifestFilesAndHashes"] = manifest_file_hashes_ok and bool(manifest_by_id)

    def load_annotation_file(path: Path, name: str) -> tuple[list[dict[str, Any]], set[str], set[str], bool]:
        try:
            rows = _read_records(path)
        except FileNotFoundError:
            _issue(issues, "missing_input", f"{name} annotation file does not exist", path=str(path))
            return [], set(), set(), False
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _issue(issues, "invalid_input", f"cannot read {name} annotations: {exc}", path=str(path))
            return [], set(), set(), False
        if not rows:
            _issue(issues, "empty_annotations", f"{name} annotation file is empty", path=str(path))
            return [], set(), set(), False
        annotator_ids = {str(row.get("annotatorId") or "").strip() for row in rows}
        annotator_ids.discard("")
        video_ids: set[str] = set()
        valid = True
        for index, row in enumerate(rows):
            video_id = str(row.get("videoId") or "").strip()
            if video_id:
                video_ids.add(video_id)
            try:
                validate_annotation_row(row)
            except (TypeError, ValueError) as exc:
                valid = False
                _issue(issues, "invalid_annotation_row", str(exc), annotatorFile=name, rowIndex=index, videoId=video_id)
                continue
            manifest = manifest_by_id.get(video_id)
            if manifest is None:
                valid = False
                _issue(issues, "annotation_video_not_in_manifest", "annotation videoId is absent from manifest", videoId=video_id)
                continue
            expected_hash = str(manifest.get("videoSha256") or "").lower()
            if str(row.get("videoSha256") or "").lower() != expected_hash:
                valid = False
                summary["videoHashMismatchCount"] += 1
                _issue(issues, "annotation_hash_mismatch", "annotation SHA-256 does not match manifest", videoId=video_id)
            try:
                expected_duration = float(manifest.get("durationS", manifest.get("videoDurationS")))
                actual_duration = float(row["videoDurationS"])
            except (TypeError, ValueError, KeyError):
                expected_duration = actual_duration = float("nan")
            if not math.isfinite(expected_duration) or not math.isfinite(actual_duration) or abs(expected_duration - actual_duration) > 1e-6:
                valid = False
                _issue(issues, "annotation_duration_mismatch", "annotation duration does not match manifest", videoId=video_id)
        if len(annotator_ids) != 1:
            valid = False
            _issue(issues, "invalid_annotator_identity", f"{name} must contain exactly one annotatorId", annotatorFile=name)
        return rows, video_ids, annotator_ids, valid

    rows_a, video_ids_a, annotator_ids_a, valid_a = load_annotation_file(annotator_a_path, "annotator-a")
    rows_b, video_ids_b, annotator_ids_b, valid_b = load_annotation_file(annotator_b_path, "annotator-b")
    summary["annotatorARowCount"] = len(rows_a)
    summary["annotatorBRowCount"] = len(rows_b)
    summary["annotatorAVideoCount"] = len(video_ids_a)
    summary["annotatorBVideoCount"] = len(video_ids_b)
    summary["annotatedVideoCount"] = len(video_ids_a & video_ids_b)
    a_id = next(iter(annotator_ids_a), "")
    b_id = next(iter(annotator_ids_b), "")
    checks["independentAnnotators"] = bool(a_id and b_id and a_id != b_id and len(annotator_ids_a) == 1 and len(annotator_ids_b) == 1)
    if not checks["independentAnnotators"]:
        _issue(issues, "annotators_not_independent", "the two annotation files must have different single annotator IDs")
    checks["validAnnotationRows"] = valid_a and valid_b and bool(rows_a) and bool(rows_b)
    checks["completePairedAnnotations"] = (
        checks["minimumManifestCount"]
        and checks["validAnnotationRows"]
        and video_ids_a == video_ids_b
        and video_ids_a <= set(manifest_by_id)
        and len(video_ids_a) >= minimum_videos
    )
    if not checks["completePairedAnnotations"]:
        _issue(issues, "incomplete_annotation_pair", "both annotators must cover the same manifest video batch")

    conflicts: list[dict[str, Any]] = []
    merge_completed = False
    if checks["validAnnotationRows"] and checks["independentAnnotators"] and video_ids_a and video_ids_b:
        try:
            conflicts = list(merge_annotations(rows_a, rows_b).get("conflicts", []))
            merge_completed = True
        except (TypeError, ValueError, KeyError) as exc:
            _issue(issues, "merge_failed", f"cannot compare annotations: {exc}")
    summary["conflictCount"] = len(conflicts)

    arbitration_records: list[dict[str, Any]] = []
    if not conflicts:
        checks["conflictsArbitrated"] = merge_completed
    elif arbitration_path is None:
        _issue(issues, "missing_arbitration", "annotation conflicts require a third-party arbitration file", count=len(conflicts))
    else:
        try:
            arbitration_records = _read_records(arbitration_path)
        except FileNotFoundError:
            _issue(issues, "missing_arbitration", "arbitration file does not exist", path=str(arbitration_path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _issue(issues, "invalid_arbitration", f"cannot read arbitration file: {exc}", path=str(arbitration_path))
        summary["arbitrationRecordCount"] = len(arbitration_records)
        expected_keys = {(str(conflict.get("videoId")), index) for index, conflict in enumerate(conflicts)}
        seen_keys: set[tuple[str, int]] = set()
        arbitration_valid = True
        for index, record in enumerate(arbitration_records):
            video_id = str(record.get("videoId") or "").strip()
            conflict_index = record.get("conflictIndex")
            arbitrator_id = str(record.get("arbitratorId") or "").strip()
            decision = str(record.get("decision") or "").strip()
            key = (video_id, conflict_index) if isinstance(conflict_index, int) and not isinstance(conflict_index, bool) else None
            if key is None or key not in expected_keys or key in seen_keys:
                arbitration_valid = False
                _issue(issues, "invalid_arbitration_record", "arbitration record does not identify one unique conflict", rowIndex=index)
                continue
            seen_keys.add(key)
            if not arbitrator_id or arbitrator_id in {a_id, b_id} or not decision:
                arbitration_valid = False
                _issue(issues, "invalid_arbitration_record", "arbitration needs a third-party ID and non-empty decision", rowIndex=index, videoId=video_id)
        missing_keys = expected_keys - seen_keys
        for video_id, conflict_index in sorted(missing_keys):
            arbitration_valid = False
            _issue(issues, "missing_arbitration", "conflict has no arbitration record", videoId=video_id, conflictIndex=conflict_index)
        checks["conflictsArbitrated"] = arbitration_valid and not missing_keys and len(arbitration_records) == len(expected_keys)

    return {
        "schemaVersion": "task3.real-annotation-readiness/1.0",
        "status": "ready" if all(checks.values()) else "not_ready",
        "ready": all(checks.values()),
        "minimumVideos": minimum_videos,
        "checks": checks,
        "summary": summary,
        "annotators": [a_id, b_id] if a_id or b_id else [],
        "issues": issues,
        "inputs": {
            "manifest": str(manifest_path),
            "annotatorA": str(annotator_a_path),
            "annotatorB": str(annotator_b_path),
            "arbitration": str(arbitration_path) if arbitration_path else None,
        },
        "limitations": [
            "该报告只证明双人标注输入满足门控条件，不生成或修正真实标签。",
            "没有正式仲裁后的最终标签文件和直播链路监控时，不得宣称真实故障指标已完成。",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--arbitration", type=Path)
    parser.add_argument("--minimum-videos", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true", help="return exit code 2 when the gate is not ready")
    args = parser.parse_args(argv)
    report = audit_real_annotations(
        args.manifest,
        args.annotator_a,
        args.annotator_b,
        arbitration_path=args.arbitration,
        minimum_videos=args.minimum_videos,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
