"""Audit train/val controlled-event coverage without using test observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEVELOPMENT_SPLITS = {"train", "val"}
EVENT_TYPES = ("blur", "dark", "overexposed", "freeze_suspected")
EXPECTED_PARAMETERS = {
    "blur": {1: {"sigma": 0.8}, 2: {"sigma": 1.5}, 3: {"sigma": 2.5}},
    "dark": {1: {"brightness": -0.2}, 2: {"brightness": -0.4}, 3: {"brightness": -0.6}},
    "overexposed": {1: {"brightness": 0.15}, 2: {"brightness": 0.3}, 3: {"brightness": 0.45}},
    "freeze_suspected": {1: {"freezeDurationS": 1.5}, 2: {"freezeDurationS": 3.0}, 3: {"freezeDurationS": 5.0}},
}


def _development(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("split") or "") in DEVELOPMENT_SPLITS]


def _event_for_center(row: dict[str, Any], start: float, end: float) -> str:
    event = row.get("anomaly")
    if not isinstance(event, dict):
        return "normal"
    center = round((start + end) / 2.0, 6)
    return str(event.get("type")) if float(event.get("startTimeS")) <= center < float(event.get("endTimeS")) else "normal"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_coverage(
    manifest_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    *,
    metadata_sha256: str | None = None,
    verify_files: bool = False,
) -> dict[str, Any]:
    """Return a deterministic train/val report; test rows do not affect it."""
    manifest = _development(manifest_rows)
    features = _development(feature_rows)
    errors: list[str] = []
    by_video: dict[str, dict[str, Any]] = {}
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    split_sources: dict[str, set[str]] = defaultdict(set)
    source_media_info: dict[str, tuple[float, int]] = {}
    record_ids: set[str] = set()

    for row in manifest:
        record_id = str(row.get("recordId") or "")
        video_id = str(row.get("videoId") or "")
        source_id = str(row.get("sourceVideoId") or "")
        split = str(row.get("split") or "")
        if not record_id or not video_id or not source_id:
            errors.append("manifest row is missing recordId, videoId, or sourceVideoId")
            continue
        if record_id in record_ids:
            errors.append(f"duplicate manifest recordId: {record_id}")
        record_ids.add(record_id)
        if video_id in by_video:
            errors.append(f"duplicate manifest videoId: {video_id}")
        by_video[video_id] = row
        by_source[source_id].append(row)
        split_sources[split].add(source_id)
        if metadata_sha256 is not None and row.get("officialMetadataPackageSha256") != metadata_sha256:
            errors.append(f"metadata SHA-256 mismatch: {record_id}")
        if len(str(row.get("videoSha256") or "")) != 64 or len(str(row.get("sourceVideoSha256") or "")) != 64:
            errors.append(f"manifest hash missing or malformed: {record_id}")
        if verify_files:
            path = Path(str(row.get("path") or ""))
            if not path.is_file():
                errors.append(f"video file missing: {record_id}")
            elif _sha256(path) != row.get("videoSha256"):
                errors.append(f"video SHA-256 mismatch: {record_id}")

    for source_id, source_rows in by_source.items():
        splits = {str(row.get("split") or "") for row in source_rows}
        if len(splits) != 1:
            errors.append(f"sourceVideoId crosses split: {source_id}")
        originals = [row for row in source_rows if not bool(row.get("isSynthetic")) and row.get("anomaly") is None]
        if len(originals) != 1:
            errors.append(f"original coverage for {source_id}: expected 1, found {len(originals)}")
        variants = [row for row in source_rows if bool(row.get("isSynthetic"))]
        media_rows = [row for row in variants if row.get("sourceFps") is not None and row.get("sourceFrameCount") is not None]
        if media_rows:
            try:
                fps = float(media_rows[0]["sourceFps"])
                frame_count = int(media_rows[0]["sourceFrameCount"])
                if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
                    raise ValueError
                source_media_info[source_id] = (fps, frame_count)
                for media_row in media_rows[1:]:
                    if float(media_row["sourceFps"]) != fps or int(media_row["sourceFrameCount"]) != frame_count:
                        errors.append(f"inconsistent source media metadata: {source_id}")
            except (TypeError, ValueError):
                errors.append(f"invalid source media metadata: {source_id}")
        expected_keys = {(event_type, strength) for event_type in EVENT_TYPES for strength in (1, 2, 3)}
        actual_keys = {(str(row.get("anomaly", {}).get("type")), int(row.get("strengthLevel", -1))) for row in variants}
        if actual_keys != expected_keys or len(variants) != 12:
            errors.append(f"variant coverage for {source_id}: expected 12 distinct type/strength pairs, found {len(variants)}")
        for row in variants:
            record_id = str(row.get("recordId") or "")
            event = row.get("anomaly")
            strength = row.get("strengthLevel")
            if not isinstance(event, dict) or event.get("type") not in EVENT_TYPES:
                errors.append(f"invalid anomaly metadata: {record_id}")
                continue
            try:
                start = float(event["startTimeS"])
                end = float(event["endTimeS"])
                duration = float(row["sourceDurationS"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"invalid event interval metadata: {record_id}")
                continue
            if not (math.isfinite(start) and math.isfinite(end) and math.isfinite(duration) and 0 <= start < end <= duration + 1e-6):
                errors.append(f"event interval out of bounds: {record_id}")
            try:
                expected = EXPECTED_PARAMETERS[str(event["type"])][int(strength)]
            except (KeyError, TypeError, ValueError):
                errors.append(f"unknown event strength: {record_id}")
            else:
                if row.get("parameters") != expected:
                    errors.append(f"event parameters mismatch: {record_id}")

    feature_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in features:
        video_id = str(row.get("videoId") or "")
        if not video_id:
            errors.append("feature row missing videoId")
            continue
        feature_by_video[video_id].append(row)
        manifest_row = by_video.get(video_id)
        if manifest_row is None:
            errors.append(f"feature videoId absent from development manifest: {video_id}")
            continue
        if str(row.get("split") or "") != str(manifest_row.get("split") or ""):
            errors.append(f"feature split mismatch: {video_id}")
        if str(row.get("sourceVideoId") or "") != str(manifest_row.get("sourceVideoId") or ""):
            errors.append(f"feature sourceVideoId mismatch: {video_id}")
        if row.get("sourceVideoSha256") != manifest_row.get("sourceVideoSha256"):
            errors.append(f"feature sourceVideoSha256 mismatch: {video_id}")
        try:
            start = float(row["windowStartS"])
            end = float(row["windowEndS"])
            if not (math.isfinite(start) and math.isfinite(end) and start <= end):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            errors.append(f"invalid feature window: {video_id}")
            continue
        if row.get("label") != _event_for_center(manifest_row, start, end):
            errors.append(f"feature label disagrees with event center: {video_id}")
        if row.get("groundTruthEvent") != manifest_row.get("anomaly"):
            errors.append(f"feature groundTruthEvent mismatch: {video_id}")

    for video_id in by_video:
        if video_id not in feature_by_video:
            errors.append(f"no feature windows for manifest video: {video_id}")
            continue
        manifest_row = by_video[video_id]
        try:
            media_info = (
                (float(manifest_row["sourceFps"]), int(manifest_row["sourceFrameCount"]))
                if manifest_row.get("sourceFps") is not None and manifest_row.get("sourceFrameCount") is not None
                else source_media_info[str(manifest_row["sourceVideoId"])]
            )
            fps, frame_count = media_info
        except (KeyError, TypeError, ValueError):
            fps, frame_count = 0.0, 0
        if math.isfinite(fps) and fps > 0 and frame_count > 0:
            sample_every = max(1, round(fps * 0.5))
            expected_windows = math.ceil(frame_count / sample_every)
            actual_windows = len(feature_by_video[video_id])
            if actual_windows != expected_windows:
                errors.append(
                    f"feature window count mismatch: {video_id} expected {expected_windows}, found {actual_windows}"
                )

    class_strength = Counter()
    for row in manifest:
        event = row.get("anomaly")
        if isinstance(event, dict):
            class_strength[f"{event.get('type')}__strength_{row.get('strengthLevel')}"] += 1
        else:
            class_strength["normal"] += 1
    return {
        "schemaVersion": "task3.controlled-coverage-audit/1.0",
        "ok": not errors,
        "errors": sorted(set(errors)),
        "manifestCounts": dict(Counter(str(row.get("split") or "") for row in manifest)),
        "sourceGroups": {split: len(values) for split, values in sorted(split_sources.items())},
        "featureCounts": dict(Counter(str(row.get("split") or "") for row in features)),
        "featureVideoGroups": {split: len({str(row.get("videoId") or "") for row in features if row.get("split") == split}) for split in sorted(DEVELOPMENT_SPLITS)},
        "classStrengthCounts": dict(sorted(class_strength.items())),
        "testNotUsed": True,
        "verifyFiles": verify_files,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/konvid-1k/synthetic/full-frame-aligned-v3/manifest-corrected.jsonl"))
    parser.add_argument("--features", type=Path, default=Path("data/raw/konvid-1k/synthetic/full-frame-aligned-v3/event-features-corrected.jsonl"))
    parser.add_argument("--metadata-sha256", default="13af8b028536bf1864361396d6b41c92dc143d359de51eeb76130fc6a48eddbe")
    parser.add_argument("--verify-files", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = audit_coverage(_read_jsonl(args.manifest), _read_jsonl(args.features), metadata_sha256=args.metadata_sha256, verify_files=args.verify_files)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
