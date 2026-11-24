"""Validate and compare blind human reviewability annotations."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from typing import Any

REVIEWABILITY_LABELS = ("reviewable", "not_reviewable", "uncertain")
CONFIDENCE_LEVELS = ("low", "medium", "high")
REVIEW_TASK = "inspect_recorded_visual_evidence"
_ROW_FIELDS = {
    "videoId", "annotatorId", "videoSha256", "videoDurationS", "label", "confidence",
    "reviewTask", "evidenceTimeRanges", "evidenceBasis", "notes", "toolVersion",
    "sourceUrl", "license",
}
_FORBIDDEN_FIELDS = {"mos", "modelScore", "predictedLabel", "predictedConfidence"}
_ARBITRATION_FIELDS = {"videoId", "arbitratorId", "decision", "reason", "resolvedAt", "videoSha256"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _normalized_sha(value: object, field: str = "videoSha256") -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{field} must be a 64-character hexadecimal SHA-256")
    return digest


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def validate_reviewability_row(row: dict[str, object]) -> None:
    """Validate one completed human row; raise ValueError on any schema violation."""
    if not isinstance(row, dict):
        raise ValueError("annotation row must be an object")
    forbidden = sorted(_FORBIDDEN_FIELDS & set(row))
    if forbidden:
        raise ValueError("MOS/model fields are not human ground truth: " + ", ".join(forbidden))
    missing = sorted(_ROW_FIELDS - set(row))
    extra = sorted(set(row) - _ROW_FIELDS)
    if missing:
        raise ValueError("annotation row missing: " + ", ".join(missing))
    if extra:
        raise ValueError("annotation row has unsupported fields: " + ", ".join(extra))
    for field in ("videoId", "annotatorId", "sourceUrl", "license", "evidenceBasis", "toolVersion"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    if str(row["reviewTask"]) != REVIEW_TASK:
        raise ValueError(f"reviewTask must be {REVIEW_TASK}")
    if row["label"] not in REVIEWABILITY_LABELS:
        raise ValueError(f"unsupported reviewability label: {row['label']}")
    if row["confidence"] not in CONFIDENCE_LEVELS:
        raise ValueError("confidence must be low, medium, or high")
    duration = _finite_number(row["videoDurationS"], "videoDurationS")
    if duration <= 0:
        raise ValueError("videoDurationS must be positive")
    _normalized_sha(row["videoSha256"])
    ranges = row["evidenceTimeRanges"]
    if not isinstance(ranges, list):
        raise ValueError("evidenceTimeRanges must be a list of [startS, endS] pairs")
    for index, interval in enumerate(ranges):
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError(f"evidenceTimeRanges[{index}] must be a [startS, endS] pair")
        start = _finite_number(interval[0], f"evidenceTimeRanges[{index}].startS")
        end = _finite_number(interval[1], f"evidenceTimeRanges[{index}].endS")
        if not 0 <= start < end <= duration:
            raise ValueError(f"evidenceTimeRanges[{index}] must satisfy 0 <= start < end <= videoDurationS")
    if not isinstance(row["notes"], str):
        raise ValueError("notes must be a string")


def _manifest_row(item: dict[str, object]) -> dict[str, object]:
    video_id = str(item.get("videoId") or "").strip()
    if not video_id:
        raise ValueError("manifest row requires videoId")
    duration_value = item.get("durationS", item.get("videoDurationS"))
    duration = _finite_number(duration_value, f"{video_id}: durationS")
    if duration <= 0:
        raise ValueError(f"{video_id}: durationS must be positive")
    source_url, license_text = str(item.get("sourceUrl") or "").strip(), str(item.get("license") or "").strip()
    if not source_url or not license_text:
        raise ValueError(f"{video_id}: manifest requires sourceUrl and license")
    return {
        "videoId": video_id,
        "videoSha256": _normalized_sha(item.get("videoSha256")),
        "videoDurationS": duration,
        "sourceUrl": source_url,
        "license": license_text,
    }


def prepare_blind_batches(
    manifest_rows: list[dict[str, object]], *, seed: int, annotators: tuple[str, str]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return two same-video blank batches with separately randomized order."""
    import random

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if len(annotators) != 2 or not all(isinstance(value, str) and value.strip() for value in annotators):
        raise ValueError("annotators must contain two non-empty IDs")
    if annotators[0] == annotators[1]:
        raise ValueError("distinct annotators are required")
    if not manifest_rows:
        raise ValueError("manifest must be non-empty")
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in manifest_rows:
        record = _manifest_row(item)
        if record["videoId"] in seen:
            raise ValueError("manifest contains duplicate videoId")
        seen.add(str(record["videoId"]))
        normalized.append(record)

    def blank_batch(annotator_id: str, order_seed: int) -> list[dict[str, object]]:
        records = [
            {
                **record,
                "annotatorId": annotator_id,
                "label": "",
                "confidence": "",
                "reviewTask": REVIEW_TASK,
                "evidenceTimeRanges": [],
                "evidenceBasis": "",
                "notes": "",
                "toolVersion": "task3-reviewability/1.0",
            }
            for record in normalized
        ]
        random.Random(order_seed).shuffle(records)
        return records

    first = blank_batch(annotators[0], seed)
    second = blank_batch(annotators[1], seed ^ 0x5A17)
    if len(first) > 1 and [row["videoId"] for row in first] == [row["videoId"] for row in second]:
        second = second[1:] + second[:1]
    return first, second


def _single_annotator_id(rows: list[dict[str, object]]) -> str:
    if not rows:
        raise ValueError("annotation batches must be non-empty")
    identities = {str(row.get("annotatorId") or "").strip() for row in rows}
    if len(identities) != 1 or not next(iter(identities)):
        raise ValueError("each batch must contain exactly one annotatorId")
    return next(iter(identities))


def _index_unique_videos(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for row in rows:
        validate_reviewability_row(row)
        video_id = str(row["videoId"])
        if video_id in indexed:
            raise ValueError(f"duplicate videoId: {video_id}")
        indexed[video_id] = row
    return indexed


def _agreement_report(
    confusion: dict[str, dict[str, int]], conflicts: list[dict[str, object]], annotators: tuple[str, str]
) -> dict[str, object]:
    pair_count = sum(sum(values.values()) for values in confusion.values())
    diagonal = sum(confusion[label][label] for label in REVIEWABILITY_LABELS)
    observed = diagonal / pair_count if pair_count else None
    expected = None
    if pair_count:
        row_totals = {label: sum(confusion[label].values()) for label in REVIEWABILITY_LABELS}
        column_totals = {
            label: sum(confusion[other][label] for other in REVIEWABILITY_LABELS)
            for label in REVIEWABILITY_LABELS
        }
        expected = sum(row_totals[label] * column_totals[label] for label in REVIEWABILITY_LABELS) / (pair_count * pair_count)
    kappa = None if expected is None or math.isclose(1.0 - expected, 0.0) else (observed - expected) / (1.0 - expected)
    uncertain_count = sum(
        confusion[left][right]
        for left in REVIEWABILITY_LABELS
        for right in REVIEWABILITY_LABELS
        if left == "uncertain" or right == "uncertain"
    )
    return {
        "schemaVersion": "task3.reviewability-agreement/1.0",
        "annotatorIds": list(annotators),
        "pairCount": pair_count,
        "confusionMatrix": confusion,
        "rawAgreement": observed,
        "kappa": kappa,
        "classAgreement": {label: confusion[label][label] for label in REVIEWABILITY_LABELS},
        "uncertainPairCount": uncertain_count,
        "conflicts": conflicts,
        "limitations": [
            "Cohen kappa is sensitive to prevalence and class imbalance.",
            "This agreement report is not evidence that MOS predicts human reviewability.",
        ],
    }


def compare_annotators(rows_a: list[dict[str, object]], rows_b: list[dict[str, object]]) -> dict[str, object]:
    """Return the agreement report and every sample requiring adjudication."""
    a_id, b_id = _single_annotator_id(rows_a), _single_annotator_id(rows_b)
    if a_id == b_id:
        raise ValueError("distinct annotators are required")
    by_a, by_b = _index_unique_videos(rows_a), _index_unique_videos(rows_b)
    if set(by_a) != set(by_b):
        raise ValueError("annotators must cover the same video set")
    confusion = {left: {right: 0 for right in REVIEWABILITY_LABELS} for left in REVIEWABILITY_LABELS}
    conflicts: list[dict[str, object]] = []
    for video_id in sorted(by_a):
        left, right = by_a[video_id], by_b[video_id]
        if _normalized_sha(left["videoSha256"]) != _normalized_sha(right["videoSha256"]):
            raise ValueError("annotator video hashes disagree")
        if _finite_number(left["videoDurationS"], "videoDurationS") != _finite_number(right["videoDurationS"], "videoDurationS"):
            raise ValueError("annotator video durations disagree")
        if str(left["sourceUrl"]).strip() != str(right["sourceUrl"]).strip():
            raise ValueError("annotator source URLs disagree")
        if str(left["license"]).strip() != str(right["license"]).strip():
            raise ValueError("annotator licenses disagree")
        confusion[str(left["label"])][str(right["label"])] += 1
        if left["label"] != right["label"] or "uncertain" in (left["label"], right["label"]):
            conflicts.append({"videoId": video_id, "status": "arbitration_required"})
    return _agreement_report(confusion, conflicts, (a_id, b_id))


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalized_annotation(row: dict[str, object]) -> dict[str, object]:
    normalized = dict(row)
    normalized["videoSha256"] = _normalized_sha(row["videoSha256"])
    normalized["videoDurationS"] = _finite_number(row["videoDurationS"], "videoDurationS")
    normalized["sourceUrl"] = str(row["sourceUrl"]).strip()
    normalized["license"] = str(row["license"]).strip()
    normalized["evidenceTimeRanges"] = [
        [_finite_number(interval[0], "evidenceTimeRanges.startS"), _finite_number(interval[1], "evidenceTimeRanges.endS")]
        for interval in row["evidenceTimeRanges"]
    ]
    return normalized


def _validate_arbitration(record: dict[str, object], left: dict[str, object], right: dict[str, object]) -> None:
    if set(record) != _ARBITRATION_FIELDS:
        raise ValueError("arbitration record must contain exactly the required fields")
    if record["decision"] not in ("reviewable", "not_reviewable"):
        raise ValueError("arbitration decision must be reviewable or not_reviewable")
    arbitrator = str(record["arbitratorId"] or "").strip()
    annotators = {str(left["annotatorId"]).strip(), str(right["annotatorId"]).strip()}
    if not arbitrator or arbitrator in annotators:
        raise ValueError("arbitrator must be a third party distinct from both annotators")
    if not str(record["reason"] or "").strip():
        raise ValueError("arbitration reason must be non-empty")
    if not str(record["videoId"] or "").strip() or record["videoId"] != left["videoId"]:
        raise ValueError("arbitration videoId must match exactly one conflict")
    if _normalized_sha(record["videoSha256"]) != _normalized_sha(left["videoSha256"]):
        raise ValueError("arbitration video hash must match both annotation rows")
    try:
        resolved_at = datetime.fromisoformat(str(record["resolvedAt"]).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("resolvedAt must be an ISO-8601 timestamp") from exc
    if resolved_at.tzinfo is None:
        raise ValueError("resolvedAt must include a timezone")


def resolve_adjudicated_labels(
    rows_a: list[dict[str, object]], rows_b: list[dict[str, object]], adjudications: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Return one provenance-bearing final row per resolved video; otherwise raise ValueError."""
    compare_annotators(rows_a, rows_b)
    a_id, b_id = _single_annotator_id(rows_a), _single_annotator_id(rows_b)
    by_a, by_b = _index_unique_videos(rows_a), _index_unique_videos(rows_b)
    conflicts = {
        video_id for video_id in by_a
        if by_a[video_id]["label"] != by_b[video_id]["label"] or "uncertain" in (by_a[video_id]["label"], by_b[video_id]["label"])
    }
    arbitration_by_video: dict[str, dict[str, object]] = {}
    for record in adjudications:
        if not isinstance(record, dict):
            raise ValueError("arbitration records must be objects")
        video_id = str(record.get("videoId") or "")
        if video_id in arbitration_by_video:
            raise ValueError(f"each conflict must have exactly one arbitration record: {video_id}")
        if video_id not in conflicts:
            raise ValueError(f"arbitration record does not match a conflict: {video_id}")
        _validate_arbitration(record, by_a[video_id], by_b[video_id])
        arbitration_by_video[video_id] = record
    if set(arbitration_by_video) != conflicts:
        raise ValueError("each conflict must have exactly one arbitration record")

    final_rows: list[dict[str, object]] = []
    for video_id in sorted(by_a):
        left, right = by_a[video_id], by_b[video_id]
        source_rows = sorted((_normalized_annotation(left), _normalized_annotation(right)), key=lambda row: str(row["annotatorId"]))
        row_hashes = [_canonical_hash(row) for row in source_rows]
        if video_id in conflicts:
            arbitration = arbitration_by_video[video_id]
            resolution_method = "third_party_arbitration"
            normalized_arbitration = dict(arbitration)
            normalized_arbitration["arbitratorId"] = str(arbitration["arbitratorId"]).strip()
            normalized_arbitration["videoId"] = str(arbitration["videoId"]).strip()
            normalized_arbitration["videoSha256"] = _normalized_sha(arbitration["videoSha256"])
            resolution_id = _canonical_hash(normalized_arbitration)
            label = str(arbitration["decision"])
            evidence_basis = str(arbitration["reason"])
            arbitration_hash: str | None = resolution_id
        else:
            resolution_method = "dual_agreement"
            resolution_id = _canonical_hash({"annotatorRows": source_rows})
            label = str(left["label"])
            evidence_basis = str(left["evidenceBasis"])
            arbitration_hash = None
        result: dict[str, object] = {
            "videoId": video_id,
            "videoSha256": _normalized_sha(left["videoSha256"]),
            "videoDurationS": left["videoDurationS"],
            "sourceUrl": str(left["sourceUrl"]).strip(),
            "license": str(left["license"]).strip(),
            "label": label,
            "confidence": left["confidence"] if left["confidence"] == right["confidence"] else "medium",
            "reviewTask": REVIEW_TASK,
            "evidenceTimeRanges": left["evidenceTimeRanges"] if left["evidenceTimeRanges"] == right["evidenceTimeRanges"] else [],
            "evidenceBasis": evidence_basis,
            "resolutionMethod": resolution_method,
            "resolutionId": resolution_id,
            "provenance": {
                "annotatorIds": [a_id, b_id],
                "annotatorRowHashes": row_hashes,
                "arbitrationRecordHash": arbitration_hash,
            },
        }
        final_rows.append(result)
    return final_rows

