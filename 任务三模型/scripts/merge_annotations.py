"""Validate and merge two independent human annotation files; never invent labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from prepare_annotation import validate_annotation_row


def interval_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return overlap / union if union > 0 else 0.0


def _read(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _clean(rows: Iterable[dict[str, Any]], expected_annotator: str) -> list[dict[str, Any]]:
    cleaned = []
    for row in rows:
        if not row.get("label"):
            raise ValueError(f"blank annotation row remains for {row.get('videoId')}")
        if str(row.get("annotatorId")) != expected_annotator:
            raise ValueError("annotation file contains an unexpected annotatorId")
        validate_annotation_row(row)
        cleaned.append(row)
    return cleaned


def merge_annotations(first: list[dict[str, Any]], second: list[dict[str, Any]], iou_threshold: float = 0.5) -> dict[str, Any]:
    if not first or not second:
        raise ValueError("both annotator files must contain completed rows")
    a_id, b_id = str(first[0]["annotatorId"]), str(second[0]["annotatorId"])
    if a_id == b_id:
        raise ValueError("annotator IDs must be distinct")
    first = _clean(first, a_id)
    second = _clean(second, b_id)
    video_ids = sorted({str(row["videoId"]) for row in first + second})
    conflicts = []
    agreements = 0
    for video_id in video_ids:
        left = [row for row in first if str(row["videoId"]) == video_id]
        right = [row for row in second if str(row["videoId"]) == video_id]
        if not left or not right:
            conflicts.append({"videoId": video_id, "status": "arbitration_required", "reason": "missing_annotator"})
            continue
        used = set()
        for left_row in left:
            best = None
            for index, right_row in enumerate(right):
                if index in used:
                    continue
                score = interval_iou((float(left_row["startTimeS"]), float(left_row["endTimeS"])), (float(right_row["startTimeS"]), float(right_row["endTimeS"])))
                candidate = (score, index, right_row)
                if best is None or candidate[0] > best[0]:
                    best = candidate
            if best is None:
                conflicts.append({"videoId": video_id, "status": "arbitration_required", "reason": "unmatched_interval"})
                continue
            score, index, right_row = best
            used.add(index)
            if str(left_row["label"]) == str(right_row["label"]) and score >= iou_threshold:
                agreements += 1
            else:
                conflicts.append({
                    "videoId": video_id,
                    "status": "arbitration_required",
                    "reason": "label_or_interval_disagreement",
                    "leftLabel": left_row["label"],
                    "rightLabel": right_row["label"],
                    "intervalIoU": score,
                })
        for index in range(len(right)):
            if index not in used:
                conflicts.append({"videoId": video_id, "status": "arbitration_required", "reason": "unmatched_interval"})
    total = agreements + len(conflicts)
    return {
        "schemaVersion": "task3.double-annotation-report/1.0",
        "annotators": [a_id, b_id],
        "summary": {
            "videoCount": len(video_ids),
            "agreementCount": agreements,
            "conflictCount": len(conflicts),
            "labelIntervalAgreementRate": agreements / total if total else 0.0,
            "formalLabelsReady": False if conflicts else True,
        },
        "conflicts": conflicts,
        "limitations": ["没有仲裁记录前不得把合并结果当作正式真实标签集。"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = merge_annotations(_read(args.annotator_a), _read(args.annotator_b))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
