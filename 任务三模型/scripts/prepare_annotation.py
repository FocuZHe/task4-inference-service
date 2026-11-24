"""Prepare auditable, blank double-annotation batches without creating labels."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Iterable

LABELS = ("normal", "blur", "dark", "overexposed", "freeze_suspected")
CONFIDENCES = ("low", "medium", "high")


def validate_annotation_row(row: dict[str, Any]) -> None:
    required = {"videoId", "annotatorId", "label", "startTimeS", "endTimeS", "confidence", "videoDurationS", "videoSha256"}
    missing = sorted(required - set(row))
    if missing:
        raise ValueError("annotation row missing: " + ", ".join(missing))
    if str(row["label"]) not in LABELS:
        raise ValueError(f"unsupported annotation label: {row['label']}")
    if str(row["confidence"]) not in CONFIDENCES:
        raise ValueError("confidence must be low, medium, or high")
    try:
        start, end, duration = float(row["startTimeS"]), float(row["endTimeS"]), float(row["videoDurationS"])
    except (TypeError, ValueError) as exc:
        raise ValueError("annotation times must be numeric") from exc
    if not (0 <= start < end <= duration):
        raise ValueError("annotation interval must satisfy 0 <= start < end <= duration")
    sha = str(row["videoSha256"]).lower()
    if len(sha) != 64 or any(character not in "0123456789abcdef" for character in sha):
        raise ValueError("videoSha256 must be a 64-character hexadecimal SHA-256")


def _blank_row(item: dict[str, Any], annotator: str, batch: int) -> dict[str, Any]:
    duration = float(item["durationS"])
    if duration <= 0:
        raise ValueError(f"{item['videoId']}: durationS must be positive")
    sha = str(item.get("videoSha256") or "")
    if len(sha) != 64 or any(character not in "0123456789abcdefABCDEF" for character in sha):
        raise ValueError(f"{item['videoId']}: manifest requires videoSha256")
    if not str(item.get("sourceUrl") or "").strip() or not str(item.get("license") or "").strip():
        raise ValueError(f"{item['videoId']}: manifest requires sourceUrl and license")
    return {
        "batch": batch,
        "videoId": str(item["videoId"]),
        "annotatorId": annotator,
        "sourceUrl": str(item.get("sourceUrl") or ""),
        "license": str(item.get("license") or ""),
        "path": str(item.get("path") or ""),
        "videoSha256": sha,
        "videoDurationS": duration,
        "clipStartS": 0.0,
        "clipEndS": min(10.0, duration),
        "label": "",
        "startTimeS": "",
        "endTimeS": "",
        "confidence": "",
        "notes": "",
        "annotationToolVersion": "task3-double-annotation/1.0",
    }


def build_batches(manifest: Iterable[dict[str, Any]], count: int = 100, seed: int = 42, annotators: tuple[str, str] = ("annotator-a", "annotator-b")) -> list[dict[str, Any]]:
    if count < 1 or len(annotators) != 2 or annotators[0] == annotators[1]:
        raise ValueError("count must be positive and annotators must contain two distinct IDs")
    records = list(manifest)
    by_id = {str(item.get("videoId")): item for item in records}
    if len(by_id) != len(records):
        raise ValueError("manifest contains duplicate videoId")
    low_motion = [
        item for item in records
        if bool(item.get("lowMotionCandidate", item.get("isLowMotionCandidate")))
    ]
    if len(low_motion) < min(30, count):
        raise ValueError(f"manifest has {len(low_motion)} low-motion candidates; need {min(30, count)}")
    if len(records) < count:
        raise ValueError(f"manifest has {len(records)} videos; need {count}")
    rng = random.Random(seed)
    selected = list(records)
    selected.sort(key=lambda item: str(item["videoId"]))
    low_ids = {str(item["videoId"]) for item in low_motion}
    required_low = min(30, count)
    selected_low = [item for item in selected if str(item["videoId"]) in low_ids][:required_low]
    remainder = [item for item in selected if str(item["videoId"]) not in low_ids]
    rng.shuffle(remainder)
    selected = selected_low + remainder[: count - len(selected_low)]
    rng.shuffle(selected)
    rows = []
    for index, item in enumerate(selected, start=1):
        for annotator in annotators:
            rows.append(_blank_row(item, annotator, index))
    return rows


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_templates(rows: list[dict[str, Any]], jsonl_path: Path, csv_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    fields = list(rows[0]) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--annotator-a-jsonl", type=Path, required=True)
    parser.add_argument("--annotator-a-csv", type=Path, required=True)
    parser.add_argument("--annotator-b-jsonl", type=Path, required=True)
    parser.add_argument("--annotator-b-csv", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = build_batches(read_manifest(args.manifest), args.count, args.seed)
    for annotator, jsonl_path, csv_path in (
        ("annotator-a", args.annotator_a_jsonl, args.annotator_a_csv),
        ("annotator-b", args.annotator_b_jsonl, args.annotator_b_csv),
    ):
        write_templates([row for row in rows if row["annotatorId"] == annotator], jsonl_path, csv_path)
    print(f"wrote {len(rows)} blank annotation rows in two independent files for {args.count} videos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
