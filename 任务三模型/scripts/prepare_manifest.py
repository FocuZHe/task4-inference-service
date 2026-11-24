"""Build a reproducible video manifest for Task 3 quality experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import zipfile
from pathlib import Path
from typing import Any, Iterable


VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
ID_FIELDS = ("video_id", "videoId", "video", "filename", "file_name", "file", "name", "id")
MOS_FIELDS = ("mos", "MOS", "mean_opinion_score", "meanOpinionScore", "quality_score")
PARENT_FIELDS = ("parent_video_id", "parentVideoId", "source_video_id", "sourceVideoId")


def _first_value(row: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return value
    return None


def _video_id(value: Any) -> str:
    if value is None:
        raise ValueError("metadata row is missing video_id")
    return Path(str(value).strip()).stem


def _mos(value: Any) -> float | None:
    if value is None or not str(value).strip():
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"MOS must be finite, got {result}")
    return result


def _read_metadata_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if not name.endswith("/")]
            by_basename = {Path(name).name: name for name in names}
            konvid_attributes = "KoNViD_1k_attributes.csv"
            konvid_mos = "KoNViD_1k_mos.csv"
            if konvid_attributes in by_basename and konvid_mos in by_basename:
                attributes = list(
                    csv.DictReader(
                        archive.read(by_basename[konvid_attributes]).decode("utf-8-sig").splitlines()
                    )
                )
                mos_rows = list(
                    csv.DictReader(
                        archive.read(by_basename[konvid_mos]).decode("utf-8-sig").splitlines()
                    )
                )
                attributes_by_id: dict[str, dict[str, str]] = {}
                mos_by_id: dict[str, dict[str, str]] = {}
                for rows, indexed, table_name in (
                    (attributes, attributes_by_id, konvid_attributes),
                    (mos_rows, mos_by_id, konvid_mos),
                ):
                    for row in rows:
                        flickr_id = str(row.get("flickr_id") or "").strip()
                        if not flickr_id:
                            raise ValueError(f"{table_name} contains a row without flickr_id")
                        if flickr_id in indexed:
                            raise ValueError(f"{table_name} contains duplicate flickr_id: {flickr_id}")
                        indexed[flickr_id] = row
                if attributes_by_id.keys() != mos_by_id.keys():
                    raise ValueError("KoNViD attributes and MOS tables have different flickr_id sets")

                normalized = []
                for flickr_id, attribute in attributes_by_id.items():
                    filename = str(attribute.get("file_name") or "").strip()
                    if not filename:
                        raise ValueError(f"KoNViD attributes missing file_name for flickr_id {flickr_id}")
                    attribute_mos = _mos(attribute.get("MOS"))
                    precise_mos = _mos(mos_by_id[flickr_id].get("mos"))
                    if attribute_mos is None or precise_mos is None:
                        raise ValueError(f"KoNViD MOS missing for flickr_id {flickr_id}")
                    if not 1 <= attribute_mos <= 5 or not 1 <= precise_mos <= 5:
                        raise ValueError(f"KoNViD MOS outside [1, 5] for flickr_id {flickr_id}")
                    decimal_places = len(attribute["MOS"].partition(".")[2])
                    rounding_tolerance = 0.5 * (10 ** -decimal_places) + 1e-9
                    if abs(attribute_mos - precise_mos) > rounding_tolerance:
                        raise ValueError(
                            f"KoNViD MOS tables disagree beyond attribute rounding for flickr_id {flickr_id}"
                        )
                    normalized.append({"file_name": filename, "mos": precise_mos})
                return normalized

            candidates = [
                name for name in names
                if Path(name).suffix.lower() in {".csv", ".json", ".jsonl", ".ndjson"}
            ]
            if not candidates:
                raise ValueError("metadata zip contains no CSV, JSON, or JSONL file")
            candidate = sorted(candidates)[0]
            extracted = Path(candidate)
            raw = archive.read(candidate)
        text = raw.decode("utf-8-sig")
        suffix = extracted.suffix.lower()
        if suffix in {".jsonl", ".ndjson"}:
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        if suffix == ".json":
            payload = json.loads(text)
            return payload if isinstance(payload, list) else payload.get("videos", payload.get("records", []))
        return list(csv.DictReader(text.splitlines()))

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8-sig")
    if suffix in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix == ".json":
        payload = json.loads(text)
        return payload if isinstance(payload, list) else payload.get("videos", payload.get("records", []))
    if suffix == ".csv":
        return list(csv.DictReader(text.splitlines()))
    raise ValueError("metadata must be .csv, .json, .jsonl, or .zip")


def read_metadata(path: Path) -> dict[str, dict[str, Any]]:
    """Read a CSV, JSON array/object, or JSONL metadata file."""
    result: dict[str, dict[str, Any]] = {}
    for raw in _read_metadata_rows(path):
        if not isinstance(raw, dict):
            raise ValueError("each metadata record must be an object")
        video_id = _video_id(_first_value(raw, ID_FIELDS))
        if video_id in result:
            raise ValueError(f"duplicate metadata video_id: {video_id}")
        result[video_id] = {
            "mos": _mos(_first_value(raw, MOS_FIELDS)),
            "parent_video_id": (
                _video_id(_first_value(raw, PARENT_FIELDS))
                if _first_value(raw, PARENT_FIELDS) is not None
                else None
            ),
        }
    return result


def _validate_ratios(train_ratio: float, val_ratio: float) -> None:
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must leave a non-empty test split")


def assign_splits(
    video_ids: Iterable[str],
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, str]:
    """Assign exact train/val/test counts using a deterministic shuffle."""
    _validate_ratios(train_ratio, val_ratio)
    ids = sorted(set(video_ids))
    shuffled = ids.copy()
    random.Random(seed).shuffle(shuffled)
    train_count = int(len(ids) * train_ratio)
    val_count = int(len(ids) * val_ratio)
    result: dict[str, str] = {}
    for index, video_id in enumerate(shuffled):
        result[video_id] = "train" if index < train_count else "val" if index < train_count + val_count else "test"
    return result


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    videos_dir: Path,
    metadata: dict[str, dict[str, Any]],
    source: str,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> list[dict[str, Any]]:
    files = sorted(path for path in videos_dir.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    if not files:
        raise ValueError(f"no supported video files found under {videos_dir}")

    records: list[dict[str, Any]] = []
    split_keys = {
        metadata.get(path.stem, {}).get("parent_video_id") or path.stem
        for path in files
    }
    split_cache = assign_splits(split_keys, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
    for path in files:
        video_id = path.stem
        row = metadata.get(video_id, {})
        parent_id = row.get("parent_video_id")
        split_key = parent_id or video_id
        split = split_cache[split_key]
        records.append(
            {
                "video_id": video_id,
                "path": str(path.resolve()),
                "mos": row.get("mos"),
                "source": source,
                "sha256": _sha256(path),
                "split": split,
                "parent_video_id": parent_id,
            }
        )
    return records


def write_jsonl(records: Iterable[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args(argv)

    metadata = read_metadata(args.metadata) if args.metadata else {}
    records = build_manifest(
        args.videos_dir,
        metadata,
        source=args.source,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    write_jsonl(records, args.output)
    print(f"wrote {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
