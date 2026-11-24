"""Import an authorized LIVE-VQC package as an auditable external-test manifest.

The importer deliberately does not download or guess a provider package layout.
The caller must supply the package, its metadata, and the SHA-256 recorded after
official access was granted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
ID_FIELDS = ("filename", "file_name", "file", "video", "video_filename", "videoFile")
MOS_FIELDS = ("mos", "MOS", "mean_opinion_score", "meanOpinionScore", "quality_score")
OFFICIAL_URL_PREFIXES = (
    "https://www.colorado.edu/lab/live/",
    "https://live.ece.utexas.edu/research/LIVEVQC/",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8-sig")
    if suffix in {".jsonl", ".ndjson"}:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif suffix == ".json":
        value = json.loads(text)
        payload = value if isinstance(value, list) else value.get("videos", value.get("records", []))
    elif suffix == ".csv":
        payload = list(csv.DictReader(text.splitlines()))
    else:
        raise ValueError("LIVE-VQC metadata must be CSV, JSON, JSONL, or NDJSON")
    if not isinstance(payload, list) or not payload or not all(isinstance(row, dict) for row in payload):
        raise ValueError("LIVE-VQC metadata must contain a non-empty list of records")
    return payload


def _first(row: dict[str, Any], fields: tuple[str, ...]) -> Any:
    for field in fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            return value
    return None


def _video_files(videos_dir: Path) -> dict[str, Path]:
    if not videos_dir.is_dir():
        raise ValueError("videos_dir must be an existing directory")
    files = [path for path in videos_dir.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS]
    by_name: dict[str, Path] = {}
    for path in files:
        name = path.name
        if name in by_name:
            raise ValueError(f"duplicate LIVE-VQC video filename: {name}")
        by_name[name] = path
    if not by_name:
        raise ValueError("no supported LIVE-VQC videos found")
    return by_name


def import_live_vqc(
    *,
    package_path: Path,
    metadata_path: Path,
    videos_dir: Path,
    official_url: str,
    license_note: str,
    expected_package_sha256: str,
    expected_video_count: int = 585,
) -> list[dict[str, Any]]:
    """Validate a locally obtained official package and return external-test rows."""
    if not package_path.is_file() or not metadata_path.is_file():
        raise ValueError("official package and metadata files must exist")
    if not any(official_url.startswith(prefix) for prefix in OFFICIAL_URL_PREFIXES):
        raise ValueError("official_url must be a recognized LIVE-VQC official page")
    if not license_note.strip():
        raise ValueError("license_note is required for an authorized external dataset")
    expected = expected_package_sha256.lower().strip()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("expected_package_sha256 must be a 64-character hexadecimal SHA-256")
    actual = sha256_file(package_path)
    if actual != expected:
        raise ValueError(f"LIVE-VQC package SHA-256 mismatch: expected {expected}, got {actual}")
    if expected_video_count < 1:
        raise ValueError("expected_video_count must be positive")
    videos = _video_files(videos_dir)
    rows = _metadata_rows(metadata_path)
    if len(rows) != expected_video_count:
        raise ValueError(f"LIVE-VQC metadata count {len(rows)} does not match expected {expected_video_count}")
    if len(videos) != expected_video_count:
        raise ValueError(f"LIVE-VQC video count {len(videos)} does not match expected {expected_video_count}")
    manifest: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_video_ids: set[str] = set()
    for index, raw in enumerate(rows, start=1):
        filename = str(_first(raw, ID_FIELDS) or "").strip()
        if not filename or Path(filename).name != filename:
            raise ValueError(f"metadata row {index} has an invalid filename")
        if filename in seen:
            raise ValueError(f"duplicate LIVE-VQC metadata filename: {filename}")
        seen.add(filename)
        if filename not in videos:
            raise ValueError(f"metadata filename is not present in videos_dir: {filename}")
        video_id = videos[filename].stem
        if video_id in seen_video_ids:
            raise ValueError(f"duplicate LIVE-VQC videoId derived from filename stem: {video_id}")
        seen_video_ids.add(video_id)
        raw_mos = _first(raw, MOS_FIELDS)
        try:
            mos = float(raw_mos)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"metadata row {index} has invalid MOS") from exc
        if not math.isfinite(mos):
            raise ValueError(f"metadata row {index} has non-finite MOS")
        video = videos[filename]
        manifest.append({
            "dataset": "LIVE-VQC",
            "source": "LIVE-VQC",
            "split": "external_test",
            "role": "external_test",
            "recordId": f"live-vqc-{index:04d}",
            "videoId": video_id,
            "filename": filename,
            "path": str(video),
            "videoSha256": sha256_file(video),
            "mos": mos,
            "mosField": next(field for field in MOS_FIELDS if raw.get(field) is not None and str(raw[field]).strip()),
            "packageSha256": actual,
            "officialUrl": official_url,
            "license": license_note,
            "externalTestOnly": True,
            "trainingAllowed": False,
            "selectionAllowed": False,
        })
    if seen != set(videos):
        missing = sorted(set(videos) - seen)
        raise ValueError(f"videos are missing metadata rows: {missing[:5]}")
    return manifest


def build_external_provenance(
    manifest: list[dict[str, Any]],
    *,
    package_path: Path,
    metadata_path: Path,
    official_url: str,
    license_note: str,
) -> dict[str, Any]:
    """Build the evaluator provenance from the already validated manifest."""
    if not manifest:
        raise ValueError("cannot build LIVE-VQC provenance from an empty manifest")
    if not package_path.is_file() or not metadata_path.is_file():
        raise ValueError("package and metadata files must exist")
    package_sha = sha256_file(package_path)
    package_hashes = {str(row.get("packageSha256") or "").lower() for row in manifest}
    official_urls = {str(row.get("officialUrl") or "") for row in manifest}
    licenses = {str(row.get("license") or "").strip() for row in manifest}
    if package_hashes != {package_sha}:
        raise ValueError("manifest packageSha256 does not match the package file")
    if official_urls != {official_url} or licenses != {license_note.strip()}:
        raise ValueError("manifest external provenance fields are inconsistent")
    if any(
        row.get("dataset") != "LIVE-VQC"
        or row.get("split") != "external_test"
        or row.get("externalTestOnly") is not True
        or row.get("trainingAllowed") is not False
        or row.get("selectionAllowed") is not False
        for row in manifest
    ):
        raise ValueError("manifest contains rows that are not strict external_test records")
    return {
        "schemaVersion": "task3.live-vqc-provenance/1.0",
        "dataset": "LIVE-VQC",
        "annotationSource": "official LIVE-VQC metadata package",
        "packageSha256": package_sha,
        "packagePath": str(package_path),
        "metadataPath": str(metadata_path),
        "officialUrl": official_url,
        "license": license_note.strip(),
        "split": {"counts": {"external_test": len(manifest)}},
        "videoCount": len(manifest),
        "externalTestOnly": True,
        "trainingAllowed": False,
        "selectionAllowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--official-url", required=True)
    parser.add_argument("--license-note", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-video-count", type=int, default=585)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, help="optional provenance JSON written alongside the manifest")
    args = parser.parse_args(argv)
    rows = import_live_vqc(
        package_path=args.package,
        metadata_path=args.metadata,
        videos_dir=args.videos_dir,
        official_url=args.official_url,
        license_note=args.license_note,
        expected_package_sha256=args.expected_sha256,
        expected_video_count=args.expected_video_count,
    )
    if args.output.exists() or (args.provenance_output is not None and args.provenance_output.exists()):
        parser.error("refusing to overwrite an existing LIVE-VQC output")
    provenance = None
    if args.provenance_output is not None:
        provenance = build_external_provenance(
            rows,
            package_path=args.package,
            metadata_path=args.metadata,
            official_url=args.official_url,
            license_note=args.license_note,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    if provenance is not None:
        args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
        args.provenance_output.write_text(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} LIVE-VQC external_test records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
