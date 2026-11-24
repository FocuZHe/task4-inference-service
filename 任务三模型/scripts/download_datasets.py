#!/usr/bin/env python3
"""Task 3（光电视频质量诊断）: fetch this branch's dataset manifests.

用于任务三（光电视频质量诊断；MOS 回归，不作为异常类别标签）.
Primary: konvid-1k (metadata by default; videos.zip only with --include-large).
External test only: live-vqc (form; never a training or model-selection set).
Supplements: cvd2014, lsvq, live-apv-livestream, youtube-ugc, live-qualcomm,
beyond8bits, live-sharechat, gamescope, live-wild-compressed.
Not on this branch: VisDrone, Anti-UAV410, UAVDT, DroneRF, RadioML, CRUW,
RadarScenes. Those are not Task 3 quality labels. The old threat-assessment
Task 3 list is obsolete.

Direct packages are downloaded and checked against manifest hashes.
KoNViD-1k videos.zip is default_download false. Beyond8Bits direct
packages are the four GitHub publish CSV/TXT files; HDR videos are not
fetched. YouTube-UGC raw and H264 video trees are not packages.
Form-gated LIVE sets, including live-sharechat, gamescope, and
live-wild-compressed, print their pages and are not errors.

The process exits non-zero only for hard failures: an unknown dataset id,
a broken manifest, or a direct download that does not match the manifest.
Synthetic blur, dark, overexpose, and freeze clips are produced in-project
by make_distortions.py. They are not a public archive in this directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_DIR = ROOT / "data" / "manifests"
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
CHUNK = 1024 * 1024
PROGRESS_EVERY = 500 * 1024 * 1024


def load_manifest(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if stripped[0] in "{[":
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            data = None
        else:
            if isinstance(data, dict):
                return [data]
            if isinstance(data, list):
                return data
    records = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno}: invalid JSONL: {exc}") from exc
    return records


def load_manifests(manifest_dir: Path) -> list[dict]:
    paths = sorted(manifest_dir.glob("*.jsonl"))
    if not paths:
        raise SystemExit(f"No manifests in {manifest_dir}")
    records = []
    for path in paths:
        records.extend(load_manifest(path))
    return records


def classify(record: dict) -> str:
    if record.get("direct_download") is True:
        return "direct"
    access = str(record.get("access") or "")
    if access in {"registration", "form"}:
        return "gated"
    return "manual"


def include_package(package: dict, include_large: bool) -> bool:
    if package.get("default_download") is False and not include_large:
        return False
    return True


def select_records(records: list[dict], dataset_ids: list[str] | None) -> list[dict]:
    if not dataset_ids:
        return records
    wanted = set(dataset_ids)
    found = {str(record.get("dataset_id")) for record in records}
    missing = sorted(wanted - found)
    if missing:
        known = ", ".join(sorted(found))
        raise SystemExit(
            "Unknown dataset id(s): "
            + ", ".join(missing)
            + f"\nKnown ids: {known}"
        )
    return [record for record in records if str(record.get("dataset_id")) in wanted]


def destination(raw_dir: Path, record: dict, package: dict) -> Path:
    filename = package.get("filename")
    if not filename or Path(filename).name != filename:
        raise RuntimeError(
            f"{record.get('dataset_id')}: package {package.get('package_id')} "
            "needs a plain filename"
        )
    return raw_dir / str(record["dataset_id"]) / filename


def file_digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def existing_is_complete(path: Path, package: dict) -> bool:
    if not path.is_file():
        return False
    expected_size = package.get("size_bytes")
    if isinstance(expected_size, int) and path.stat().st_size != expected_size:
        return False
    algorithm = package.get("hash_algorithm")
    digest = package.get("hash")
    if algorithm and digest:
        actual = file_digest(path, str(algorithm))
        return actual.lower() == str(digest).lower()
    if expected_size is None and not digest:
        return True
    return isinstance(expected_size, int)


def download_package(url: str, dest: Path, package: dict) -> None:
    allow_http = bool(package.get("insecure_http_ok"))
    if url.startswith("https://"):
        pass
    elif allow_http and url.startswith("http://"):
        pass
    else:
        raise RuntimeError(f"Refusing non-HTTPS URL: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    hasher = None
    algorithm = package.get("hash_algorithm")
    if algorithm and package.get("hash"):
        hasher = hashlib.new(str(algorithm))
    total = 0
    next_report = PROGRESS_EVERY
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                if hasher is not None:
                    hasher.update(chunk)
                total += len(chunk)
                if total >= next_report:
                    print(
                        f"  ... {dest.name}: {total / 1e9:.1f} GB",
                        flush=True,
                    )
                    next_report += PROGRESS_EVERY
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"download failed for {url}: {exc}") from exc

    expected_size = package.get("size_bytes")
    if isinstance(expected_size, int) and total != expected_size:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"{dest.name}: size {total} does not match manifest size {expected_size}"
        )
    if hasher is not None:
        actual = hasher.hexdigest()
        expected = str(package["hash"]).lower()
        if actual.lower() != expected:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"{dest.name}: {algorithm} {actual} does not match manifest {expected}"
            )
    partial.replace(dest)


def print_registration(record: dict, raw_dir: Path) -> None:
    dataset_id = record["dataset_id"]
    print(f"[registration] {dataset_id}: {record.get('name')}")
    print(f"  {record.get('fetch_notes') or record.get('access_notes')}")
    for url in record.get("official_urls") or []:
        print(f"  official: {url}")
    for package in record.get("packages") or []:
        print(f"  package {package.get('package_id')}: {package.get('description')}")
        if package.get("download_url"):
            print(f"    gated url: {package['download_url']}")
        filename = package.get("filename")
        if filename:
            print(f"    save as: {raw_dir / dataset_id / filename}")
        if package.get("hash_note"):
            print(f"    hash: null ({package['hash_note']})")
    print("  This entry is registration-gated. Open the form in a browser.")
    print("  A gated dataset is not a hard failure, so the process can still exit zero.")
    print()


def print_gated(record: dict, raw_dir: Path) -> None:
    dataset_id = record["dataset_id"]
    access = record.get("access") or "gated"
    print(f"[gated:{access}] {dataset_id}: {record.get('name')}")
    print(f"  {record.get('fetch_notes') or record.get('access_notes')}")
    for url in record.get("official_urls") or []:
        print(f"  official: {url}")
    single = record.get("official_url")
    if single and single not in (record.get("official_urls") or []):
        print(f"  official: {single}")
    for package in record.get("packages") or []:
        print(f"  package {package.get('package_id')}: {package.get('description')}")
        if package.get("download_url"):
            print(f"    gated url: {package['download_url']}")
        filename = package.get("filename")
        if filename:
            print(f"    save as: {raw_dir / dataset_id / filename}")
        if package.get("hash") is None:
            print(f"    hash: null ({package.get('hash_note') or 'not published'})")
    print("  Browser or form gate. Not a hard failure.")
    print()


def print_manual(record: dict, raw_dir: Path) -> None:
    dataset_id = record["dataset_id"]
    print(f"[manual] {dataset_id}: {record.get('name')} ({record.get('status') or record.get('access')})")
    print(f"  Save archives under {raw_dir / dataset_id}/ and do not commit them.")
    if record.get("official_url"):
        print(f"  official: {record['official_url']}")
    for url in record.get("official_urls") or []:
        if url != record.get("official_url"):
            print(f"  official: {url}")
    if record.get("fetch_notes"):
        print(f"  {record['fetch_notes']}")
    for package in record.get("packages") or []:
        role = package.get("role") or "data"
        print(f"  - {package.get('package_id')} [{role}]: {package.get('description')}")
        if package.get("download_url"):
            print(f"    {package['download_url']}")
        for mirror in package.get("mirror_urls") or []:
            print(f"    mirror: {mirror}")
        if package.get("hash") is None:
            print(f"    hash: null ({package.get('hash_note')})")
    print()


def print_direct_plan(record: dict, raw_dir: Path, include_large: bool) -> None:
    dataset_id = record["dataset_id"]
    print(f"[direct] {dataset_id}: {record.get('name')}")
    for package in record.get("packages") or []:
        if not include_package(package, include_large):
            print(
                f"  - skip {package.get('package_id')} unless --include-large "
                f"({package.get('size_note') or package.get('description')})"
            )
            if package.get("download_url"):
                print(f"    {package['download_url']}")
            continue
        dest = destination(raw_dir, record, package)
        if package.get("hash"):
            hash_text = f"{package.get('hash_algorithm') or 'hash'} {package['hash']}"
        else:
            hash_text = "hash null"
        size = package.get("size_bytes")
        size_text = f"{size} bytes" if isinstance(size, int) else "size unknown"
        print(f"  - {dest.relative_to(raw_dir.parent)} ({size_text}, {hash_text})")
        print(f"    {package.get('download_url')}")
    print()


def fetch_direct(record: dict, raw_dir: Path, force: bool, include_large: bool) -> list[str]:
    errors = []
    dataset_id = str(record.get("dataset_id"))
    packages = record.get("packages") or []
    if not packages:
        return [f"{dataset_id}: direct dataset has no packages"]
    selected = [package for package in packages if include_package(package, include_large)]
    if not selected:
        print(f"[direct] {dataset_id}: {record.get('name')}")
        print("  no package is selected. Pass --include-large to fetch optional archives.")
        print()
        return []
    print(f"[direct] {dataset_id}: {record.get('name')}")
    for package in packages:
        package_id = package.get("package_id")
        if not include_package(package, include_large):
            print(f"  skip {package_id}: default_download is false (pass --include-large)")
            continue
        url = package.get("download_url")
        try:
            if not url:
                raise RuntimeError("missing download_url")
            dest = destination(raw_dir, record, package)
            if dest.exists() and not force and existing_is_complete(dest, package):
                print(f"  skip {dest.name}: already present and matches the manifest")
                continue
            print(f"  get {package_id}: {url}")
            download_package(url, dest, package)
            print(f"  saved {dest}")
        except (RuntimeError, OSError) as exc:
            message = f"{dataset_id}/{package_id}: {exc}"
            print(f"  ERROR {message}", file=sys.stderr)
            errors.append(message)
    print()
    return errors


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Task 3（光电视频质量诊断）. Download direct dataset packages "
            "from data/manifests into data/raw/<dataset_id>/. KoNViD-1k "
            "metadata is included by default; videos.zip needs --include-large. "
            "Form-gated sets print their links and do not by themselves cause "
            "a non-zero exit."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help=(
            "Dataset id to process. Repeat for several ids. Default: every "
            "manifest on this branch."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without downloading. Gated entries still exit zero.",
    )
    parser.add_argument(
        "--include-large",
        action="store_true",
        help=(
            "Also fetch direct packages marked default_download false. "
            "Those archives stay under data/raw/ and must not be committed."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download a direct package even when the destination already matches.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
        help=f"Manifest directory (default: {DEFAULT_MANIFEST_DIR})",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Download directory (default: {DEFAULT_RAW_DIR})",
    )
    args = parser.parse_args(argv)

    records = select_records(load_manifests(args.manifest_dir), args.dataset)
    failures: list[str] = []
    counts = {"direct": 0, "manual": 0, "gated": 0}

    for record in records:
        kind = classify(record)
        counts[kind] += 1
        if kind == "gated":
            if str(record.get("access") or "") == "registration":
                print_registration(record, args.raw_dir)
            else:
                print_gated(record, args.raw_dir)
        elif kind == "manual":
            print_manual(record, args.raw_dir)
        elif args.dry_run:
            try:
                print_direct_plan(record, args.raw_dir, args.include_large)
            except RuntimeError as exc:
                failures.append(str(exc))
                print(f"ERROR {exc}", file=sys.stderr)
        else:
            failures.extend(
                fetch_direct(record, args.raw_dir, args.force, args.include_large)
            )

    print(
        "Summary: "
        f"{counts['direct']} direct, "
        f"{counts['manual']} manual, "
        f"{counts['gated']} gated (browser/form/registration; not a failure)."
    )
    if failures:
        print("Not fetched automatically:", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
