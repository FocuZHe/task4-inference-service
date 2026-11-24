"""Generate deterministic, annotated Task 3 distortions from official KoNViD videos."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any


GENERATOR_VERSION = "task3-distortions-v3-frame-count-aligned"
VIDEO_ENCODER = "libx264"
BLUR_SIGMAS = (0.8, 1.5, 2.5)
DARK_BRIGHTNESS = (-0.2, -0.4, -0.6)
OVEREXPOSED_BRIGHTNESS = (0.15, 0.3, 0.45)
FREEZE_DURATIONS = (1.5, 3.0, 5.0)
SPATIAL_EVENT_DURATION_S = 3.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_interval(
    video_id: str,
    duration_s: float,
    event_duration_s: float,
    seed: int,
    fps: float = 25.0,
    frame_count: int | None = None,
) -> dict[str, float]:
    if not math.isfinite(duration_s) or not math.isfinite(event_duration_s) or event_duration_s <= 0:
        raise ValueError("video and event duration must be finite and event duration must be positive")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if event_duration_s > duration_s:
        raise ValueError("event duration does not fit source video duration")
    if frame_count is not None and frame_count <= 0:
        raise ValueError("frame_count must be positive")
    total_frames = frame_count if frame_count is not None else max(1, round(duration_s * fps))
    event_frames = max(1, round(event_duration_s * fps))
    if event_frames > total_frames:
        raise ValueError("event duration does not fit source video duration")
    latest_start_frame = total_frames - event_frames
    rng = random.Random(f"{seed}:{video_id}:{event_duration_s:.6f}")
    start_frame = rng.randint(0, latest_start_frame) if latest_start_frame else 0
    end_frame = start_frame + event_frames
    start = round(start_frame / fps, 6)
    end = round(min(duration_s, end_frame / fps), 6)
    return {"startTimeS": start, "endTimeS": end}


def build_variant_specs(
    video_id: str,
    *,
    split: str,
    duration_s: float,
    seed: int,
    fps: float = 25.0,
    frame_count: int | None = None,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for anomaly_type, strengths in (
        ("blur", BLUR_SIGMAS),
        ("dark", DARK_BRIGHTNESS),
        ("overexposed", OVEREXPOSED_BRIGHTNESS),
    ):
        for level, value in enumerate(strengths, 1):
            interval = event_interval(
                f"{video_id}:{anomaly_type}:{level}", duration_s, SPATIAL_EVENT_DURATION_S, seed, fps=fps, frame_count=frame_count
            )
            specs.append(
                {
                    "type": anomaly_type,
                    "strengthLevel": level,
                    "durationS": interval["endTimeS"] - interval["startTimeS"],
                    "event": {"type": anomaly_type, **interval},
                    "parameters": {"sigma": value} if anomaly_type == "blur" else {"brightness": value},
                    "split": split,
                    "mos": None,
                }
            )
    for level, freeze_duration in enumerate(FREEZE_DURATIONS, 1):
        interval = event_interval(
            f"{video_id}:freeze_suspected:{level}", duration_s, freeze_duration, seed, fps=fps, frame_count=frame_count
        )
        specs.append(
            {
                "type": "freeze_suspected",
                "strengthLevel": level,
                "durationS": freeze_duration,
                "event": {"type": "freeze_suspected", **interval},
                "parameters": {"freezeDurationS": freeze_duration},
                "split": split,
                "mos": None,
            }
        )
    return specs


def _run(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return completed.stdout.strip()


def probe_video(path: Path, ffprobe: str) -> dict[str, float | int]:
    payload = json.loads(
        _run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,r_frame_rate,nb_frames,width,height:format=duration",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError(f"video has no readable video stream: {path}")
    stream = streams[0]
    fps_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    numerator, denominator = (int(value) for value in fps_text.split("/", 1))
    fps = numerator / denominator if denominator else 0.0
    duration = float((payload.get("format") or {}).get("duration") or 0.0)
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"video has invalid frame rate or duration: {path}")
    frames_text = stream.get("nb_frames")
    frames = int(frames_text) if frames_text and frames_text != "N/A" else round(fps * duration)
    return {"fps": fps, "durationS": duration, "frameCount": frames}


def _filter_for(spec: dict[str, Any], info: dict[str, float | int]) -> str:
    start = float(spec["event"]["startTimeS"])
    end = float(spec["event"]["endTimeS"])
    fps = float(info["fps"])
    start_frame = round(start * fps)
    end_frame = round(end * fps)
    start = start_frame / fps
    end = end_frame / fps
    if spec["type"] == "blur":
        return f"gblur=sigma={spec['parameters']['sigma']}:enable='gte(t,{start:.9f})*lt(t,{end:.9f})'"
    if spec["type"] == "dark":
        return f"eq=brightness={spec['parameters']['brightness']}:enable='gte(t,{start:.9f})*lt(t,{end:.9f})'"
    if spec["type"] == "overexposed":
        return f"eq=brightness={spec['parameters']['brightness']}:enable='gte(t,{start:.9f})*lt(t,{end:.9f})'"
    first = max(0, start_frame)
    last = max(first, end_frame - 1)
    return f"freezeframes=first={first}:last={last}:replace={first}"


def generate_variant(
    source: Path,
    destination: Path,
    spec: dict[str, Any],
    info: dict[str, float | int],
    *,
    ffmpeg: str,
) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite generated video: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle, temporary_name = tempfile.mkstemp(prefix=".tmp-", suffix=".partial.mp4", dir=destination.parent)
    os.close(temporary_handle)
    temporary = Path(temporary_name)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-an",
        "-sn",
        "-dn",
    ]
    if spec["type"] == "freeze_suspected":
        freeze_filter = _filter_for(spec, info)
        command.extend(
            [
                "-i",
                str(source),
                "-filter_complex",
                f"[0:v][1:v]{freeze_filter}[outv]",
                "-map",
                "[outv]",
            ]
        )
    else:
        command.extend(["-vf", _filter_for(spec, info), "-map", "0:v:0"])
    command.extend([
        "-c:v",
        VIDEO_ENCODER,
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(temporary),
    ])
    try:
        _run(command)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"encoder produced no output for {source}")
        os.link(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_manifest(path: Path, split: str | None, limit: int | None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(row.get("source") != "konvid-1k" for row in rows):
        raise ValueError("only the verified official konvid-1k manifest is supported")
    if any(row.get("split") not in {"train", "val", "test"} for row in rows):
        raise ValueError("manifest rows must have train, val, or test split")
    selected = [row for row in rows if split is None or row["split"] == split]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    if not selected:
        raise ValueError("manifest selection is empty")
    return selected


def _read_resume_manifest(
    path: Path, expected_source_ids: list[str]
) -> tuple[list[dict[str, Any]], tuple[bytes, int] | None]:
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    last_nonempty = max((index for index, line in enumerate(lines) if line.strip()), default=-1)
    entries: list[tuple[dict[str, Any], int]] = []
    malformed_start: int | None = None
    offset = 0
    for index, line in enumerate(lines):
        start = offset
        offset += len(line)
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            if index != last_nonempty:
                raise ValueError("resume manifest has corrupt middle data") from exc
            malformed_start = start
            break
        if not isinstance(record, dict):
            raise ValueError("resume manifest records must be objects")
        entries.append((record, start))

    groups: list[tuple[str, int, int]] = []
    for index, (record, _) in enumerate(entries):
        source_id = str(record.get("sourceVideoId"))
        if groups and groups[-1][0] == source_id:
            group_id, group_start, _ = groups[-1]
            groups[-1] = (group_id, group_start, index + 1)
        else:
            groups.append((source_id, index, index + 1))
    group_ids = [group[0] for group in groups]
    if group_ids != expected_source_ids[: len(group_ids)]:
        raise ValueError("resume manifest must contain a contiguous prefix of the selected official manifest")
    if any(end - start != 13 for _, start, end in groups[:-1]):
        raise ValueError("resume manifest has an incomplete or corrupt middle source group")
    last_count = groups[-1][2] - groups[-1][1] if groups else 0
    if last_count > 13:
        raise ValueError("resume manifest has too many records in a source group")

    partial_group = bool(groups and last_count < 13)
    if malformed_start is None and not partial_group:
        return [record for record, _ in entries], None
    cutoff_index = groups[-1][1] if partial_group else len(entries)
    cutoff = entries[cutoff_index][1] if cutoff_index < len(entries) else malformed_start
    return [record for record, _ in entries[:cutoff_index]], (raw, len(raw) if cutoff is None else cutoff)


def _preserve_and_restore_manifest(path: Path, raw: bytes, cutoff: int) -> None:
    while True:
        backup = path.with_name(f"{path.stem}.recovery-{uuid.uuid4().hex}{path.suffix}")
        try:
            with backup.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            break
        except FileExistsError:
            continue
    temporary_handle, temporary_name = tempfile.mkstemp(prefix=".manifest-", suffix=".partial", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(temporary_handle, "wb") as handle:
            handle.write(raw[:cutoff])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()




def build_records(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    ffmpeg: str,
    ffprobe: str,
    seed: int,
    official_package_sha256: str,
    manifest_path: Path,
    workers: int = 4,
    resume: bool = False,
) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    version_output = _run([ffmpeg, "-version"]).splitlines()[0]
    records_by_source: dict[str, list[dict[str, Any]]] = {}
    manifest_repair: tuple[bytes, int] | None = None
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"manifest already exists; pass --resume to continue: {manifest_path}")
        manifest_records, manifest_repair = _read_resume_manifest(
            manifest_path, [str(row["video_id"]) for row in rows]
        )
        for record in manifest_records:
            records_by_source.setdefault(str(record.get("sourceVideoId")), []).append(record)

    source_by_id = {str(row["video_id"]): row for row in rows}
    existing_order = [str(row["video_id"]) for row in rows if str(row["video_id"]) in records_by_source]
    if existing_order != [str(row["video_id"]) for row in rows[: len(existing_order)]]:
        raise ValueError("resume manifest must contain a contiguous prefix of the selected official manifest")
    for source_id, records in records_by_source.items():
        source_row = source_by_id.get(source_id)
        if source_row is None or len(records) != 13:
            raise ValueError(f"resume manifest has unexpected/incomplete source group: {source_id}")
        if any(
            record.get("sourceVideoSha256") != source_row["sha256"]
            or record.get("split") != source_row["split"]
            or record.get("officialMetadataPackageSha256") != official_package_sha256
            or record.get("seed") != seed
            or record.get("generatorVersion") != GENERATOR_VERSION
            for record in records
        ):
            raise ValueError(f"resume manifest configuration/source mismatch for {source_id}")
        if {record.get("recordId") for record in records} != {
            f"{source_id}__original",
            *{
                f"{source_id}__{anomaly_type}__s{level}"
                for anomaly_type in ("blur", "dark", "overexposed", "freeze_suspected")
                for level in range(1, 4)
            },
        }:
            raise ValueError(f"resume manifest has incomplete or duplicate variants for {source_id}")
        for record in records:
            path = Path(record["path"])
            if not path.is_file() or sha256_file(path) != record.get("videoSha256"):
                raise ValueError(f"resume output missing or hash-mismatched: {record['recordId']}")
    if manifest_repair is not None:
        _preserve_and_restore_manifest(manifest_path, *manifest_repair)
    if not resume and manifest_path.exists():
        raise FileExistsError(f"manifest already exists: {manifest_path}")

    def build_source_records(row: dict[str, Any]) -> list[dict[str, Any]]:
        records = []
        source = Path(row["path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        source_hash = sha256_file(source)
        if source_hash != row["sha256"]:
            raise ValueError(f"official video SHA-256 mismatch: {row['video_id']}")
        info = probe_video(source, ffprobe)
        common = {
            "dataset": "KoNViD-1k-controlled-distortions",
            "source": "konvid-1k",
            "sourceVideoId": row["video_id"],
            "sourceVideoSha256": source_hash,
            "officialMetadataPackageSha256": official_package_sha256,
            "parentVideoId": row["video_id"],
            "split": row["split"],
            "isSynthetic": False,
            "anomaly": None,
            "mos": None,
            "videoId": row["video_id"],
            "path": str(source.resolve()),
            "videoSha256": source_hash,
            "generatorVersion": GENERATOR_VERSION,
            "encoder": VIDEO_ENCODER,
            "encoderVersion": version_output,
            "seed": seed,
        }
        records.append({**common, "recordId": f"{row['video_id']}__original"})
        for spec in build_variant_specs(
            row["video_id"],
            split=row["split"],
            duration_s=float(info["durationS"]),
            seed=seed,
            fps=float(info["fps"]),
            frame_count=int(info["frameCount"]),
        ):
            variant_id = f"{row['video_id']}__{spec['type']}__s{spec['strengthLevel']}"
            destination = output_dir / "videos" / row["split"] / row["video_id"] / f"{variant_id}.mp4"
            destination.parent.mkdir(parents=True, exist_ok=True)
            output_path = destination
            if destination.exists() and resume:
                with tempfile.TemporaryDirectory(prefix=".verify-", dir=destination.parent) as temporary_dir:
                    verification_path = Path(temporary_dir) / "output.mp4"
                    generate_variant(source, verification_path, spec, info, ffmpeg=ffmpeg)
                    if sha256_file(verification_path) == sha256_file(destination):
                        # The existing orphan is byte-identical to a fresh render; leave it in place.
                        pass
                    else:
                        while True:
                            recovery_path = destination.with_name(f".recovery-{uuid.uuid4().hex}.mp4")
                            try:
                                os.link(verification_path, recovery_path)
                            except FileExistsError:
                                continue
                            output_path = recovery_path
                            break
            else:
                generate_variant(source, destination, spec, info, ffmpeg=ffmpeg)
            records.append(
                {
                    **common,
                    "recordId": variant_id,
                    "videoId": variant_id,
                    "path": str(output_path.resolve()),
                    "videoSha256": sha256_file(output_path),
                    "isSynthetic": True,
                    "anomaly": spec["event"],
                    "strengthLevel": spec["strengthLevel"],
                    "parameters": spec["parameters"],
                    "sourceDurationS": info["durationS"],
                    "sourceFps": info["fps"],
                    "sourceFrameCount": info["frameCount"],
                }
            )
        return records

    pending = []
    for row in rows:
        source_id = str(row["video_id"])
        if source_id in records_by_source:
            continue
        existing_dir = output_dir / "videos" / row["split"] / source_id
        if not resume and existing_dir.exists() and any(existing_dir.iterdir()):
            raise FileExistsError(
                f"uncheckpointed output exists for {source_id}; preserve it and choose a fresh output directory"
            )
        pending.append(row)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        manifest_path.touch()
    completed_groups: dict[str, list[dict[str, Any]]] = {}
    next_index = len(existing_order)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending_rows = iter(pending)
        futures: dict[Any, dict[str, Any]] = {}

        def submit_next() -> bool:
            try:
                row = next(pending_rows)
            except StopIteration:
                return False
            futures[executor.submit(build_source_records, row)] = row
            return True

        for _ in range(min(workers, len(pending))):
            submit_next()
        completed = 0
        with manifest_path.open("a", encoding="utf-8", newline="\n") as checkpoint:
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    row = futures.pop(future)
                    source_id = str(row["video_id"])
                    try:
                        group = future.result()
                    except BaseException:
                        for pending_future in futures:
                            pending_future.cancel()
                        raise
                    records_by_source[source_id] = group
                    completed_groups[source_id] = group
                    completed += 1
                    while next_index < len(rows):
                        next_id = str(rows[next_index]["video_id"])
                        next_group = completed_groups.pop(next_id, None)
                        if next_group is None:
                            break
                        checkpoint.write(
                            "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in next_group)
                        )
                        checkpoint.flush()
                        os.fsync(checkpoint.fileno())
                        next_index += 1
                    print(f"checkpointed source {completed}/{len(pending)}: {row['video_id']}", flush=True)
                    submit_next()

    return [
        record
        for source_row in rows
        for record in records_by_source.get(str(source_row["video_id"]), [])
    ]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(temporary_handle, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/konvid-1k/manifest-official.jsonl"))
    parser.add_argument("--provenance", type=Path, default=Path("data/raw/konvid-1k/annotation-provenance-official.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/konvid-1k/synthetic"))
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int, help="process the first N source videos (for smoke tests)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4, help="parallel source-video encoders")
    parser.add_argument("--resume", action="store_true", help="verify and continue from completed source checkpoints")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe"))
    args = parser.parse_args(argv)
    if not args.ffmpeg or not args.ffprobe:
        parser.error("ffmpeg and ffprobe must be installed or passed explicitly")
    if not args.manifest.is_file():
        parser.error(f"manifest does not exist: {args.manifest}")
    if not args.provenance.is_file():
        parser.error(f"official provenance record does not exist: {args.provenance}")
    if not args.output_dir.resolve().is_relative_to(Path("data/raw/konvid-1k/synthetic").resolve()):
        parser.error("output-dir must be inside data/raw/konvid-1k/synthetic")

    rows = _read_manifest(args.manifest, args.split, args.limit)
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    package_sha256 = provenance.get("packageSha256")
    if (
        provenance.get("dataset") != "KoNViD-1k"
        or provenance.get("annotationSource") != "official KoNViD-1k metadata package"
        or not isinstance(package_sha256, str)
        or len(package_sha256) != 64
        or provenance.get("videoMatch", {}).get("unmatched") != 0
    ):
        parser.error("provenance must verify the official KoNViD package and exact filename mapping")
    manifest_path = args.output_dir / "manifest.jsonl"
    records = build_records(
        rows,
        output_dir=args.output_dir,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        seed=args.seed,
        official_package_sha256=package_sha256,
        manifest_path=manifest_path,
        workers=args.workers,
        resume=args.resume,
    )
    print(f"wrote {len(records)} records ({len(rows)} source videos) to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
