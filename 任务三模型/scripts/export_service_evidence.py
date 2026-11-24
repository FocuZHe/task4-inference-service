"""Export source-bound service evidence; never run model inference."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from shared_contracts.quality import VideoQualityResult
from services.task3_quality_service.registry import load_registry


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render_html(summary: dict) -> bytes:
    title = html.escape(f'{summary["videoId"]} | {summary["modelVersion"]}')
    extent = max((frame["endTimeS"] for frame in summary["frames"]), default=1.0)
    rows, figures = [], []
    for frame in summary["frames"]:
        number = frame["eventIndex"]
        start, end = frame["startTimeS"], frame["endTimeS"]
        label = html.escape(frame["type"])
        status = html.escape(frame["status"])
        rows.append(
            f'<tr><td><a href="#event-{number}">{label}</a></td><td>{status}</td>'
            f'<td>{start:.6f}–{end:.6f}s</td><td>{frame["frameTimeS"]:.6f}s</td>'
            f'<td><div class="track"><span style="margin-left:{100 * start / extent:.6f}%;'
            f'width:{100 * (end - start) / extent:.6f}%"></span></div></td></tr>')
        figures.append(
            f'<figure id="event-{number}"><img src="{frame["file"]}" alt="{label}">'
            f'<figcaption>{label} / {status}：[{start:.6f}, {end:.6f})s；'
            f'证据请求 {frame["evidenceTimeS"]:.6f}s，实际帧 {frame["frameIndex"]} / '
            f'{frame["frameTimeS"]:.6f}s</figcaption></figure>')
    notes = html.escape("; ".join(summary["limitations"]))
    source = html.escape(summary["sourceVideoRef"])
    synthetic = "是（工程/合成输入）" if summary["isSynthetic"] else "否（按注册表记录，非真实性认证）"
    document = (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>视频异常复核证据</title><style>'
        'body{font-family:system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 16px;'
        'color:#17212b;line-height:1.6}h1{overflow-wrap:anywhere;font-size:1.4rem}'
        'img{max-width:100%;image-rendering:auto}table{border-collapse:collapse;width:100%}'
        'th,td{padding:10px;text-align:left;border-bottom:1px solid #ccd5df}'
        '.track{background:#e4eaf0;width:220px;height:12px}.track span{display:block;'
        'height:12px;background:#416ca5}code{overflow-wrap:anywhere}figure{margin:32px 0}'
        '</style></head><body>'
        f'<h1>{title}</h1><p>源视频：{source}<br>合成标记：{synthetic}<br>'
        f'视频 SHA-256：<code>{summary["sourceVideoSha256"]}</code></p>'
        f'<p>本次事件数：{summary["anomalyCount"]}；导出截图数：{summary["exportedFrameCount"]}。'
        '自动可用门槛尚未验证。时间轴为文件帧序/FPS，不是现场PTS；截图仅供视觉复核。</p>'
        '<table><thead><tr><th>类型</th><th>状态</th><th>区间</th><th>截图时刻</th>'
        f'<th>时间轴 0–{extent:.6f}s</th></tr></thead><tbody>'
        + "\n".join(rows) + '</tbody></table>'
        + ("\n".join(figures) if figures else '<p>没有异常事件，证据覆盖率不适用。</p>')
        + f'<p>{notes}</p><p>完整来源、原结果限制和哈希见 evidence.json。</p></body></html>'
    )
    return document.encode("utf-8")


def export_evidence(result_path: Path, registry_path: Path, data_root: Path, output: Path) -> dict:
    result_path, registry_path, output = Path(result_path), Path(registry_path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    result_hash, registry_hash = _sha(result_path), _sha(registry_path)
    result = VideoQualityResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    registry = load_registry(registry_path, Path(data_root))
    video = registry.get(result.video_id)
    if video is None or result.score_status != "scored":
        raise ValueError("registered scored result required")
    if (video.channel_type, video.is_synthetic) != (result.channel_type, result.is_synthetic):
        raise ValueError("result and registry source metadata differ")
    source_hash = _sha(video.path)
    source_ref = f"registry-video:{video.video_id}"
    if source_hash != video.sha256 or not any(
        ref.uri == source_ref and ref.sha256 == source_hash
        for ref in result.provenance.evidence_refs
    ):
        raise ValueError("result, registry and source video hashes must agree")
    for event in result.anomalies:
        if (event.evidence_time_s is None
                or not all(math.isfinite(value) for value in
                           (event.start_time_s, event.end_time_s, event.evidence_time_s))
                or not event.start_time_s <= event.evidence_time_s < event.end_time_s):
            raise ValueError("every event requires a finite in-interval evidence time")

    encoded, targets = {}, {}
    fps = None
    if result.anomalies:
        capture = cv2.VideoCapture(str(video.path))
        try:
            if not capture.isOpened():
                raise ValueError("source video cannot be decoded")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("source video has no usable frame timing")
            for number, event in enumerate(result.anomalies):
                position = event.evidence_time_s * fps
                if not math.isfinite(position):
                    raise ValueError("evidence frame position is not finite")
                index = round(position)
                if index < 0 or not event.start_time_s <= index / fps < event.end_time_s:
                    raise ValueError("evidence frame falls outside event interval")
                targets.setdefault(index, []).append(number)
            index = 0
            while index <= max(targets):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError("source ended before the requested evidence frame")
                if index in targets:
                    ok, buffer = cv2.imencode(".png", frame)
                    if not ok:
                        raise ValueError("evidence PNG could not be encoded")
                    encoded[index] = buffer.tobytes()
                index += 1
        finally:
            capture.release()
    if (_sha(video.path) != source_hash or _sha(result_path) != result_hash
            or _sha(registry_path) != registry_hash):
        raise ValueError("evidence inputs changed during export")

    frames, payloads = [], {}
    for index, numbers in sorted(targets.items()):
        for number in numbers:
            event = result.anomalies[number]
            name = f"frames/{number:04d}.png"
            content = encoded[index]
            payloads[name] = content
            frames.append({"eventIndex": number, "type": event.type, "status": event.status,
                "startTimeS": event.start_time_s, "endTimeS": event.end_time_s, "file": name,
                "evidenceTimeS": event.evidence_time_s, "frameIndex": index, "frameTimeS": index / fps,
                "sha256": hashlib.sha256(content).hexdigest()})
    count = len(result.anomalies)
    summary = {"schemaVersion": "task3.service-evidence-export/1.0",
        "status": "exported" if count else "no_events", "videoId": result.video_id,
        "jobId": result.job_id, "modelVersion": result.provenance.model_version,
        "channelType": result.channel_type, "isSynthetic": result.is_synthetic,
        "provenance": result.provenance.model_dump(mode="json", by_alias=True, exclude_none=True),
        "sourceVideoRef": source_ref, "sourceVideoSha256": source_hash,
        "inputResultSha256": result_hash, "inputRegistrySha256": registry_hash,
        "timeline": "decoded_frame_index_divided_by_fps", "sourceFps": fps,
        "anomalyCount": count, "exportedFrameCount": len(frames),
        "evidenceCoverageRate": len(frames) / count if count else None,
        "frames": frames, "automaticGateValidated": False,
        "limitations": [*result.limitations, "frame-index/FPS timeline, not live PTS",
            "screenshots provide context, not independent proof of class correctness",
            "coverage applies only to these exported events; no real-data accuracy claim"]}
    payloads["index.html"] = _render_html(summary)
    summary["htmlSha256"] = hashlib.sha256(payloads["index.html"]).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    for name, content in payloads.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(content)
        if _sha(target) != hashlib.sha256(content).hexdigest():
            raise OSError("exported evidence failed read-back integrity check")
    # Publish the success index last. Partial output directories must not be reused.
    with (output / "evidence.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = export_evidence(args.result, args.registry, args.data_root, args.output_dir)
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
