"""从冻结的任务三 test 产物生成 CSV 指标表和静态时间轴证据页。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from shared_contracts.quality import VideoQualityResult


EVENT_CLASSES = ("blur", "dark", "overexposed", "freeze_suspected")
CSV_COLUMNS = (
    "metric_type",
    "metric",
    "precision",
    "recall",
    "f1",
    "value",
    "target",
    "passed",
    "sample_count",
    "model_version",
    "event_model_version",
    "test_report_sha256",
    "test_results_sha256",
    "manifest_sha256",
)


class EvidenceExportError(ValueError):
    """输入不是可追溯的一次性 test 证据，或输出目标不安全。"""


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceExportError(f"{description} could not be read as JSON") from exc
    if not isinstance(value, dict):
        raise EvidenceExportError(f"{description} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise EvidenceExportError("an input file could not be hashed") from exc
    return digest.hexdigest()


def _finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceExportError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise EvidenceExportError(f"{label} must be finite")
    return number


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceExportError(f"{label} must be a non-negative integer")
    return value


def _load_test_manifest(path: Path) -> dict[str, dict[str, Any]]:
    test_rows: list[dict[str, Any]] = []
    durations_by_source: dict[str, float] = {}
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise EvidenceExportError("test manifest could not be opened") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvidenceExportError(f"test manifest has invalid JSON at line {line_number}") from exc
            if not isinstance(row, dict):
                raise EvidenceExportError(f"test manifest record at line {line_number} must be an object")
            if row.get("split") != "test":
                continue
            test_rows.append(row)
            raw_duration = row.get("sourceDurationS")
            if raw_duration is not None:
                source_id = row.get("sourceVideoId")
                if not isinstance(source_id, str) or not source_id:
                    raise EvidenceExportError(f"test manifest record at line {line_number} has no sourceVideoId")
                duration = _finite_number(raw_duration, "sourceDurationS")
                if duration <= 0:
                    raise EvidenceExportError("sourceDurationS must be positive")
                previous = durations_by_source.get(source_id)
                if previous is not None and not math.isclose(previous, duration, rel_tol=0.0, abs_tol=1e-6):
                    raise EvidenceExportError("test variants of one source have inconsistent durations")
                durations_by_source[source_id] = duration

    videos: dict[str, dict[str, Any]] = {}
    for row in test_rows:
        video_id = row.get("videoId")
        if not isinstance(video_id, str) or not video_id:
            raise EvidenceExportError("test manifest record has no videoId")
        if video_id in videos:
            raise EvidenceExportError("test manifest contains duplicate videoId values")
        duration = row.get("sourceDurationS")
        if duration is None:
            duration = durations_by_source.get(str(row.get("sourceVideoId", "")))
        duration = _finite_number(duration, "sourceDurationS")
        if duration <= 0:
            raise EvidenceExportError("sourceDurationS must be positive")
        row = {**row, "sourceDurationS": duration}
        event = row.get("anomaly")
        if event is not None:
            if not isinstance(event, dict) or event.get("type") not in EVENT_CLASSES:
                raise EvidenceExportError("test manifest contains an unsupported ground-truth event")
            start = _finite_number(event.get("startTimeS"), "ground-truth startTimeS")
            end = _finite_number(event.get("endTimeS"), "ground-truth endTimeS")
            if start < 0 or end < start or end > duration + 1e-6:
                raise EvidenceExportError("ground-truth event interval is invalid")
        videos[video_id] = row
    if not videos:
        raise EvidenceExportError("test manifest contains no videos in the test split")
    return videos


def _load_results(path: Path) -> dict[str, VideoQualityResult]:
    results: dict[str, VideoQualityResult] = {}
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise EvidenceExportError("test results could not be opened") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                result = VideoQualityResult.model_validate_json(line)
            except Exception as exc:
                raise EvidenceExportError(f"test result at line {line_number} fails the shared contract") from exc
            if result.video_id in results:
                raise EvidenceExportError("test results contain duplicate videoId values")
            results[result.video_id] = result
    if not results:
        raise EvidenceExportError("test results contain no records")
    return results


def _metric_rows(
    report: dict[str, Any], input_hashes: dict[str, str], result_model_version: str
) -> list[dict[str, str]]:
    model_version = report.get("modelVersion")
    per_class = report.get("perClassEventMetrics")
    targets = report.get("acceptanceTargets")
    if not isinstance(model_version, str) or not model_version:
        raise EvidenceExportError("test report is missing modelVersion")
    if not isinstance(per_class, dict) or set(per_class) != set(EVENT_CLASSES):
        raise EvidenceExportError("test report does not contain all supported event metrics")
    if not isinstance(targets, dict):
        raise EvidenceExportError("test report is missing acceptanceTargets")

    shared = {
        "model_version": result_model_version,
        "event_model_version": model_version,
        "test_report_sha256": input_hashes["test-report"],
        "test_results_sha256": input_hashes["test-results"],
        "manifest_sha256": input_hashes["manifest"],
    }
    rows: list[dict[str, str]] = []
    for event_class in EVENT_CLASSES:
        metrics = per_class[event_class]
        if not isinstance(metrics, dict):
            raise EvidenceExportError(f"test report metrics for {event_class} are invalid")
        tp = _nonnegative_integer(metrics.get("tp"), f"{event_class} tp")
        _nonnegative_integer(metrics.get("fp"), f"{event_class} fp")
        fn = _nonnegative_integer(metrics.get("fn"), f"{event_class} fn")
        rows.append(
            {
                **shared,
                "metric_type": "event_class",
                "metric": event_class,
                "precision": _number_text(metrics.get("precision"), f"{event_class} precision"),
                "recall": _number_text(metrics.get("recall"), f"{event_class} recall"),
                "f1": _number_text(metrics.get("f1"), f"{event_class} f1"),
                "value": "",
                "target": "",
                "passed": "",
                "sample_count": str(tp + fn),
            }
        )

    summaries = (
        ("macroF1", report.get("macroF1"), targets.get("macroF1"), "ge", report.get("testVideoCount")),
        (
            "freezeRecallAtLeast3s",
            report.get("freezeRecallAtLeast3s"),
            targets.get("freezeRecallAtLeast3s"),
            "ge",
            report.get("freezeEventsAtLeast3s"),
        ),
        (
            "lowMotionNegativeFreezeFalseAlarmRate",
            report.get("lowMotionNegativeFreezeFalseAlarmRate"),
            targets.get("lowMotionNegativeFreezeFalseAlarmRateMaximum"),
            "le",
            report.get("lowMotionNegativeVideoCount"),
        ),
    )
    for name, raw_value, raw_target, comparison, raw_count in summaries:
        value = _finite_number(raw_value, name)
        target = _finite_number(raw_target, f"{name} acceptance target")
        count = _nonnegative_integer(raw_count, f"{name} sample count")
        passed = value >= target if comparison == "ge" else value <= target
        rows.append(
            {
                **shared,
                "metric_type": "acceptance",
                "metric": name,
                "precision": "",
                "recall": "",
                "f1": "",
                "value": _number_text(value, name),
                "target": _number_text(target, f"{name} target"),
                "passed": str(passed).lower(),
                "sample_count": str(count),
            }
        )
    return rows


def _number_text(value: Any, label: str) -> str:
    return format(_finite_number(value, label), ".12g")


def _timeline_bar(event: dict[str, Any], duration: float, *, truth: bool) -> str:
    event_type = html.escape(str(event["type"]), quote=True)
    start = _finite_number(event.get("startTimeS"), "timeline startTimeS")
    end = _finite_number(event.get("endTimeS"), "timeline endTimeS")
    if start < 0 or end < start:
        raise EvidenceExportError("timeline event interval is invalid")
    if end > duration + 1e-6:
        raise EvidenceExportError("timeline event interval exceeds source duration")
    left = min(100.0, max(0.0, start / duration * 100.0))
    right = min(100.0, max(left, end / duration * 100.0))
    width = right - left
    title = f"{event_type}: {start:.3f}–{end:.3f}s"
    classes = "interval ground-truth" if truth else "interval prediction"
    return (
        f'<span class="{classes} event-{event_type}" '
        f'style="left:{left:.6f}%;width:{width:.6f}%" title="{html.escape(title, quote=True)}"></span>'
    )


def _timeline_html(
    report: dict[str, Any],
    results: dict[str, VideoQualityResult],
    manifest: dict[str, dict[str, Any]],
    input_hashes: dict[str, str],
    metric_rows: list[dict[str, str]],
    result_model_version: str,
) -> str:
    summary_rows = [row for row in metric_rows if row["metric_type"] == "acceptance"]
    summary_html = "\n".join(
        "<tr>"
        f"<th>{html.escape(row['metric'])}</th>"
        f"<td>{html.escape(row['value'])}</td>"
        f"<td>{html.escape(row['target'])}</td>"
        f"<td>{'达标' if row['passed'] == 'true' else '未达标'}</td>"
        f"<td>{html.escape(row['sample_count'])}</td>"
        "</tr>"
        for row in summary_rows
    )
    timeline_sections: list[str] = []
    for video_id in sorted(results):
        row = manifest[video_id]
        result = results[video_id]
        duration = _finite_number(row.get("sourceDurationS"), "sourceDurationS")
        truth = row.get("anomaly")
        truth_bar = _timeline_bar(truth, duration, truth=True) if truth else ""
        prediction_bars = "".join(
            _timeline_bar(anomaly.model_dump(mode="json", by_alias=True), duration, truth=False)
            for anomaly in result.anomalies
        )
        ground_truth_text = "无" if truth is None else (
            f"{html.escape(str(truth['type']))} "
            f"{_finite_number(truth['startTimeS'], 'ground-truth startTimeS'):.3f}–"
            f"{_finite_number(truth['endTimeS'], 'ground-truth endTimeS'):.3f}s"
        )
        predicted_text = "；".join(
            f"{html.escape(anomaly.type)} {anomaly.start_time_s:.3f}–{anomaly.end_time_s:.3f}s"
            for anomaly in result.anomalies
        ) or "无"
        video_label = html.escape(video_id, quote=True)
        source_id = html.escape(str(row.get("sourceVideoId", "")), quote=True)
        axis = (
            '<div class="axis"><span>0 秒</span>'
            f'<span>{duration / 2:.3f} 秒</span><span>{duration:.3f} 秒</span></div>'
        )
        timeline_sections.append(
            f'<section class="video" id="video-{len(timeline_sections) + 1}">'
            f"<h3>{video_label}</h3><p>来源视频：{source_id}　时长：{duration:.3f} 秒</p>"
            f'<div class="track-row"><span>真实标注</span><div class="track">{truth_bar}</div></div>'
            f'<div class="track-row"><span>模型预测</span><div class="track">{prediction_bars}</div></div>'
            f"{axis}<p>真值：{ground_truth_text}</p><p>预测：{predicted_text}</p></section>"
        )
    input_hash_html = "\n".join(
        f"<li>{html.escape(name)} SHA-256：<code>{digest}</code></li>"
        for name, digest in input_hashes.items()
    )
    event_model_version = html.escape(str(report["modelVersion"]))
    model_version = html.escape(result_model_version)
    dataset = html.escape(str(report.get("dataset", "")))
    count = len(results)
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline';">
  <title>任务三受控事件 test 时间轴证据</title>
  <style>
    :root {{ color-scheme: light; font-family: "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; }}
    body {{ margin: 2rem auto; max-width: 1100px; padding: 0 1rem 3rem; color: #17212b; }}
    h1, h2 {{ color: #12344d; }}
    .notice {{ border-left: 4px solid #ca5b27; padding: .75rem 1rem; background: #fff5ed; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #cbd5df; padding: .55rem; text-align: left; }}
    .video {{ border-top: 1px solid #cbd5df; padding: 1rem 0; }}
    .track-row {{ display: grid; grid-template-columns: 110px 1fr; gap: .75rem; align-items: center; margin: .5rem 0; }}
    .track {{ position: relative; height: 1.3rem; background: #edf1f4; border-radius: .2rem; }}
    .interval {{ position: absolute; top: 0; height: 100%; min-width: 2px; border-radius: .2rem; }}
    .ground-truth {{ background: #197a54; }}
    .prediction {{ background: #c55b30; opacity: .88; }}
    .axis {{ display: flex; justify-content: space-between; margin-left: 118px; color: #536576; font-size: .85rem; }}
    code {{ overflow-wrap: anywhere; }}
    @media (max-width: 650px) {{ .track-row {{ grid-template-columns: 80px 1fr; }} .axis {{ margin-left: 88px; }} }}
  </style>
</head>
<body>
  <h1>任务三受控事件 test 时间轴证据</h1>
  <p>数据集：{dataset}；结果模型版本：<code>{model_version}</code>；异常模型版本：<code>{event_model_version}</code>；test 视频数：{count}；test 窗口数：{int(report['testWindowCount'])}。</p>
  <p class="notice">该页面只呈现既有冻结 test 报告、预测 JSONL 和 manifest 真值；不重新推理。合成受控事件不等于真实设备故障验收，低运动代理也不是人工确认的静止场景标签。</p>
  <h2>验收指标</h2>
  <table><thead><tr><th>指标</th><th>实际值</th><th>任务目标</th><th>结论</th><th>样本数</th></tr></thead><tbody>{summary_html}</tbody></table>
  <h2>输入追溯</h2>
  <ul>{input_hash_html}</ul>
  <h2>逐视频时间轴</h2>
  <p>绿色为 manifest 中的受控事件真值；橙色为既有模型预测。每条轨道按对应视频秒数缩放。</p>
  {''.join(timeline_sections)}
</body>
</html>
'''


def render_report(
    report_path: Path,
    results_path: Path,
    manifest_path: Path,
    csv_path: Path,
    html_path: Path,
) -> dict[str, int]:
    inputs = [report_path.resolve(strict=True), results_path.resolve(strict=True), manifest_path.resolve(strict=True)]
    outputs = [csv_path.resolve(), html_path.resolve()]
    if outputs[0] == outputs[1] or set(inputs) & set(outputs):
        raise EvidenceExportError("output paths must be distinct from each other and the inputs")
    if any(path.exists() for path in outputs):
        raise EvidenceExportError("an output file already exists; refusing to overwrite")
    if any(not path.parent.is_dir() for path in outputs):
        raise EvidenceExportError("output parent directories must already exist")

    report = _read_json(inputs[0], "test report")
    if report.get("schemaVersion") != "task3.controlled-event-test/1.0":
        raise EvidenceExportError("test report schemaVersion is incompatible")
    if report.get("testUsedForSelection") is not False:
        raise EvidenceExportError("test report must declare testUsedForSelection=false")
    results = _load_results(inputs[1])
    manifest = _load_test_manifest(inputs[2])
    if set(results) != set(manifest):
        raise EvidenceExportError("test result and test manifest videoId sets do not match")
    test_video_count = _nonnegative_integer(report.get("testVideoCount"), "testVideoCount")
    _nonnegative_integer(report.get("testWindowCount"), "testWindowCount")
    if test_video_count != len(results):
        raise EvidenceExportError("test result count does not match test report")
    report_version = report.get("modelVersion")
    result_model_versions = {result.provenance.model_version for result in results.values()}
    if len(result_model_versions) != 1:
        raise EvidenceExportError("test results contain inconsistent model versions")
    result_model_version = next(iter(result_model_versions))
    if any(not version.startswith(f"{report_version}+") for version in result_model_versions):
        raise EvidenceExportError("result model versions do not match the frozen test report")

    input_hashes = {
        "test-report": _sha256(inputs[0]),
        "test-results": _sha256(inputs[1]),
        "manifest": _sha256(inputs[2]),
    }
    metric_rows = _metric_rows(report, input_hashes, result_model_version)
    csv_stream = io.StringIO(newline="")
    writer = csv.DictWriter(csv_stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(metric_rows)
    html_text = _timeline_html(report, results, manifest, input_hashes, metric_rows, result_model_version)

    created: list[Path] = []
    try:
        with csv_path.open("x", encoding="utf-8-sig", newline="") as handle:
            created.append(csv_path)
            handle.write(csv_stream.getvalue())
        with html_path.open("x", encoding="utf-8", newline="\n") as handle:
            created.append(html_path)
            handle.write(html_text)
    except OSError as exc:
        for path in created:
            path.unlink(missing_ok=True)
        raise EvidenceExportError("could not create both evidence outputs") from exc
    return {"testVideoCount": len(results), "csvMetricRows": len(metric_rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument("--html-out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        counts = render_report(args.report, args.results, args.manifest, args.csv_out, args.html_out)
    except EvidenceExportError as exc:
        parser.error(str(exc))
    print(json.dumps({"status": "exported", **counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
