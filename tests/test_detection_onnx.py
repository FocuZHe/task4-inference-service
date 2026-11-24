import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts" / "reports" / "detection-parity.json"
SAMPLE = ROOT / "任务一模型" / "demo" / "samples" / "01576.jpg"


def test_fp32_class_agreement_on_01576_and_all_samples():
    from serving.detection_runtime import build_detection_parity_report

    report = build_detection_parity_report()
    assert REPORT.is_file()
    saved = json.loads(REPORT.read_text(encoding="utf-8"))
    assert saved["rows"]
    fp32 = [row for row in report["rows"] if row["precision"] == "fp32"]
    assert len(fp32) == 12
    assert all(row["classAgreement"] == 1.0 for row in fp32)
    focus = [row for row in fp32 if row["sample"] == SAMPLE.name]
    assert len(focus) == 2
    assert all(row["classAgreement"] == 1.0 for row in focus)
