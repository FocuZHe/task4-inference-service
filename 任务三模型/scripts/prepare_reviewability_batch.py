"""Prepare blind reviewability batches or compare and resolve human labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from reviewability_annotations import compare_annotators, prepare_blind_batches, resolve_adjudicated_labels


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        values.append(value)
    return values


def _write_exclusive(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(content)


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows)


def _write_compare_outputs(
    report_path: Path,
    final_path: Path,
    report: dict[str, Any],
    final_rows: list[dict[str, Any]] | None,
) -> None:
    if report_path.resolve() == final_path.resolve():
        raise ValueError("report and final-label outputs must be different paths")
    if report_path.exists():
        raise FileExistsError(report_path)
    if final_path.exists():
        raise FileExistsError(final_path)
    report_content = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _write_exclusive(report_path, report_content)
    if final_rows is not None:
        _write_exclusive(final_path, _jsonl(final_rows))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--annotator-a", type=Path, required=True)
    prepare.add_argument("--annotator-b", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=42)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--annotator-a", type=Path, required=True)
    compare.add_argument("--annotator-b", type=Path, required=True)
    compare.add_argument("--adjudication", type=Path, required=True)
    compare.add_argument("--report", type=Path, required=True)
    compare.add_argument("--final-labels", type=Path, required=True)
    compare.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        if args.annotator_a.resolve() == args.annotator_b.resolve():
            raise ValueError("annotator output paths must be different")
        if args.annotator_a.exists() or args.annotator_b.exists():
            raise FileExistsError(args.annotator_a if args.annotator_a.exists() else args.annotator_b)
        manifest = _read_jsonl(args.manifest)
        batch_a, batch_b = prepare_blind_batches(manifest, seed=args.seed, annotators=("annotator-a", "annotator-b"))
        _write_exclusive(args.annotator_a, _jsonl(batch_a))
        try:
            _write_exclusive(args.annotator_b, _jsonl(batch_b))
        except Exception:
            args.annotator_a.unlink(missing_ok=True)
            raise
        print(f"wrote {len(batch_a)} blank rows for two independent annotators")
        return 0

    rows_a = _read_jsonl(args.annotator_a)
    rows_b = _read_jsonl(args.annotator_b)
    adjudications = _read_jsonl(args.adjudication)
    report = compare_annotators(rows_a, rows_b)
    final_rows: list[dict[str, Any]] | None = None
    resolution_error: str | None = None
    try:
        final_rows = resolve_adjudicated_labels(rows_a, rows_b, adjudications)
    except ValueError as exc:
        resolution_error = str(exc)
    report.update({"ready": final_rows is not None, "resolutionError": resolution_error})
    _write_compare_outputs(args.report, args.final_labels, report, final_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if final_rows is not None or not args.strict else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
