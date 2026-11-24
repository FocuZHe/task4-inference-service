"""Run exclusive train-only quality diagnosis, audit, extraction, study and verification."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["diagnose", "audit", "extract", "study", "verify"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    from quality_research.runner import run_stage
    return run_stage(args.stage, args.root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
