"""Safety checks shared by quality-regression experiment writers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_DATA_ROOT = REPOSITORY_ROOT / "data" / "raw" / "konvid-1k"
EXPERIMENT_ROOT = FROZEN_DATA_ROOT / "quality-experiments"
FROZEN_MODEL_ROOT = FROZEN_DATA_ROOT / "models-official"
FROZEN_FEATURES = FROZEN_DATA_ROOT / "features-official.jsonl"
FROZEN_REPORTS = {
    FROZEN_MODEL_ROOT / "training_report.json",
    FROZEN_MODEL_ROOT / "test_evaluation.json",
    FROZEN_MODEL_ROOT / "test_evaluation-reproducible.json",
}


def ensure_experiment_output_dir(output_dir: Path) -> Path:
    """Return a resolved output directory unless it overlaps frozen artifacts."""
    resolved = output_dir.resolve()
    if resolved == FROZEN_FEATURES or FROZEN_FEATURES in resolved.parents:
        raise ValueError(f"experiment output must not overlap frozen features: {output_dir}")
    if resolved == FROZEN_MODEL_ROOT or FROZEN_MODEL_ROOT in resolved.parents:
        raise ValueError(f"experiment output must not overlap frozen models: {output_dir}")
    if any(report == resolved or report in resolved.parents for report in FROZEN_REPORTS):
        raise ValueError(f"experiment output must not overlap frozen report: {output_dir}")
    if (resolved == REPOSITORY_ROOT or REPOSITORY_ROOT in resolved.parents) and not (
        resolved == EXPERIMENT_ROOT or EXPERIMENT_ROOT in resolved.parents
    ):
        raise ValueError(f"experiment output must be under quality-experiments: {output_dir}")
    return resolved


def refuse_existing_outputs(output_dir: Path, filenames: Iterable[str]) -> Path:
    """Validate an output directory and fail before touching any existing target."""
    resolved = ensure_experiment_output_dir(output_dir)
    existing = [resolved / filename for filename in filenames if (resolved / filename).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing output: {existing[0]}")
    return resolved


def refuse_existing_output(output: Path) -> Path:
    """Validate a single output file before its parent is created."""
    resolved = output.resolve()
    if resolved == FROZEN_FEATURES or resolved in FROZEN_REPORTS:
        raise ValueError(f"experiment output must not overlap frozen artifact: {output}")
    ensure_experiment_output_dir(resolved.parent)
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    return resolved
