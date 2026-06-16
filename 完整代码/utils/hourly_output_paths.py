"""Shared path helpers for organizing hourly experiment artifacts."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOURLY_OUTPUT_DIR = ROOT / "outputs" / "hourly"

SUMMARY_FILES = {
    "overall_metrics_summary.csv",
    "experiment_registry.csv",
    "analysis_notes.md",
}

ANALYSIS_FILES = {
    "aqi_bucket_metrics.csv",
    "cross_city_metrics.csv",
    "feature_ablation.csv",
    "high_pollution_error_analysis.csv",
    "hourly_error_by_dayperiod.csv",
    "hourly_error_by_hour.csv",
    "xgboost_feature_importance_h1.csv",
    "xgboost_feature_importance_seq6.csv",
}

CATEGORY_DIRS = {
    "summaries": HOURLY_OUTPUT_DIR / "summaries",
    "metrics": HOURLY_OUTPUT_DIR / "metrics",
    "predictions": HOURLY_OUTPUT_DIR / "predictions",
    "analyses": HOURLY_OUTPUT_DIR / "analyses",
    "configs": HOURLY_OUTPUT_DIR / "configs",
    "histories": HOURLY_OUTPUT_DIR / "histories",
    "models": HOURLY_OUTPUT_DIR / "models",
    "figures": HOURLY_OUTPUT_DIR / "figures",
}


def category_for_artifact(filename: str) -> str:
    """Map an hourly artifact filename to its semantic subdirectory."""
    if filename.endswith(".png"):
        return "figures"
    if filename in SUMMARY_FILES:
        return "summaries"
    if filename in ANALYSIS_FILES:
        return "analyses"
    if "predictions" in filename and filename.endswith(".csv"):
        return "predictions"
    if filename.endswith("_metrics.csv"):
        return "metrics"
    if filename.endswith("_train_history.json"):
        return "histories"
    if filename.endswith(".json"):
        return "configs"
    if filename.endswith(".pkl") or filename.endswith(".pt"):
        return "models"
    return "summaries"


def artifact_relpath(filename: str) -> Path:
    """Return the subdirectory-relative path for an hourly artifact."""
    return Path(category_for_artifact(filename)) / filename


def artifact_path(filename: str, ensure_parent: bool = False) -> Path:
    """Return the organized path for an hourly artifact."""
    path = HOURLY_OUTPUT_DIR / artifact_relpath(filename)
    if ensure_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_artifact_path(filename: str) -> Path:
    """Resolve an existing artifact from either the new or legacy flat layout."""
    organized = artifact_path(filename)
    legacy = HOURLY_OUTPUT_DIR / filename
    if organized.exists():
        return organized
    if legacy.exists():
        return legacy
    return organized


def relative_artifact_string(filename: str) -> str:
    """Return a stable registry-friendly relative path with forward slashes."""
    return artifact_path(filename).relative_to(ROOT).as_posix()


def ensure_hourly_output_dirs() -> None:
    """Create the hourly artifact root and all semantic subdirectories."""
    HOURLY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in CATEGORY_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def organize_existing_hourly_outputs() -> list[tuple[Path, Path]]:
    """Move legacy root-level files into the organized subdirectories."""
    ensure_hourly_output_dirs()
    moves: list[tuple[Path, Path]] = []
    for path in HOURLY_OUTPUT_DIR.iterdir():
        if not path.is_file():
            continue
        target = artifact_path(path.name, ensure_parent=True)
        if target == path or target.exists():
            continue
        path.replace(target)
        moves.append((path, target))
    return moves
