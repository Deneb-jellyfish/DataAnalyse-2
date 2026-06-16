"""Assemble summary tables and registry for the hourly experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.metrics import compute_metrics
from utils.hourly_output_paths import artifact_path, ensure_hourly_output_dirs, relative_artifact_string, resolve_artifact_path

OUTPUT_DIR = ROOT / "outputs" / "hourly"

METRIC_FILES = [
    "arima_metrics.csv",
    "prophet_metrics.csv",
    "xgboost_metrics.csv",
    "lstm_metrics.csv",
    "transformer_metrics.csv",
]

PREDICTION_TASK_FILES = {
    ("ARIMA", "h1"): "arima_predictions_h1.csv",
    ("ARIMA", "seq6"): "arima_predictions_seq6.csv",
    ("Prophet", "h1"): "prophet_predictions_h1.csv",
    ("Prophet", "seq6"): "prophet_predictions_seq6.csv",
    ("XGBoost", "h1"): "xgboost_predictions_h1.csv",
    ("XGBoost", "seq6"): "xgboost_predictions_seq6.csv",
    ("LSTM", "h1"): "lstm_predictions_h1.csv",
    ("LSTM", "seq6"): "lstm_predictions_seq6.csv",
    ("Transformer", "h1"): "transformer_predictions_h1.csv",
    ("Transformer", "seq6"): "transformer_predictions_seq6.csv",
}

EXPECTED_ARTIFACTS = [
    "xgboost_predictions_h1.csv",
    "xgboost_predictions_seq6.csv",
    "arima_predictions_h1.csv",
    "arima_predictions_seq6.csv",
    "prophet_predictions_h1.csv",
    "prophet_predictions_seq6.csv",
    "lstm_predictions_h1.csv",
    "lstm_predictions_seq6.csv",
    "transformer_predictions_h1.csv",
    "transformer_predictions_seq6.csv",
    "overall_metrics_summary.csv",
    "feature_ablation.csv",
    "aqi_bucket_metrics.csv",
    "high_pollution_error_analysis.csv",
    "hourly_error_by_hour.csv",
    "hourly_error_by_dayperiod.csv",
    "cross_city_metrics.csv",
    "cross_city_predictions.csv",
]


def read_metric_files() -> pd.DataFrame:
    """Read and concatenate all model metric CSVs."""
    frames: list[pd.DataFrame] = []
    for name in METRIC_FILES:
        path = resolve_artifact_path(name)
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise RuntimeError("No metrics files found under outputs/hourly")
    df = pd.concat(frames, ignore_index=True)
    if "task" not in df.columns:
        df["task"] = ""
    return df


def build_summary_rows() -> pd.DataFrame:
    """Compute task-level summary rows from prediction files."""
    rows: list[dict] = []
    for (model, task), filename in PREDICTION_TASK_FILES.items():
        path = resolve_artifact_path(filename)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "split" in df.columns:
            df = df[df["split"] == "test"].copy()
        if df.empty:
            continue
        metrics = compute_metrics(df["y_true"].values, df["y_pred"].values)
        summary_task = "seq6_mean" if task == "seq6" else "h1"
        rows.append(
            {
                "model": model,
                "task": summary_task,
                "horizon_hours": 0 if task == "seq6" else 1,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "mape": metrics["mape"],
                "n_samples": int(metrics["n_samples"]),
                "notes": "summary_from_prediction_rows",
            }
        )
    return pd.DataFrame(rows)


def write_overall_metrics() -> pd.DataFrame:
    """Write the combined overall metrics CSV."""
    raw_metrics = read_metric_files()
    summary_rows = build_summary_rows()
    overall = pd.concat([raw_metrics, summary_rows], ignore_index=True, sort=False)
    overall = overall.sort_values(["model", "task", "horizon_hours"]).reset_index(drop=True)
    overall.to_csv(artifact_path("overall_metrics_summary.csv", ensure_parent=True), index=False)
    return overall


def write_registry() -> pd.DataFrame:
    """Write expected artifact registry with existence flags."""
    rows = []
    for relative_path in EXPECTED_ARTIFACTS:
        path = resolve_artifact_path(relative_path)
        rows.append(
            {
                "path": relative_artifact_string(relative_path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(artifact_path("experiment_registry.csv", ensure_parent=True), index=False)
    return df


def write_analysis_notes(overall: pd.DataFrame, registry: pd.DataFrame) -> None:
    """Write a short markdown summary for quick acceptance."""
    lines = ["# Hourly Experiment Notes", ""]

    h1 = overall[overall["task"] == "h1"].dropna(subset=["rmse"]).sort_values("rmse")
    seq6_mean = overall[overall["task"] == "seq6_mean"].dropna(subset=["rmse"]).sort_values("rmse")
    seq6_steps = overall[overall["task"] == "seq6"].dropna(subset=["rmse"]).sort_values(["model", "horizon_hours"])

    if not h1.empty:
        best = h1.iloc[0]
        lines.append(f"- Best h1 model: `{best['model']}` with RMSE `{best['rmse']:.3f}` and MAE `{best['mae']:.3f}`.")
    if not seq6_mean.empty:
        best = seq6_mean.iloc[0]
        lines.append(f"- Best seq6 mean model: `{best['model']}` with RMSE `{best['rmse']:.3f}` and MAE `{best['mae']:.3f}`.")

    if not seq6_steps.empty:
        xgb_seq6 = seq6_steps[seq6_steps["model"] == "XGBoost"].sort_values("horizon_hours")
        if not xgb_seq6.empty:
            trend = "nondecreasing" if xgb_seq6["rmse"].is_monotonic_increasing else "mixed"
            rmse_values = ", ".join(
                f"h{int(row.horizon_hours)}={row.rmse:.2f}" for row in xgb_seq6.itertuples()
            )
            lines.append(f"- XGBoost seq6 RMSE trend: `{trend}` ({rmse_values}).")

    missing = registry[~registry["exists"]]
    if missing.empty:
        lines.append("- All expected experiment artifacts are present.")
    else:
        lines.append("- Missing artifacts:")
        for row in missing.itertuples():
            lines.append(f"  - `{row.path}`")

    artifact_path("analysis_notes.md", ensure_parent=True).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_hourly_output_dirs()
    overall = write_overall_metrics()
    registry = write_registry()
    write_analysis_notes(overall, registry)
    print("Generated overall_metrics_summary.csv, experiment_registry.csv, analysis_notes.md")


if __name__ == "__main__":
    main()
