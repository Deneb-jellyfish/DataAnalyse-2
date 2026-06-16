"""Train ARIMA for hourly PM2.5 h1 and seq6 tasks."""

from __future__ import annotations

from datetime import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.arima_model import ARIMAHourlyModel
from utils.hourly_experiment import make_multi_horizon_prediction_frame, make_prediction_frame, metric_rows_from_predictions
from utils.hourly_output_paths import artifact_path, ensure_hourly_output_dirs
from utils.io import PROCESSED_DIR

OUTPUT_DIR = ROOT / "outputs" / "hourly"
MAX_TRAIN_LEN = 12000
ARIMA_ORDER = (7, 1, 3)
ARIMA_SEASONAL = (0, 0, 0, 0)


def log(message: str) -> None:
    """Print timestamped script progress messages."""
    print(f"[train_arima {datetime.now():%H:%M:%S}] {message}", flush=True)


def load_series() -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Load the Beijing hourly PM2.5 series."""
    df = pd.read_csv(PROCESSED_DIR / "beijing_hourly.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    pm25 = df["pm25"].astype(float).ffill().bfill().values
    return pm25.astype(np.float64), df["datetime"]


def fit_model(train_pm25: np.ndarray) -> ARIMAHourlyModel:
    """Fit one ARIMA model on the truncated train series."""
    original_len = len(train_pm25)
    if len(train_pm25) > MAX_TRAIN_LEN:
        train_pm25 = train_pm25[-MAX_TRAIN_LEN:]
    log(
        f"prepare fit: original_train_len={original_len}, "
        f"used_train_len={len(train_pm25)}"
    )
    model = ARIMAHourlyModel(order=ARIMA_ORDER, seasonal_order=ARIMA_SEASONAL)
    model.fit(train_pm25)
    return model


def run_h1(model: ARIMAHourlyModel, test_pm25: np.ndarray, test_dti: pd.DatetimeIndex) -> list[dict]:
    """Run rolling one-step evaluation."""
    log(f"start h1 stage: n_test={len(test_pm25)}")
    y_pred = model.predict_h1(test_pm25)
    valid = ~(np.isnan(test_pm25) | np.isnan(y_pred))
    pred_df = make_prediction_frame(
        forecast_times=test_dti[valid],
        y_true=test_pm25[valid],
        y_pred=y_pred[valid],
        horizon=1,
        model_name="ARIMA",
        task="h1",
    )
    pred_path = artifact_path("arima_predictions_h1.csv", ensure_parent=True)
    config_path = artifact_path("arima_best_order_h1.json", ensure_parent=True)
    pred_df.to_csv(pred_path, index=False)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump({"order": list(ARIMA_ORDER), "seasonal_order": list(ARIMA_SEASONAL)}, handle, ensure_ascii=False, indent=2)
    log(
        f"h1 outputs saved: rows={len(pred_df)}, "
        f"path={pred_path}"
    )
    return metric_rows_from_predictions(
        pred_df,
        model_name="ARIMA",
        task="h1",
        notes=f"order={ARIMA_ORDER},seasonal={ARIMA_SEASONAL}",
    )


def run_seq6(model: ARIMAHourlyModel, test_pm25: np.ndarray, test_dti: pd.DatetimeIndex) -> list[dict]:
    """Run rolling six-step evaluation and save long-format outputs."""
    log(f"start seq6 stage: n_test={len(test_pm25)}")
    pred_matrix = model.predict_steps(test_pm25, steps=6)
    n_rows = len(test_pm25) - 6
    y_true_matrix = np.zeros((n_rows, 6), dtype=np.float64)
    for index in range(n_rows):
        y_true_matrix[index] = test_pm25[index + 1 : index + 7]

    pred_df = make_multi_horizon_prediction_frame(
        forecast_times=test_dti[:n_rows],
        y_true=y_true_matrix,
        y_pred=pred_matrix[:n_rows, :6],
        horizons=range(1, 7),
        model_name="ARIMA",
        task="seq6",
    )
    pred_df = pred_df.dropna(subset=["y_true", "y_pred"]).reset_index(drop=True)
    pred_path = artifact_path("arima_predictions_seq6.csv", ensure_parent=True)
    config_path = artifact_path("arima_best_order_seq6.json", ensure_parent=True)
    pred_df.to_csv(pred_path, index=False)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump({"order": list(ARIMA_ORDER), "seasonal_order": list(ARIMA_SEASONAL)}, handle, ensure_ascii=False, indent=2)
    log(
        f"seq6 outputs saved: rows={len(pred_df)}, "
        f"path={pred_path}"
    )
    return metric_rows_from_predictions(
        pred_df,
        model_name="ARIMA",
        task="seq6",
        notes=f"order={ARIMA_ORDER},seasonal={ARIMA_SEASONAL},rolling_steps=6",
    )


def save_metrics(rows: list[dict]) -> None:
    """Persist combined metrics."""
    path = artifact_path("arima_metrics.csv", ensure_parent=True)
    df = pd.DataFrame(rows)
    cols = ["model", "task", "horizon_hours", "rmse", "mae", "mape", "n_samples", "notes"]
    for column in cols:
        if column not in df.columns:
            df[column] = ""
    df = df[cols].sort_values(["task", "horizon_hours"]).reset_index(drop=True)
    df.to_csv(path, index=False)
    log(f"metrics saved: rows={len(df)}, path={path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_hourly_output_dirs()

    pm25, dti = load_series()
    split_idx = int(len(pm25) * 0.8)
    train_pm25 = pm25[:split_idx]
    test_pm25 = pm25[split_idx:]
    test_dti = dti[split_idx:].reset_index(drop=True)
    log(
        f"loaded series: total={len(pm25)}, split_idx={split_idx}, "
        f"train={len(train_pm25)}, test={len(test_pm25)}"
    )

    rows: list[dict] = []
    rows.extend(run_h1(fit_model(train_pm25), test_pm25, test_dti))
    rows.extend(run_seq6(fit_model(train_pm25), test_pm25, test_dti))
    save_metrics(rows)
    log("all stages complete")


if __name__ == "__main__":
    main()
