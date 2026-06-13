"""Train ARIMA for hourly PM2.5 prediction tasks (+1h and +12h)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.metrics import compute_metrics
from models.arima_model import ARIMAHourlyModel
from utils.io import PROCESSED_DIR

OUTPUT_DIR = ROOT / "outputs" / "hourly"
MAX_TRAIN_LEN = 12000
ARIMA_ORDER = (7, 1, 3)
ARIMA_SEASONAL = (0, 0, 0, 0)


def load_series() -> tuple[np.ndarray, pd.DatetimeIndex]:
    df = pd.read_csv(PROCESSED_DIR / "beijing_hourly.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    pm25 = df["pm25"].astype(float).ffill().bfill().values
    return pm25.astype(np.float64), df["datetime"]


def make_prediction_records(
    forecast_times: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    model_name: str = "ARIMA",
    city: str = "Beijing",
    split: str = "test",
) -> list[dict]:
    forecast_dt = pd.to_datetime(forecast_times).to_numpy()
    target_dt = forecast_dt + np.timedelta64(horizon, "h")
    records = []
    for i in range(len(forecast_dt)):
        records.append(
            {
                "forecast_origin_time": pd.Timestamp(forecast_dt[i]).strftime("%Y-%m-%d %H:%M:%S"),
                "target_time": pd.Timestamp(target_dt[i]).strftime("%Y-%m-%d %H:%M:%S"),
                "horizon_hours": int(horizon),
                "city": city,
                "model": model_name,
                "y_true": float(y_true[i]),
                "y_pred": float(y_pred[i]),
                "split": split,
            }
        )
    return records


def run_h1(model: ARIMAHourlyModel, test_pm25: np.ndarray, test_dti: pd.DatetimeIndex) -> tuple[list[dict], dict]:
    all_preds = model.predict_h1(test_pm25)
    eval_idx = np.arange(len(test_pm25))

    y_true = test_pm25[eval_idx]
    y_pred = all_preds[eval_idx]
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))

    y_true_v, y_pred_v = y_true[valid], y_pred[valid]
    dti_v = test_dti[eval_idx][valid]
    metrics = compute_metrics(y_true_v, y_pred_v)

    records = make_prediction_records(dti_v, y_true_v, y_pred_v, horizon=1)
    row = {
        "model": "ARIMA",
        "horizon_hours": 1,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mape": metrics["mape"],
        "n_samples": int(metrics["n_samples"]),
        "notes": f"order={model.order},seasonal={model.seasonal_order}",
    }
    return records, row


def run_h12(model: ARIMAHourlyModel, test_pm25: np.ndarray, test_dti: pd.DatetimeIndex) -> tuple[list[dict], dict]:
    all_preds = model.predict_h12(test_pm25)
    eval_idx = np.arange(0, len(test_pm25) - 11)

    y_true = test_pm25[eval_idx]
    y_pred = all_preds[eval_idx]
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))

    y_true_v, y_pred_v = y_true[valid], y_pred[valid]
    dti_v = test_dti[eval_idx][valid]
    metrics = compute_metrics(y_true_v, y_pred_v)

    records = make_prediction_records(dti_v, y_true_v, y_pred_v, horizon=12)
    row = {
        "model": "ARIMA",
        "horizon_hours": 12,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mape": metrics["mape"],
        "n_samples": int(metrics["n_samples"]),
        "notes": f"order={model.order},seasonal={model.seasonal_order}",
    }
    return records, row


def save_predictions(records: list[dict], path: Path) -> None:
    fieldnames = [
        "forecast_origin_time",
        "target_time",
        "horizon_hours",
        "city",
        "model",
        "y_true",
        "y_pred",
        "split",
    ]
    pd.DataFrame(records, columns=fieldnames).to_csv(path, index=False)


def save_metrics(rows: list[dict]) -> None:
    path = OUTPUT_DIR / "arima_metrics.csv"
    df = pd.DataFrame(rows)
    cols = ["model", "horizon_hours", "rmse", "mae", "mape", "n_samples", "notes"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]
    df.to_csv(path, index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pm25, dti = load_series()
    split_idx = int(len(pm25) * 0.8)
    train_pm25 = pm25[:split_idx]
    test_pm25 = pm25[split_idx:]
    test_dti = dti[split_idx:].reset_index(drop=True)

    if len(train_pm25) > MAX_TRAIN_LEN:
        train_pm25 = train_pm25[-MAX_TRAIN_LEN:]

    model_h1 = ARIMAHourlyModel(order=ARIMA_ORDER, seasonal_order=ARIMA_SEASONAL)
    model_h1.fit(train_pm25)
    recs_h1, metrics_h1 = run_h1(model_h1, test_pm25, test_dti)
    save_predictions(recs_h1, OUTPUT_DIR / "arima_predictions_h1.csv")

    model_h12 = ARIMAHourlyModel(order=ARIMA_ORDER, seasonal_order=ARIMA_SEASONAL)
    model_h12.fit(train_pm25)
    recs_h12, metrics_h12 = run_h12(model_h12, test_pm25, test_dti)
    save_predictions(recs_h12, OUTPUT_DIR / "arima_predictions_h12.csv")

    with open(OUTPUT_DIR / "arima_best_order_h1.json", "w", encoding="utf-8") as f:
        json.dump({"order": list(ARIMA_ORDER), "seasonal_order": list(ARIMA_SEASONAL)}, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / "arima_best_order_h12.json", "w", encoding="utf-8") as f:
        json.dump({"order": list(ARIMA_ORDER), "seasonal_order": list(ARIMA_SEASONAL)}, f, ensure_ascii=False, indent=2)

    save_metrics([metrics_h1, metrics_h12])


if __name__ == "__main__":
    main()
