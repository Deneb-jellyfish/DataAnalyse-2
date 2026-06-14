"""Train Prophet for hourly PM2.5 h1 and seq6 tasks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.hourly_experiment import make_multi_horizon_prediction_frame, make_prediction_frame, metric_rows_from_predictions
from utils.io import PROCESSED_DIR

OUTPUT_DIR = ROOT / "outputs" / "hourly"


def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the Beijing hourly series and split chronologically."""
    df = pd.read_csv(PROCESSED_DIR / "beijing_hourly.csv")
    df["ds"] = pd.to_datetime(df["datetime"])
    df["y"] = df["pm25"].astype(float)
    df = df.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return df, train_df, test_df


def fit_prophet(train_df: pd.DataFrame):
    """Fit one Prophet model on the train split."""
    from prophet import Prophet

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )
    t0 = time.time()
    model.fit(train_df[["ds", "y"]].copy())
    fit_time = time.time() - t0
    return model, fit_time


def run_h1(model, fit_time: float, full_lookup: dict[pd.Timestamp, float], test_df: pd.DataFrame) -> list[dict]:
    """Generate h1 prediction and metrics artifacts."""
    target_times = pd.to_datetime(test_df["ds"]) + pd.Timedelta(hours=1)
    t0 = time.time()
    forecast = model.predict(pd.DataFrame({"ds": target_times}))
    pred_time = time.time() - t0

    y_true = np.array([full_lookup.get(pd.Timestamp(ts), np.nan) for ts in target_times], dtype=np.float64)
    y_pred = forecast["yhat"].values.astype(np.float64)
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))

    pred_df = make_prediction_frame(
        forecast_times=test_df["ds"].values[valid],
        y_true=y_true[valid],
        y_pred=y_pred[valid],
        horizon=1,
        model_name="Prophet",
        task="h1",
    )
    pred_df.to_csv(OUTPUT_DIR / "prophet_predictions_h1.csv", index=False)

    with open(OUTPUT_DIR / "prophet_config_h1.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "task": "h1",
                "horizon_hours": 1,
                "daily_seasonality": True,
                "weekly_seasonality": True,
                "yearly_seasonality": True,
                "fit_time_s": round(fit_time, 1),
                "pred_time_s": round(pred_time, 1),
                "status": "done",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return metric_rows_from_predictions(
        pred_df,
        model_name="Prophet",
        task="h1",
        notes=f"fit_time_s={fit_time:.1f},pred_time_s={pred_time:.1f}",
    )


def run_seq6(model, fit_time: float, full_lookup: dict[pd.Timestamp, float], test_df: pd.DataFrame) -> list[dict]:
    """Generate seq6 prediction and metrics artifacts."""
    n_rows = len(test_df) - 6
    origin_times = pd.to_datetime(test_df["ds"].iloc[:n_rows]).reset_index(drop=True)
    y_true_matrix = np.zeros((n_rows, 6), dtype=np.float64)
    for index in range(n_rows):
        y_true_matrix[index] = test_df["y"].iloc[index + 1 : index + 7].to_numpy(dtype=np.float64)

    target_grid = {
        horizon: origin_times + pd.Timedelta(hours=horizon)
        for horizon in range(1, 7)
    }
    future_df = pd.DataFrame({"ds": pd.Index(np.concatenate([target_grid[h].values for h in range(1, 7)]))})

    t0 = time.time()
    forecast = model.predict(future_df)
    pred_time = time.time() - t0
    yhat = forecast["yhat"].values.astype(np.float64).reshape(6, n_rows).T

    pred_df = make_multi_horizon_prediction_frame(
        forecast_times=origin_times.values,
        y_true=y_true_matrix,
        y_pred=yhat,
        horizons=range(1, 7),
        model_name="Prophet",
        task="seq6",
    )
    pred_df.to_csv(OUTPUT_DIR / "prophet_predictions_seq6.csv", index=False)

    with open(OUTPUT_DIR / "prophet_config_seq6.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "task": "seq6",
                "horizons": [1, 2, 3, 4, 5, 6],
                "daily_seasonality": True,
                "weekly_seasonality": True,
                "yearly_seasonality": True,
                "fit_time_s": round(fit_time, 1),
                "pred_time_s": round(pred_time, 1),
                "status": "done",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return metric_rows_from_predictions(
        pred_df,
        model_name="Prophet",
        task="seq6",
        notes=f"fit_time_s={fit_time:.1f},pred_time_s={pred_time:.1f},rolling_steps=6",
    )


def save_metrics(rows: list[dict]) -> None:
    """Persist combined metrics."""
    path = OUTPUT_DIR / "prophet_metrics.csv"
    df = pd.DataFrame(rows)
    cols = ["model", "task", "horizon_hours", "rmse", "mae", "mape", "n_samples", "notes"]
    for column in cols:
        if column not in df.columns:
            df[column] = ""
    df = df[cols].sort_values(["task", "horizon_hours"]).reset_index(drop=True)
    df.to_csv(path, index=False)


def write_blocked_outputs(reason: str) -> None:
    """Write blocked configs and empty metrics when Prophet cannot run."""
    rows = []
    for task, horizons in [("h1", [1]), ("seq6", [1, 2, 3, 4, 5, 6])]:
        for horizon in horizons:
            rows.append(
                {
                    "model": "Prophet",
                    "task": task,
                    "horizon_hours": horizon,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "mape": np.nan,
                    "n_samples": 0,
                    "notes": f"blocked:{reason}",
                }
            )
        config_path = OUTPUT_DIR / ("prophet_config_h1.json" if task == "h1" else "prophet_config_seq6.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump({"task": task, "status": "blocked", "reason": reason}, handle, ensure_ascii=False, indent=2)
    save_metrics(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Prophet for hourly h1 and seq6 tasks")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually run Prophet training (default: write blocked outputs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.run:
        write_blocked_outputs(reason="skipped_by_request")
        print("Prophet skipped by request. Use --run to enable real Prophet training.")
        return

    try:
        from prophet import Prophet  # noqa: F401
    except Exception as exc:
        write_blocked_outputs(reason=f"prophet_import_failed:{exc.__class__.__name__}")
        print("Prophet blocked: prophet package is unavailable in current environment.")
        return

    _, train_df, test_df = load_and_split()
    full_lookup = dict(zip(pd.to_datetime(pd.concat([train_df["ds"], test_df["ds"]]).reset_index(drop=True)), pd.concat([train_df["y"], test_df["y"]]).reset_index(drop=True)))
    model, fit_time = fit_prophet(train_df)

    rows: list[dict] = []
    rows.extend(run_h1(model, fit_time, full_lookup, test_df))
    rows.extend(run_seq6(model, fit_time, full_lookup, test_df))
    save_metrics(rows)


if __name__ == "__main__":
    main()
