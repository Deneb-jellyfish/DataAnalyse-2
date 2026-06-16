"""Shared helpers for hourly h1 and seq6 experiment outputs."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from evaluation.metrics import compute_metrics


def make_prediction_frame(
    forecast_times: np.ndarray | pd.Series | pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    model_name: str,
    task: str,
    city: str = "Beijing",
    split: str = "test",
) -> pd.DataFrame:
    """Build a normalized long-format prediction table for one horizon."""
    forecast_dt = pd.to_datetime(forecast_times)
    forecast_text = pd.Series(forecast_dt).dt.strftime("%Y-%m-%d %H:%M:%S")
    target_text = (pd.Series(forecast_dt) + pd.Timedelta(hours=int(horizon))).dt.strftime("%Y-%m-%d %H:%M:%S")
    return pd.DataFrame(
        {
            "forecast_origin_time": forecast_text.to_numpy(),
            "target_time": target_text.to_numpy(),
            "task": task,
            "horizon_hours": int(horizon),
            "city": city,
            "model": model_name,
            "y_true": np.asarray(y_true, dtype=np.float64),
            "y_pred": np.asarray(y_pred, dtype=np.float64),
            "split": split,
        }
    )


def make_multi_horizon_prediction_frame(
    forecast_times: np.ndarray | pd.Series | pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: Iterable[int],
    model_name: str,
    task: str = "seq6",
    city: str = "Beijing",
    split: str = "test",
) -> pd.DataFrame:
    """Expand multi-output predictions into the shared long-format table."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("y_true and y_pred must be 2-D for multi-horizon output")

    horizons = list(horizons)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    if y_true.shape[1] != len(horizons):
        raise ValueError("Number of horizons must match the second dimension")

    frames = []
    for index, horizon in enumerate(horizons):
        frames.append(
            make_prediction_frame(
                forecast_times=forecast_times,
                y_true=y_true[:, index],
                y_pred=y_pred[:, index],
                horizon=int(horizon),
                model_name=model_name,
                task=task,
                city=city,
                split=split,
            )
        )
    return pd.concat(frames, ignore_index=True)


def metric_rows_from_predictions(
    pred_df: pd.DataFrame,
    model_name: str,
    task: str,
    notes: str = "",
) -> list[dict]:
    """Compute one metrics row per horizon from a long-format prediction table."""
    rows: list[dict] = []
    for horizon in sorted(pred_df["horizon_hours"].dropna().astype(int).unique()):
        sub = pred_df[pred_df["horizon_hours"] == horizon]
        metrics = compute_metrics(sub["y_true"].values, sub["y_pred"].values)
        rows.append(
            {
                "model": model_name,
                "task": task,
                "horizon_hours": int(horizon),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "mape": metrics["mape"],
                "n_samples": int(metrics["n_samples"]),
                "notes": notes,
            }
        )
    return rows
