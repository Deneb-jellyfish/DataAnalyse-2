"""Regression metrics and AQI-stratified evaluation for hourly PM2.5 prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute standard regression metrics.

    Returns dict with keys: rmse, mae, mape, n_samples.
    MAPE is in percentage (0-100). Samples where y_true == 0 are excluded from MAPE.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = len(y_true)

    if n == 0:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "n_samples": 0}

    error = y_true - y_pred
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))

    # MAPE: exclude zero or near-zero true values to avoid division explosion
    mape_mask = np.abs(y_true) > 1e-6
    if mape_mask.sum() > 0:
        mape = float(
            np.mean(np.abs(error[mape_mask] / y_true[mape_mask])) * 100
        )
    else:
        mape = np.nan

    return {"rmse": rmse, "mae": mae, "mape": mape, "n_samples": n}


# ---------------------------------------------------------------------------
# AQI bucket definitions (Chinese PM2.5 breakpoints, hourly approximation)
# ---------------------------------------------------------------------------
AQI_BREAKPOINTS = [
    (0.0, 12.0, "Good"),
    (12.0, 35.4, "Moderate"),
    (35.4, 55.4, "Unhealthy-Sensitive"),
    (55.4, 150.4, "Unhealthy"),
    (150.4, 250.4, "Very Unhealthy"),
    (250.4, float("inf"), "Hazardous"),
]


def aqi_bucket(pm25: float) -> str:
    """Return AQI bucket label for a single PM2.5 value."""
    if np.isnan(pm25):
        return "Unknown"
    for low, high, label in AQI_BREAKPOINTS:
        if low <= pm25 < high:
            return label
    return "Hazardous"


def compute_bucket_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Compute RMSE/MAE/MAPE per AQI bucket."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    buckets = [aqi_bucket(v) for v in y_true]
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "bucket": buckets})

    records = []
    for bucket in ["Good", "Moderate", "Unhealthy-Sensitive", "Unhealthy",
                   "Very Unhealthy", "Hazardous"]:
        sub = df[df["bucket"] == bucket]
        if len(sub) == 0:
            continue
        m = compute_metrics(sub["y_true"].values, sub["y_pred"].values)
        records.append({
            "aqi_bucket": bucket,
            "rmse": m["rmse"],
            "mae": m["mae"],
            "mape": m["mape"],
            "n_samples": m["n_samples"],
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Diurnal / hourly error helpers
# ---------------------------------------------------------------------------

DAY_PERIOD_MAP = {
    **{h: "凌晨" for h in range(0, 6)},
    **{h: "早高峰" for h in range(6, 10)},
    **{h: "白天平峰" for h in range(10, 17)},
    **{h: "晚高峰" for h in range(17, 21)},
    **{h: "夜间" for h in range(21, 24)},
}


def compute_hourly_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hours: np.ndarray,
) -> pd.DataFrame:
    """Compute RMSE/MAE per hour (0–23)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    hours = np.asarray(hours, dtype=int)

    records = []
    for h in range(24):
        mask = hours == h
        if mask.sum() == 0:
            continue
        m = compute_metrics(y_true[mask], y_pred[mask])
        records.append({"hour": h, "rmse": m["rmse"], "mae": m["mae"],
                        "n_samples": m["n_samples"]})

    return pd.DataFrame(records)


def compute_dayperiod_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hours: np.ndarray,
) -> pd.DataFrame:
    """Compute RMSE/MAE per day period (凌晨/早高峰/白天平峰/晚高峰/夜间)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    hours = np.asarray(hours, dtype=int)

    records = []
    for period in ["凌晨", "早高峰", "白天平峰", "晚高峰", "夜间"]:
        mask = np.array([DAY_PERIOD_MAP.get(h, "Unknown") == period for h in hours])
        if mask.sum() == 0:
            continue
        m = compute_metrics(y_true[mask], y_pred[mask])
        records.append({"day_period": period, "rmse": m["rmse"], "mae": m["mae"],
                        "n_samples": m["n_samples"]})

    return pd.DataFrame(records)


def compute_high_pollution_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 150.0,
) -> dict[str, float]:
    """Compute metrics on the high-pollution subset (y_true > threshold)."""
    mask = y_true > threshold
    if mask.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "n_samples": 0,
                "threshold": threshold}
    m = compute_metrics(y_true[mask], y_pred[mask])
    m["threshold"] = threshold
    return m
