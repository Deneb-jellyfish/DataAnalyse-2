"""Train ARIMA for hourly PM2.5 prediction tasks (+1h and +24h).

Usage:
  python scripts/train_arima.py

Notes on practical configuration
--------------------------------
SARIMA with seasonal order s=24 is computationally intractable in
statsmodels for this dataset size: the Kalman-filter state dimension
blows up to 27-51, making a single likelihood evaluation O(n * dim^2).

We use a non-seasonal ARIMA(7,1,3) instead, which captures temporal
dependence through a higher AR/MA order while keeping state dim ~8
and fitting time under 60 s on 12000 training points.

The daily cycle is partially captured by the long AR lags (7 hours
of autoregressive memory), though not as explicitly as a seasonal term.

These design decisions are documented in the saved metrics.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.arima_model import ARIMAHourlyModel
from evaluation.metrics import compute_metrics
from utils.io import PROCESSED_DIR

OUTPUT_DIR = ROOT / "outputs" / "hourly"

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
MAX_TRAIN_LEN = 12000           # cap training series to last N hours
ARIMA_ORDER = (7, 1, 3)         # non-seasonal (p, d, q) -- higher order
ARIMA_SEASONAL = (0, 0, 0, 0)   # seasonal disabled (s=24 too slow)
# ---------------------------------------------------------------------------


def load_series() -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Load Beijing hourly PM2.5 time series.

    Returns
    -------
    pm25 : 1D float64 array
    dti  : DatetimeIndex aligned with pm25
    """
    df = pd.read_csv(PROCESSED_DIR / "beijing_hourly.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Extract pm25 and forward-fill any NaNs (ARIMA cannot handle NaNs)
    pm25 = df["pm25"].values.astype(np.float64)
    nan_mask = np.isnan(pm25)
    if nan_mask.any():
        print(f"  Found {nan_mask.sum()} NaN(s) in pm25; forward-filling.")
        pm25 = pd.Series(pm25).ffill().bfill().values.astype(np.float64)

    return pm25, df["datetime"]


def make_prediction_records(
    forecast_times: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    model_name: str = "ARIMA",
    city: str = "Beijing",
    split: str = "test",
) -> list[dict]:
    """Build list-of-dicts conforming to the unified predictions schema."""
    target_dt = forecast_times + pd.Timedelta(hours=horizon)
    records = []
    for i in range(len(forecast_times)):
        records.append({
            "forecast_origin_time": forecast_times[i].strftime("%Y-%m-%d %H:%M:%S"),
            "target_time": target_dt[i].strftime("%Y-%m-%d %H:%M:%S"),
            "horizon_hours": horizon,
            "city": city,
            "model": model_name,
            "y_true": y_true[i],
            "y_pred": y_pred[i],
            "split": split,
        })
    return records


def run_h1(
    model: ARIMAHourlyModel,
    test_pm25: np.ndarray,
    test_dti: pd.DatetimeIndex,
    eval_indices: np.ndarray,
) -> tuple[list[dict], dict]:
    """Evaluate +1h rolling one-step-ahead forecast.

    The model still receives *all* true values for state updates;
    predictions are only recorded at eval_indices.
    """
    print("=" * 60)
    print("ARIMA +1h  rolling one-step-ahead forecast")
    print(f"  Evaluating at {len(eval_indices)} / {len(test_pm25)} points")
    print("=" * 60)

    t0 = time.time()
    all_preds = model.predict_h1(test_pm25)
    elapsed = time.time() - t0
    print(f"  Predictions computed in {elapsed:.1f}s")

    # Subset to evaluation indices
    y_true = test_pm25[eval_indices]
    y_pred = all_preds[eval_indices]
    dti_eval = test_dti[eval_indices]

    # Keep only valid (non-NaN) pairs
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_v = y_true[valid]
    y_pred_v = y_pred[valid]
    dti_v = dti_eval[valid]

    if len(y_true_v) == 0:
        print("  WARNING: no valid prediction pairs after filtering.")
        return [], {}

    metrics = compute_metrics(y_true_v, y_pred_v)
    print(f"  RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, "
          f"MAPE={metrics['mape']:.1f}%, n={metrics['n_samples']}")

    records = make_prediction_records(dti_v, y_true_v, y_pred_v, horizon=1)

    return records, {
        "model": "ARIMA",
        "horizon_hours": 1,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mape": metrics["mape"],
        "n_train": model._train_length,
        "n_test_total": len(test_pm25),
        "n_evaluated": len(y_true_v),
        "train_time_s": round(elapsed, 1),
        "seasonal_order": str(model.seasonal_order),
        "order": str(model.order),
    }


def run_h24(
    model: ARIMAHourlyModel,
    test_pm25: np.ndarray,
    test_dti: pd.DatetimeIndex,
    eval_indices: np.ndarray,
) -> tuple[list[dict], dict]:
    """Evaluate +24h 24-step-ahead forecast.

    The last 23 test points cannot receive a +24h prediction (would
    require future data), so eval_indices beyond len-24 are skipped.
    """
    max_h24_idx = len(test_pm25) - 24
    eval_indices = eval_indices[eval_indices <= max_h24_idx]

    print("=" * 60)
    print("ARIMA +24h 24-step-ahead forecast")
    print(f"  Evaluating at {len(eval_indices)} / {len(test_pm25)} points")
    print(f"  (last 24 excluded from evaluation)")
    print("=" * 60)

    t0 = time.time()
    all_preds = model.predict_h24(test_pm25)
    elapsed = time.time() - t0
    print(f"  Predictions computed in {elapsed:.1f}s")

    # Subset to evaluation indices
    y_true = test_pm25[eval_indices]
    y_pred = all_preds[eval_indices]
    dti_eval = test_dti[eval_indices]

    # Keep only valid (non-NaN) pairs
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_v = y_true[valid]
    y_pred_v = y_pred[valid]
    dti_v = dti_eval[valid]

    if len(y_true_v) == 0:
        print("  WARNING: no valid prediction pairs after filtering.")
        return [], {}

    metrics = compute_metrics(y_true_v, y_pred_v)
    print(f"  RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, "
          f"MAPE={metrics['mape']:.1f}%, n={metrics['n_samples']}")

    records = make_prediction_records(dti_v, y_true_v, y_pred_v, horizon=24)

    return records, {
        "model": "ARIMA",
        "horizon_hours": 24,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mape": metrics["mape"],
        "n_train": model._train_length,
        "n_test_total": len(test_pm25),
        "n_evaluated": len(y_true_v),
        "train_time_s": round(elapsed, 1),
        "seasonal_order": str(model.seasonal_order),
        "order": str(model.order),
    }


def save_predictions(records: list[dict], path: Path) -> None:
    """Save prediction records to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "forecast_origin_time", "target_time", "horizon_hours",
        "city", "model", "y_true", "y_pred", "split",
    ]
    df = pd.DataFrame(records, columns=fieldnames)
    df.to_csv(path, index=False)
    print(f"  Saved: {path}  ({len(df)} rows)")


def save_metrics(rows: list[dict]) -> None:
    """Save combined metrics CSV."""
    path = OUTPUT_DIR / "arima_metrics.csv"
    df = pd.DataFrame(rows)
    cols = ["model", "horizon_hours", "rmse", "mae", "mape",
            "n_train", "n_test_total", "n_evaluated", "train_time_s",
            "order", "seasonal_order"]
    extras = [c for c in df.columns if c not in cols]
    df = df[cols + extras]
    df.to_csv(path, index=False)
    print(f"\nMetrics saved: {path}")


def main() -> None:
    print("Loading data ...")
    pm25, dti = load_series()
    print(f"  Total series length: {len(pm25)}")
    print(f"  Date range: {dti.iloc[0]}  ->  {dti.iloc[-1]}")

    # Chronological 80 / 20 split
    split_idx = int(len(pm25) * 0.8)
    train_pm25 = pm25[:split_idx]
    test_pm25 = pm25[split_idx:]
    test_dti = dti[split_idx:].reset_index(drop=True)

    # Cap training series to keep ARIMA fitting tractable
    if len(train_pm25) > MAX_TRAIN_LEN:
        train_pm25 = train_pm25[-MAX_TRAIN_LEN:]
        print(f"  Train capped to last {MAX_TRAIN_LEN} hours")
    print(f"  Train: {len(train_pm25)}  (ending {dti.iloc[split_idx - 1]})")
    print(f"  Test:  {len(test_pm25)}  ({dti.iloc[split_idx]} -> {dti.iloc[-1]})")

    # ------------------------------------------------------------------
    # Fit ARIMA on train
    # ------------------------------------------------------------------
    print("\nFitting ARIMA on training series ...")
    print(f"  order={ARIMA_ORDER}, seasonal_order={ARIMA_SEASONAL}")
    print(f"  (non-seasonal: seasonal SARIMA s=24 intractable on this data size)")
    t_fit = time.time()
    model = ARIMAHourlyModel(order=ARIMA_ORDER, seasonal_order=ARIMA_SEASONAL)
    fit_meta = model.fit(train_pm25)
    fit_time = time.time() - t_fit
    print(f"  Fitted in {fit_time:.1f}s")
    print(f"  AIC={fit_meta['aic']:.1f}, BIC={fit_meta['bic']:.1f}")

    # ------------------------------------------------------------------
    # Determine evaluation sampling
    # ------------------------------------------------------------------
    n_test = len(test_pm25)

    if n_test > 5000:
        # Sample every 6th point to keep runtime manageable
        sample_step = 6
        eval_indices = np.arange(0, n_test, sample_step)
        print(f"\n  Test set has {n_test} points -> evaluating every "
              f"{sample_step}th point ({len(eval_indices)} points)")
    else:
        eval_indices = np.arange(n_test)
        print(f"\n  Evaluating all {n_test} test points")

    # ------------------------------------------------------------------
    # Run +1h and +24h
    # ------------------------------------------------------------------
    all_rows: list[dict] = []

    # +1h
    recs_h1, metrics_h1 = run_h1(model, test_pm25, test_dti, eval_indices)
    if recs_h1:
        save_predictions(recs_h1, OUTPUT_DIR / "arima_predictions_h1.csv")
    all_rows.append(metrics_h1)

    # Re-fit ARIMA on train for +24h (fresh state avoids contamination from
    # the +1h rolling updates)
    print("\nRe-fitting ARIMA for +24h (fresh state) ...")
    model_h24 = ARIMAHourlyModel(order=ARIMA_ORDER, seasonal_order=ARIMA_SEASONAL)
    model_h24.fit(train_pm25)

    # +24h
    recs_h24, metrics_h24 = run_h24(model_h24, test_pm25, test_dti, eval_indices)
    if recs_h24:
        save_predictions(recs_h24, OUTPUT_DIR / "arima_predictions_h24.csv")
    all_rows.append(metrics_h24)

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    save_metrics(all_rows)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ARIMA Summary")
    print("=" * 60)
    for r in all_rows:
        print(f"  +{r['horizon_hours']}h: RMSE={r['rmse']:.2f}, "
              f"MAE={r['mae']:.2f}, MAPE={r['mape']:.1f}%, "
              f"n_eval={r['n_evaluated']}")


if __name__ == "__main__":
    main()
