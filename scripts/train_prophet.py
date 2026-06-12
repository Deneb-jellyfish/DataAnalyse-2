"""Train Prophet for hourly PM2.5 prediction tasks (+1h and +24h).

Usage:
  python scripts/train_prophet.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.prophet_model import ProphetHourlyModel
from evaluation.metrics import compute_metrics
from utils.io import PROCESSED_DIR

OUTPUT_DIR = ROOT / "outputs" / "hourly"


def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load beijing_hourly.csv and split chronologically 80/20.

    Returns (full_df, train_df, test_df) each with ds, y columns.
    """
    df = pd.read_csv(PROCESSED_DIR / "beijing_hourly.csv")
    df["ds"] = pd.to_datetime(df["datetime"])
    df["y"] = df["pm25"]
    df = df.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)

    print(f"Total valid hourly records: {len(df)}")
    print(f"  Range: {df['ds'].iloc[0]} to {df['ds'].iloc[-1]}")

    # Chronological 80/20 split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx][["ds", "y"]].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"  Train: {len(train_df)} ({train_df['ds'].iloc[0]} to {train_df['ds'].iloc[-1]})")
    print(f"  Test:  {len(test_df)} ({test_df['ds'].iloc[0]} to {test_df['ds'].iloc[-1]})")

    return df, train_df, test_df


def downsample_train(train_df: pd.DataFrame, step: int = 3) -> pd.DataFrame:
    """Down-sample training data to every `step`-th row to speed up Prophet."""
    ds = train_df.iloc[::step].reset_index(drop=True)
    print(f"  Down-sampled train from {len(train_df)} to {len(ds)} (every {step}h)")
    return ds


def build_target_lookup(full_df: pd.DataFrame) -> dict:
    """Build a dict mapping datetime -> pm25 value."""
    # Use the latest value per timestamp (no duplicates expected)
    return dict(zip(full_df["ds"], full_df["y"]))


def make_prediction_csv(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    target_lookup: dict,
    horizon: int,
    model_name: str,
    city: str = "Beijing",
    split: str = "test",
) -> pd.DataFrame:
    """Build unified-format predictions DataFrame.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test origin rows with columns ds, y (among others).
    y_pred : np.ndarray
        Predicted PM2.5 values for each target_time.
    target_lookup : dict
        Mapping from pd.Timestamp -> actual PM2.5 value.
    horizon : int
        Prediction horizon in hours.
    model_name : str
        Model identifier string.
    """
    origin_times = test_df["ds"].values
    target_times = pd.to_datetime(origin_times) + pd.Timedelta(hours=horizon)

    # Look up y_true for each target_time
    y_true_values = np.array([
        target_lookup.get(tt, np.nan) for tt in target_times
    ], dtype=np.float64)

    # Filter out NaN targets (e.g., if target_time is beyond available data)
    valid = ~np.isnan(y_true_values)
    if not valid.all():
        n_dropped = (~valid).sum()
        print(f"  Dropped {n_dropped} predictions where target y_true is NaN (beyond data range)")
        target_times = target_times[valid]
        y_true_values = y_true_values[valid]
        y_pred = y_pred[valid]
        origin_times = origin_times[valid]

    return pd.DataFrame({
        "forecast_origin_time": pd.to_datetime(origin_times).strftime("%Y-%m-%d %H:%M:%S"),
        "target_time": target_times.strftime("%Y-%m-%d %H:%M:%S"),
        "horizon_hours": horizon,
        "city": city,
        "model": model_name,
        "y_true": y_true_values,
        "y_pred": y_pred,
        "split": split,
    })


def run_horizon(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_lookup: dict,
    horizon: int,
    model_name: str,
) -> dict:
    """Fit Prophet and evaluate for a given horizon."""
    print()
    print("=" * 60)
    print(f"Prophet +{horizon}h")
    print("=" * 60)

    # Down-sample training if very large
    if len(train_df) > 20000:
        print("  Large training set detected; down-sampling...")
        fit_train = downsample_train(train_df, step=3)
    else:
        fit_train = train_df

    # Fit
    print(f"  Fitting Prophet on {len(fit_train)} rows...")
    t0 = time.time()
    model = ProphetHourlyModel()
    meta = model.fit(fit_train)
    fit_time = time.time() - t0
    meta["fit_time_s"] = round(fit_time, 1)
    print(f"  Model: {meta['model']}, fit time: {meta['fit_time_s']:.1f}s")
    if meta.get("fallback_reason"):
        print(f"  Fallback reason: {meta['fallback_reason']}")

    # Build target times for prediction
    origin_times = test_df["ds"].values
    target_times = pd.to_datetime(origin_times) + pd.Timedelta(hours=horizon)

    print(f"  Predicting {len(target_times)} target times...")
    t0 = time.time()
    y_pred = model.predict(target_times)
    pred_time = time.time() - t0
    print(f"  Prediction time: {pred_time:.1f}s")

    # Build prediction CSV
    pred_df = make_prediction_csv(
        test_df, y_pred, target_lookup, horizon=horizon, model_name=model_name
    )

    # Compute metrics
    metrics = compute_metrics(
        pred_df["y_true"].values, pred_df["y_pred"].values
    )
    print(f"  Test RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, "
          f"MAPE={metrics['mape']:.1f}%, n={metrics['n_samples']}")

    # Save predictions
    pred_path = OUTPUT_DIR / f"prophet_predictions_h{horizon}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"  Saved: {pred_path}")

    return {
        "model": model_name,
        "horizon_hours": horizon,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mape": metrics["mape"],
        "n_train": len(fit_train),
        "n_test": metrics["n_samples"],
        "fit_time_s": meta["fit_time_s"],
        "pred_time_s": round(pred_time, 1),
        "fallback_reason": meta.get("fallback_reason", ""),
    }


def save_metrics(rows: list[dict]) -> None:
    """Save combined metrics CSV."""
    path = OUTPUT_DIR / "prophet_metrics.csv"
    df = pd.DataFrame(rows)
    # Ensure notes column exists
    if "notes" not in df.columns:
        df["notes"] = ""
    cols = ["model", "horizon_hours", "rmse", "mae", "mape", "n_train",
            "n_test", "fit_time_s", "pred_time_s", "fallback_reason", "notes"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]
    df.to_csv(path, index=False)
    print(f"\nMetrics saved: {path}")


def main() -> None:
    print("=" * 60)
    print("Prophet Hourly PM2.5 Training")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    full_df, train_df, test_df = load_and_split()

    # 2. Build target lookup for y_true retrieval
    target_lookup = build_target_lookup(full_df)

    all_rows = []

    # 3. Run +1h
    r1 = run_horizon(full_df, train_df, test_df, target_lookup,
                     horizon=1, model_name="Prophet")
    all_rows.append(r1)

    # 4. Run +24h
    r24 = run_horizon(full_df, train_df, test_df, target_lookup,
                      horizon=24, model_name="Prophet")
    all_rows.append(r24)

    # 5. Save metrics
    save_metrics(all_rows)

    # 6. Summary
    print()
    print("=" * 60)
    print("Prophet Summary")
    print("=" * 60)
    for r in all_rows:
        print(f"  +{r['horizon_hours']}h: RMSE={r['rmse']:.2f}, "
              f"MAE={r['mae']:.2f}, MAPE={r['mape']:.1f}% "
              f"(model: {r['model']}, fit: {r['fit_time_s']}s)")


if __name__ == "__main__":
    main()
