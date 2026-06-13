"""Train Prophet for hourly PM2.5 prediction tasks (+1h and +12h).

Important:
- Must use REAL Prophet when explicitly enabled.
- Default behavior is skip (write blocked status) to avoid accidental runs.
"""

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

from evaluation.metrics import compute_metrics
from utils.io import PROCESSED_DIR

OUTPUT_DIR = ROOT / "outputs" / "hourly"


def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(PROCESSED_DIR / "beijing_hourly.csv")
    df["ds"] = pd.to_datetime(df["datetime"])
    df["y"] = df["pm25"].astype(float)
    df = df.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return df, train_df, test_df


def make_prediction_csv(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    target_lookup: dict,
    horizon: int,
    model_name: str,
) -> pd.DataFrame:
    origin_times = test_df["ds"].values
    target_times = pd.to_datetime(origin_times) + pd.Timedelta(hours=int(horizon))
    y_true_values = np.array([target_lookup.get(tt, np.nan) for tt in target_times], dtype=np.float64)

    valid = ~np.isnan(y_true_values)
    origin_times = origin_times[valid]
    target_times = target_times[valid]
    y_true_values = y_true_values[valid]
    y_pred = y_pred[valid]

    return pd.DataFrame(
        {
            "forecast_origin_time": pd.to_datetime(origin_times).strftime("%Y-%m-%d %H:%M:%S"),
            "target_time": target_times.strftime("%Y-%m-%d %H:%M:%S"),
            "horizon_hours": int(horizon),
            "city": "Beijing",
            "model": model_name,
            "y_true": y_true_values,
            "y_pred": y_pred,
            "split": "test",
        }
    )


def run_horizon(train_df: pd.DataFrame, test_df: pd.DataFrame, target_lookup: dict, horizon: int) -> dict:
    from prophet import Prophet  # real prophet required

    prophet_train = train_df[["ds", "y"]].copy()

    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    t0 = time.time()
    model.fit(prophet_train)
    fit_time = time.time() - t0

    target_times = pd.to_datetime(test_df["ds"]) + pd.Timedelta(hours=int(horizon))
    future_df = pd.DataFrame({"ds": target_times})

    t1 = time.time()
    forecast = model.predict(future_df)
    pred_time = time.time() - t1
    y_pred = forecast["yhat"].values.astype(np.float64)

    pred_df = make_prediction_csv(test_df, y_pred, target_lookup, horizon=horizon, model_name="Prophet")
    pred_df.to_csv(OUTPUT_DIR / f"prophet_predictions_h{horizon}.csv", index=False)

    metrics = compute_metrics(pred_df["y_true"].values, pred_df["y_pred"].values)

    config = {
        "task": f"h{horizon}",
        "horizon_hours": int(horizon),
        "model": "Prophet",
        "daily_seasonality": True,
        "weekly_seasonality": True,
        "yearly_seasonality": True,
        "fit_time_s": round(fit_time, 1),
        "pred_time_s": round(pred_time, 1),
        "status": "done",
    }
    with open(OUTPUT_DIR / f"prophet_config_h{horizon}.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return {
        "model": "Prophet",
        "horizon_hours": int(horizon),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mape": metrics["mape"],
        "n_samples": int(metrics["n_samples"]),
        "notes": f"fit_time_s={fit_time:.1f},pred_time_s={pred_time:.1f}",
    }


def save_metrics(rows: list[dict]) -> None:
    path = OUTPUT_DIR / "prophet_metrics.csv"
    df = pd.DataFrame(rows)
    cols = ["model", "horizon_hours", "rmse", "mae", "mape", "n_samples", "notes"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]
    df.to_csv(path, index=False)


def write_blocked_outputs(reason: str) -> None:
    rows = []
    for horizon in [1, 12]:
        rows.append(
            {
                "model": "Prophet",
                "horizon_hours": horizon,
                "rmse": np.nan,
                "mae": np.nan,
                "mape": np.nan,
                "n_samples": 0,
                "notes": f"blocked:{reason}",
            }
        )
        cfg = {
            "task": f"h{horizon}",
            "horizon_hours": horizon,
            "status": "blocked",
            "reason": reason,
        }
        with open(OUTPUT_DIR / f"prophet_config_h{horizon}.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    save_metrics(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Prophet for hourly h1/h12 tasks")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually run Prophet training (default: skip and write blocked outputs)",
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
    except Exception as e:
        write_blocked_outputs(reason=f"prophet_import_failed:{e.__class__.__name__}")
        print("Prophet blocked: prophet package is unavailable in current environment.")
        return

    full_df, train_df, test_df = load_and_split()
    target_lookup = dict(zip(full_df["ds"], full_df["y"]))

    rows = [
        run_horizon(train_df, test_df, target_lookup, horizon=1),
        run_horizon(train_df, test_df, target_lookup, horizon=12),
    ]
    save_metrics(rows)


if __name__ == "__main__":
    main()
