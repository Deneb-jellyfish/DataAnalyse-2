"""Prophet model for Beijing PM2.5 daily prediction (B-2).

Prophet decomposes the series into trend, seasonality, and holiday
effects.  For a fair comparison with other models the train/test split
is identical — 80 % chronological train, 20 % test.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io import PROCESSED_DIR

log = logging.getLogger(__name__)

TEST_RATIO = 0.2


# ---------------------------------------------------------------------------
def _load_df(path: Path | None = None) -> pd.DataFrame:
    path = path or (PROCESSED_DIR / "beijing" / "beijing_daily_city.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df.rename(columns={"date": "ds", "PM2.5": "y"})


def _holiday_frame() -> pd.DataFrame:
    """Build a Prophet-compatible Chinese-holiday DataFrame."""
    try:
        import chinese_calendar as cc  # type: ignore[import-untyped]

        start = pd.Timestamp("2013-01-01")
        end = pd.Timestamp("2017-12-31")
        dates = pd.date_range(start, end, freq="D")
        rows = []
        for d in dates:
            if cc.is_holiday(d.date()):
                rows.append({"holiday": "cn_holiday", "ds": d})
        return pd.DataFrame(rows)
    except Exception:
        log.warning("Cannot build holiday calendar — proceeding without holidays")
        return pd.DataFrame(columns=["holiday", "ds"])


def run() -> dict[str, Any]:
    """Full Prophet pipeline."""
    df = _load_df()
    split_idx = int(len(df) * (1 - TEST_RATIO))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    holidays = _holiday_frame()
    log.info("Prophet: %d train days, %d test days, %d holidays",
             len(train_df), len(test_df), len(holidays))

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=5.0,
        holidays_prior_scale=5.0,
    )
    model.add_country_holidays(country_name="CN")
    model.fit(train_df[["ds", "y"]])

    future = model.make_future_dataframe(periods=len(test_df), include_history=False)
    forecast = model.predict(future)

    y_true = test_df["y"].values.astype(np.float64)
    y_pred = forecast["yhat"].values[-len(test_df):].astype(np.float64)

    # --- metrics ---
    residuals = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    ape = np.abs(residuals / np.maximum(y_true, 1e-6)) * 100
    mape = float(np.mean(np.minimum(ape, 200.0)))

    # --- predictions ---
    test_dates = test_df["ds"].values
    rows: list[dict[str, Any]] = []
    for d, t, p in zip(test_dates, y_true, y_pred):
        rows.append({
            "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
            "city": "Beijing",
            "model": "Prophet",
            "y_true": round(float(t), 4),
            "y_pred": round(max(0.0, float(p)), 4),
        })

    log.info("Prophet  RMSE=%.2f  MAE=%.2f  MAPE=%.1f%%", rmse, mae, mape)
    return {
        "model": "Prophet",
        "rmse": rmse, "mae": mae, "mape": mape,
        "n_train": int(len(train_df)), "n_test": int(len(test_df)),
        "predictions": rows,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run()
    print(f"\n{'='*50}")
    print(
        f"Prophet | RMSE={result['rmse']:.2f}  MAE={result['mae']:.2f}  "
        f"MAPE={result['mape']:.1f}%  (n={result['n_test']})"
    )
