"""ARIMA model for Beijing PM2.5 daily prediction (B-1).

Approach:
1. ``auto_arima`` on the training set determines the best (p,d,q)(P,D,Q,m)
   order **once**.
2. Test-set evaluation uses true 1-step-ahead rolling forecasts via
   :meth:`pmdarima.arima.ARIMA.update` — the model ingests each new
   observation as it arrives without a full refit.  This is the correct
   way to evaluate a time-series model without data leakage.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pmdarima.arima import ARIMA
from pmdarima import auto_arima

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io import PROCESSED_DIR

log = logging.getLogger(__name__)

TEST_RATIO = 0.2


# ---------------------------------------------------------------------------
def load_daily_series(path: Path | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Load Beijing daily PM2.5, sorted by date."""
    path = path or (PROCESSED_DIR / "beijing" / "beijing_daily_city.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df, df["PM2.5"].values.astype(np.float64)


def _split(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = int(len(series) * (1 - TEST_RATIO))
    return series[:idx].copy(), series[idx:].copy()


def _search_order(
    train: np.ndarray, seasonal: bool = True, m: int = 7
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    """Run auto_arima ONCE to select the best order."""
    mdl = auto_arima(
        train, start_p=1, start_q=1, max_p=5, max_q=5,
        max_P=2, max_Q=2, seasonal=seasonal, m=m,
        trace=False, error_action="ignore", suppress_warnings=True,
        stepwise=True, n_jobs=1,
    )
    log.info("  order=%s seasonal=%s  AIC=%.1f",
             mdl.order, mdl.seasonal_order, mdl.aic())
    return mdl.order, mdl.seasonal_order


def rolling_forecast(
    train: np.ndarray,
    test: np.ndarray,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> np.ndarray:
    """True 1-step-ahead forecasts via incremental ``update``.

    The model is fit once on the training series.  For each test point we
    predict one step ahead, then call ``model.update(true_value)`` to
    incorporate the observation — no data leakage, no periodic re-search.
    """
    model = ARIMA(order=order, seasonal_order=seasonal_order,
                  suppress_warnings=True)
    model.fit(train)

    preds = []
    for true_val in test:
        fc = model.predict(n_periods=1)
        preds.append(float(fc[0]))
        model.update(true_val)

    return np.array(preds, dtype=np.float64)


# ---------------------------------------------------------------------------
def run() -> dict[str, Any]:
    """Full ARIMA pipeline — returns metrics and predictions."""
    df, series = load_daily_series()
    train, test = _split(series)
    split_idx = len(train)

    log.info("searching ARIMA order on %d train points ...", len(train))
    order, seasonal_order = _search_order(train)
    log.info("rolling forecast on %d test points ...", len(test))
    preds = rolling_forecast(train, test, order, seasonal_order)

    # --- metrics ---
    residuals = test - preds
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    ape = np.abs(residuals / np.maximum(test, 1e-6)) * 100
    mape = float(np.mean(np.minimum(ape, 200.0)))

    # --- predictions DataFrame ---
    test_dates = df["date"].iloc[split_idx:].values
    rows: list[dict[str, Any]] = []
    for d, t, p in zip(test_dates, test, preds):
        rows.append({
            "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
            "city": "Beijing",
            "model": "ARIMA",
            "y_true": round(float(t), 4),
            "y_pred": round(max(0.0, float(p)), 4),
        })

    log.info("ARIMA  RMSE=%.2f  MAE=%.2f  MAPE=%.1f%%", rmse, mae, mape)
    return {
        "model": "ARIMA",
        "order": order,
        "seasonal_order": seasonal_order,
        "rmse": rmse, "mae": mae, "mape": mape,
        "n_train": int(len(train)), "n_test": int(len(test)),
        "predictions": rows,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run()
    print(f"\n{'='*50}")
    print(
        f"ARIMA | RMSE={result['rmse']:.2f}  MAE={result['mae']:.2f}  "
        f"MAPE={result['mape']:.1f}%  (n={result['n_test']})"
    )
