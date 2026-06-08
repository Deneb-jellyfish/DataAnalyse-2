"""Unified model evaluation (B-7).

Computes MAE / RMSE / MAPE for every model and also breaks down
performance by AQI level.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# AQI level classification (China standard, 24-h PM2.5 μg/m³)
# ---------------------------------------------------------------------------
def aqi_level(pm25: float) -> str:
    if pm25 <= 35:
        return "优 (Good)"
    if pm25 <= 75:
        return "良 (Moderate)"
    if pm25 <= 115:
        return "轻度污染 (Unhealthy-S)"
    if pm25 <= 150:
        return "中度污染 (Unhealthy)"
    return "重度污染 (Very Unhealthy)"


# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> dict[str, float]:
    residuals = y_true - y_pred
    ape = np.abs(residuals / np.maximum(y_true, 1e-6)) * 100
    return {
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "mae": float(np.mean(np.abs(residuals))),
        "mape": float(np.mean(np.minimum(ape, 200.0))),
        "n": int(len(y_true)),
    }


def evaluate_from_predictions(
    pred_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Produce overall + per-AQI-level metrics from prediction records.

    ``pred_records`` is a list of dicts like:
        {"date": ..., "city": ..., "model": ..., "y_true": ..., "y_pred": ...}
    """
    df = pd.DataFrame(pred_records)
    model_names = sorted(df["model"].unique())

    # ---- overall ----
    overall: dict[str, dict[str, float]] = {}
    for m in model_names:
        sub = df[df["model"] == m]
        overall[m] = compute_metrics(sub["y_true"].values, sub["y_pred"].values)

    # ---- by AQI level ----
    df["level"] = df["y_true"].apply(aqi_level)
    levels = ["优 (Good)", "良 (Moderate)", "轻度污染 (Unhealthy-S)",
              "中度污染 (Unhealthy)", "重度污染 (Very Unhealthy)"]
    stratified: dict[str, dict[str, dict[str, float]]] = {}
    for m in model_names:
        stratified[m] = {}
        for lv in levels:
            sub = df[(df["model"] == m) & (df["level"] == lv)]
            if len(sub) >= 3:
                stratified[m][lv] = compute_metrics(sub["y_true"].values,
                                                     sub["y_pred"].values)

    # ---- summary table ----
    rows = []
    for m in model_names:
        rows.append({
            "Model": m,
            "RMSE": f"{overall[m]['rmse']:.2f}",
            "MAE": f"{overall[m]['mae']:.2f}",
            "MAPE(%)": f"{overall[m]['mape']:.1f}",
            "N": overall[m]["n"],
        })

    return {
        "overall": overall,
        "by_aqi_level": stratified,
        "summary_table": rows,
        "model_names": model_names,
    }


# ---------------------------------------------------------------------------
# load all predictions from model runs
# ---------------------------------------------------------------------------
def collect_all_predictions() -> list[dict[str, Any]]:
    """Run all model scripts and collect their prediction dicts."""
    import importlib

    all_rows: list[dict[str, Any]] = []
    model_modules = [
        ("models.arima_model", "ARIMA"),
        ("models.prophet_model", "Prophet"),
        ("models.xgboost_model", "XGBoost"),
        ("models.lstm_model", "LSTM"),
        ("models.transformer_model", "Informer"),
        ("models.transformer_standard", "Transformer"),
    ]

    for mod_name, model_name in model_modules:
        try:
            mod = importlib.import_module(mod_name)
            result = mod.run()
            rows = result.get("predictions", [])
            print(f"  {model_name}: {len(rows)} predictions, "
                  f"RMSE={result.get('rmse','?'):.2f}")
            all_rows.extend(rows)
        except Exception as exc:
            print(f"  {model_name}: FAILED — {exc}")

    return all_rows


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("Collecting predictions from all models ...")
    preds = collect_all_predictions()
    print(f"\nTotal: {len(preds)} prediction rows from {len(set(r['model'] for r in preds))} models")

    result = evaluate_from_predictions(preds)

    print("\n" + "=" * 65)
    print(f"{'Model':<16} {'RMSE':>8} {'MAE':>8} {'MAPE':>8}  {'N':>5}")
    print("-" * 65)
    for row in result["summary_table"]:
        print(f"{row['Model']:<16} {row['RMSE']:>8} {row['MAE']:>8} "
              f"{row['MAPE(%)']:>7}% {row['N']:>5}")

    # stratified table
    print("\n--- By AQI Level ---")
    for level in ["优 (Good)", "良 (Moderate)", "轻度污染 (Unhealthy-S)",
                   "中度污染 (Unhealthy)", "重度污染 (Very Unhealthy)"]:
        print(f"\n  [{level}]")
        for m in result["model_names"]:
            if level in result["by_aqi_level"].get(m, {}):
                s = result["by_aqi_level"][m][level]
                print(f"    {m:<14} RMSE={s['rmse']:6.2f}  MAE={s['mae']:6.2f}  "
                      f"MAPE={s['mape']:5.1f}%  n={s['n']}")
