"""Cross-city generalization for XGBoost (train Beijing, test Shanghai)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.metrics import compute_metrics
from models.xgboost_model import XGBoostHourlyModel
from utils.feature_engineering import StandardScaler, build_hourly_feature_frame

OUTPUT_DIR = ROOT / "outputs" / "hourly"
BEIJING_PATH = ROOT / "data" / "processed" / "beijing_hourly.csv"
SHANGHAI_PATH = ROOT / "data" / "processed" / "aligned_hourly" / "shanghai.csv"


def _make_xy(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    frame, features = build_hourly_feature_frame(df)
    target_col = f"pm25_target_h{horizon}"
    frame[target_col] = frame["pm25"].shift(-horizon)
    valid = frame.dropna(subset=features + [target_col]).reset_index(drop=True)
    return valid, features


def _run_task(horizon: int) -> tuple[pd.DataFrame, dict]:
    bj = pd.read_csv(BEIJING_PATH)
    sh = pd.read_csv(SHANGHAI_PATH)

    bj_valid, features = _make_xy(bj, horizon=horizon)
    sh_valid, _ = _make_xy(sh, horizon=horizon)

    split_idx = int(len(bj_valid) * 0.8)
    bj_train = bj_valid.iloc[:split_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(bj_train[features].values.astype(np.float64))
    y_train = bj_train[f"pm25_target_h{horizon}"].values.astype(np.float64)

    X_test = scaler.transform(sh_valid[features].values.astype(np.float64))
    y_test = sh_valid[f"pm25_target_h{horizon}"].values.astype(np.float64)

    val_split = int(len(X_train) * 0.8)
    X_tr, X_va = X_train[:val_split], X_train[val_split:]
    y_tr, y_va = y_train[:val_split], y_train[val_split:]

    params_path = OUTPUT_DIR / f"xgboost_h{horizon}_best_params.json"
    params = json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else None

    model = XGBoostHourlyModel(params=params)
    model.fit(X_tr, y_tr, X_va, y_va, feature_names=features)
    y_pred = model.predict(X_test)

    pred_df = pd.DataFrame(
        {
            "forecast_origin_time": pd.to_datetime(sh_valid["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "target_time": (pd.to_datetime(sh_valid["datetime"]) + pd.Timedelta(hours=horizon)).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "horizon_hours": horizon,
            "city": "Shanghai",
            "model": "XGBoost",
            "y_true": y_test,
            "y_pred": y_pred,
            "split": "cross_city_test",
        }
    )

    m = compute_metrics(y_test, y_pred)
    metrics = {
        "model": "XGBoost",
        "horizon_hours": horizon,
        "train_city": "Beijing",
        "test_city": "Shanghai",
        "rmse": m["rmse"],
        "mae": m["mae"],
        "mape": m["mape"],
        "n_samples": int(m["n_samples"]),
        "notes": "trained_on_beijing_tested_on_shanghai",
    }
    return pred_df, metrics


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pred_all = []
    metric_rows = []
    for h in [1, 12]:
        pred_df, m = _run_task(h)
        pred_all.append(pred_df)
        metric_rows.append(m)

    pd.concat(pred_all, ignore_index=True).to_csv(OUTPUT_DIR / "cross_city_predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(OUTPUT_DIR / "cross_city_metrics.csv", index=False)
    print("Generated cross_city_predictions.csv and cross_city_metrics.csv")


if __name__ == "__main__":
    main()
