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
from utils.hourly_output_paths import artifact_path, ensure_hourly_output_dirs, resolve_artifact_path

OUTPUT_DIR = ROOT / "outputs" / "hourly"
BEIJING_PATH = ROOT / "data" / "processed" / "beijing_hourly.csv"
SHANGHAI_PATH = ROOT / "data" / "processed" / "aligned_hourly" / "shanghai.csv"


def make_xy(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    """Construct direct horizon features and targets."""
    frame, features = build_hourly_feature_frame(df)
    target_col = f"pm25_target_h{horizon}"
    frame[target_col] = frame["pm25"].shift(-horizon)
    valid = frame.dropna(subset=features + [target_col]).reset_index(drop=True)
    return valid, features


def run_horizon(horizon: int, task: str, params: dict | None) -> tuple[pd.DataFrame, dict]:
    """Train on Beijing and test on Shanghai for one direct horizon."""
    beijing = pd.read_csv(BEIJING_PATH)
    shanghai = pd.read_csv(SHANGHAI_PATH)

    bj_valid, features = make_xy(beijing, horizon=horizon)
    sh_valid, _ = make_xy(shanghai, horizon=horizon)

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

    model = XGBoostHourlyModel(params=params)
    model.fit(X_tr, y_tr, X_va, y_va, feature_names=features)
    y_pred = model.predict(X_test)

    pred_df = pd.DataFrame(
        {
            "forecast_origin_time": pd.to_datetime(sh_valid["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "target_time": (pd.to_datetime(sh_valid["datetime"]) + pd.Timedelta(hours=horizon)).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "task": task,
            "horizon_hours": horizon,
            "city": "Shanghai",
            "model": "XGBoost",
            "y_true": y_test,
            "y_pred": y_pred,
            "split": "cross_city_test",
        }
    )

    metrics = compute_metrics(y_test, y_pred)
    row = {
        "model": "XGBoost",
        "task": task,
        "horizon_hours": horizon,
        "train_city": "Beijing",
        "test_city": "Shanghai",
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mape": metrics["mape"],
        "n_samples": int(metrics["n_samples"]),
        "notes": "trained_on_beijing_tested_on_shanghai",
    }
    return pred_df, row


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_hourly_output_dirs()

    pred_frames: list[pd.DataFrame] = []
    metric_rows: list[dict] = []

    h1_params_path = resolve_artifact_path("xgboost_h1_best_params.json")
    h1_params = json.loads(h1_params_path.read_text(encoding="utf-8")) if h1_params_path.exists() else None
    pred_df, metrics = run_horizon(1, task="h1", params=h1_params)
    pred_frames.append(pred_df)
    metric_rows.append(metrics)

    seq6_params_path = resolve_artifact_path("xgboost_seq6_best_params.json")
    seq6_params_blob = json.loads(seq6_params_path.read_text(encoding="utf-8")) if seq6_params_path.exists() else {}
    for horizon in range(1, 7):
        pred_df, metrics = run_horizon(horizon, task="seq6", params=seq6_params_blob.get(f"h{horizon}"))
        pred_frames.append(pred_df)
        metric_rows.append(metrics)

    pd.concat(pred_frames, ignore_index=True).to_csv(artifact_path("cross_city_predictions.csv", ensure_parent=True), index=False)
    pd.DataFrame(metric_rows).to_csv(artifact_path("cross_city_metrics.csv", ensure_parent=True), index=False)
    print("Generated cross_city_predictions.csv and cross_city_metrics.csv")


if __name__ == "__main__":
    main()
