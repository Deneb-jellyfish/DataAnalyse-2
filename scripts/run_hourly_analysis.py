"""Run hourly analysis modules required by hourly_experiment_plan2.md.

Outputs:
- outputs/hourly/feature_ablation.csv
- outputs/hourly/aqi_bucket_metrics.csv
- outputs/hourly/high_pollution_error_analysis.csv
- outputs/hourly/hourly_error_by_hour.csv
- outputs/hourly/hourly_error_by_dayperiod.csv
"""

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
from utils.io import PROCESSED_DIR

OUTPUT_DIR = ROOT / "outputs" / "hourly"


def _load_data(task: str) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(PROCESSED_DIR / "beijing_hourly.csv")
    frame, all_features = build_hourly_feature_frame(df)

    if task == "h1":
        target_col = "pm25_target_h1"
        horizon = 1
    elif task == "h12":
        target_col = "pm25_target_h12"
        horizon = 12
    else:
        raise ValueError(f"Unsupported task: {task}")

    frame[target_col] = frame["pm25"].shift(-horizon)
    valid = frame.dropna(subset=all_features + [target_col]).reset_index(drop=True)

    split_idx = int(len(valid) * 0.8)
    train = valid.iloc[:split_idx]
    test = valid.iloc[split_idx:]

    return valid, all_features, train[all_features].values, test[all_features].values, train[target_col].values.astype(np.float64), test[target_col].values.astype(np.float64)


def _feature_groups(all_features: list[str]) -> dict[str, list[str]]:
    has = set(all_features)

    def keep(names: list[str]) -> list[str]:
        return [n for n in names if n in has]

    base_short_lag = keep([
        "pm25_lag1", "pm25_lag3", "temp", "pres", "dewp", "humidity", "wind_speed", "precipitation",
    ])
    base_plus_daily_cycle = list(dict.fromkeys(base_short_lag + keep(["hour_sin", "hour_cos"])))
    mid_range_memory = list(dict.fromkeys(base_plus_daily_cycle + keep(["pm25_lag6", "pm25_lag12", "pm25_lag18"])))
    trend_and_roll = list(
        dict.fromkeys(
            mid_range_memory
            + keep([
                "pm25_roll_mean_6", "pm25_roll_mean_12", "pm25_roll_mean_24",
                "pm25_diff_1", "pm25_diff_3", "pm25_diff_6", "pm25_diff_12",
            ])
        )
    )

    return {
        "base_short_lag": base_short_lag,
        "base_plus_daily_cycle": base_plus_daily_cycle,
        "mid_range_memory": mid_range_memory,
        "trend_and_roll": trend_and_roll,
        "full_48": list(all_features),
    }


def run_feature_ablation() -> None:
    rows: list[dict] = []

    for task in ["h1", "h12"]:
        _, all_features, X_train_raw, X_test_raw, y_train, y_test = _load_data(task)
        groups = _feature_groups(all_features)

        params_path = OUTPUT_DIR / f"xgboost_{task}_best_params.json"
        params = json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else None

        for group_name, group_features in groups.items():
            idx = [all_features.index(c) for c in group_features]
            X_train_sel = X_train_raw[:, idx]
            X_test_sel = X_test_raw[:, idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_sel)
            X_test = scaler.transform(X_test_sel)

            val_split = int(len(X_train) * 0.8)
            X_tr, X_va = X_train[:val_split], X_train[val_split:]
            y_tr, y_va = y_train[:val_split], y_train[val_split:]

            model = XGBoostHourlyModel(params=params)
            model.fit(X_tr, y_tr, X_va, y_va, feature_names=group_features)
            y_pred = model.predict(X_test)
            m = compute_metrics(y_test, y_pred)

            rows.append(
                {
                    "model": "XGBoost",
                    "task": task,
                    "horizon_hours": 1 if task == "h1" else 12,
                    "feature_group": group_name,
                    "n_features": len(group_features),
                    "rmse": m["rmse"],
                    "mae": m["mae"],
                    "mape": m["mape"],
                    "n_samples": int(m["n_samples"]),
                    "notes": "best_params_based",
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "feature_ablation.csv", index=False)


def _collect_prediction_rows() -> pd.DataFrame:
    files = [
        "arima_predictions_h1.csv",
        "arima_predictions_h12.csv",
        "xgboost_predictions_h1.csv",
        "xgboost_predictions_h12.csv",
        "lstm_predictions_h1.csv",
        "lstm_predictions_h12.csv",
        "transformer_predictions_h1.csv",
        "transformer_predictions_h12.csv",
    ]

    dfs = []
    for name in files:
        path = OUTPUT_DIR / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "split" in df.columns:
            df = df[df["split"] == "test"].copy()
        df["target_time"] = pd.to_datetime(df["target_time"])
        df["abs_error"] = (df["y_true"] - df["y_pred"]).abs()
        df["sq_error"] = (df["y_true"] - df["y_pred"]) ** 2
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No prediction files found for analysis")

    return pd.concat(dfs, ignore_index=True)


def run_error_analyses() -> None:
    df = _collect_prediction_rows()

    # AQI bucket
    bins = [-np.inf, 35, 75, 115, 150, np.inf]
    labels = ["0-35", "35-75", "75-115", "115-150", ">150"]
    df["aqi_bucket"] = pd.cut(df["y_true"], bins=bins, labels=labels, right=True)

    aqi_rows = []
    for (model, horizon, bucket), g in df.groupby(["model", "horizon_hours", "aqi_bucket"], dropna=False):
        if len(g) == 0:
            continue
        m = compute_metrics(g["y_true"].values, g["y_pred"].values)
        aqi_rows.append(
            {
                "model": model,
                "horizon_hours": int(horizon),
                "aqi_bucket": str(bucket),
                "n_samples": int(m["n_samples"]),
                "rmse": m["rmse"],
                "mae": m["mae"],
                "mape": m["mape"],
            }
        )
    pd.DataFrame(aqi_rows).to_csv(OUTPUT_DIR / "aqi_bucket_metrics.csv", index=False)

    # High pollution
    hp = df[df["y_true"] > 150].copy()
    hp_rows = []
    for (model, horizon), g in hp.groupby(["model", "horizon_hours"]):
        if len(g) == 0:
            continue
        m = compute_metrics(g["y_true"].values, g["y_pred"].values)
        hp_rows.append(
            {
                "model": model,
                "horizon_hours": int(horizon),
                "n_samples": int(m["n_samples"]),
                "rmse": m["rmse"],
                "mae": m["mae"],
                "mape": m["mape"],
                "mean_abs_error": float(g["abs_error"].mean()),
            }
        )
    pd.DataFrame(hp_rows).to_csv(OUTPUT_DIR / "high_pollution_error_analysis.csv", index=False)

    # Error by hour
    df["hour"] = df["target_time"].dt.hour
    hour_rows = []
    for (model, horizon, hour), g in df.groupby(["model", "horizon_hours", "hour"]):
        m = compute_metrics(g["y_true"].values, g["y_pred"].values)
        hour_rows.append(
            {
                "model": model,
                "horizon_hours": int(horizon),
                "hour": int(hour),
                "n_samples": int(m["n_samples"]),
                "rmse": m["rmse"],
                "mae": m["mae"],
                "mape": m["mape"],
                "mean_abs_error": float(g["abs_error"].mean()),
            }
        )
    pd.DataFrame(hour_rows).to_csv(OUTPUT_DIR / "hourly_error_by_hour.csv", index=False)

    # Error by day period
    def period_from_hour(h: int) -> str:
        if 0 <= h <= 5:
            return "凌晨"
        if 6 <= h <= 9:
            return "早高峰"
        if 10 <= h <= 16:
            return "白天平峰"
        if 17 <= h <= 20:
            return "晚高峰"
        return "夜间"

    df["dayperiod"] = df["hour"].map(period_from_hour)

    dp_rows = []
    for (model, horizon, dayperiod), g in df.groupby(["model", "horizon_hours", "dayperiod"]):
        m = compute_metrics(g["y_true"].values, g["y_pred"].values)
        dp_rows.append(
            {
                "model": model,
                "horizon_hours": int(horizon),
                "dayperiod": dayperiod,
                "n_samples": int(m["n_samples"]),
                "rmse": m["rmse"],
                "mae": m["mae"],
                "mape": m["mape"],
                "mean_abs_error": float(g["abs_error"].mean()),
            }
        )
    pd.DataFrame(dp_rows).to_csv(OUTPUT_DIR / "hourly_error_by_dayperiod.csv", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_feature_ablation()
    run_error_analyses()
    print("Generated: feature_ablation, AQI/high-pollution/hour/dayperiod analysis files")


if __name__ == "__main__":
    main()
