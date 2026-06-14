"""Run hourly analysis modules for the h1 + seq6 experiment."""

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


def load_direct_data(horizon: int) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a direct supervised dataset for one forecast horizon."""
    df = pd.read_csv(PROCESSED_DIR / "beijing_hourly.csv")
    frame, all_features = build_hourly_feature_frame(df)
    target_col = f"pm25_target_h{horizon}"
    frame[target_col] = frame["pm25"].shift(-horizon)
    valid = frame.dropna(subset=all_features + [target_col]).reset_index(drop=True)

    split_idx = int(len(valid) * 0.8)
    train = valid.iloc[:split_idx]
    test = valid.iloc[split_idx:]
    return (
        all_features,
        train[all_features].values.astype(np.float64),
        test[all_features].values.astype(np.float64),
        train[target_col].values.astype(np.float64),
        test[target_col].values.astype(np.float64),
    )


def feature_groups(all_features: list[str]) -> dict[str, list[str]]:
    """Define progressive feature subsets for ablation."""
    has = set(all_features)

    def keep(names: list[str]) -> list[str]:
        return [name for name in names if name in has]

    base_short_lag = keep(
        [
            "pm25_lag1",
            "pm25_lag2",
            "pm25_lag3",
            "temp",
            "pres",
            "dewp",
            "humidity",
            "wind_speed",
            "precipitation",
        ]
    )
    base_plus_daily_cycle = list(dict.fromkeys(base_short_lag + keep(["hour_sin", "hour_cos", "is_rush_hour"])))
    mid_range_memory = list(dict.fromkeys(base_plus_daily_cycle + keep(["pm25_lag6", "pm25_lag8", "pm25_lag12", "pm25_lag24"])))
    trend_and_roll = list(
        dict.fromkeys(
            mid_range_memory
            + keep(
                [
                    "pm25_roll_mean_3",
                    "pm25_roll_mean_6",
                    "pm25_roll_mean_12",
                    "pm25_roll_mean_24",
                    "pm25_roll_std_3",
                    "pm25_roll_std_6",
                    "pm25_diff_1",
                    "pm25_diff_3",
                    "pm25_diff_6",
                ]
            )
        )
    )
    weather_enhanced = list(
        dict.fromkeys(
            trend_and_roll
            + keep(
                [
                    "wind_dir_sin",
                    "wind_dir_cos",
                    "precipitation_flag",
                    "dewp_temp_gap",
                    "temp_diff_1",
                    "pres_diff_1",
                    "wind_speed_diff_1",
                    "temp_x_humidity",
                    "wind_speed_x_wind_dir_sin",
                    "wind_speed_x_pm25_lag1",
                ]
            )
        )
    )

    return {
        "base_short_lag": base_short_lag,
        "base_plus_daily_cycle": base_plus_daily_cycle,
        "mid_range_memory": mid_range_memory,
        "trend_and_roll": trend_and_roll,
        "weather_enhanced": weather_enhanced,
        "full_48": list(all_features),
    }


def run_feature_ablation() -> None:
    """Run XGBoost ablation for h1 and seq6 horizons."""
    rows: list[dict] = []

    for task, horizons, params_path in [
        ("h1", [1], OUTPUT_DIR / "xgboost_h1_best_params.json"),
        ("seq6", [1, 2, 3, 4, 5, 6], OUTPUT_DIR / "xgboost_seq6_best_params.json"),
    ]:
        params_blob = json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else {}

        for horizon in horizons:
            all_features, X_train_raw, X_test_raw, y_train, y_test = load_direct_data(horizon)
            groups = feature_groups(all_features)
            horizon_params = params_blob.get(f"h{horizon}", params_blob) if isinstance(params_blob, dict) else None

            for group_name, group_features in groups.items():
                indices = [all_features.index(column) for column in group_features]
                X_train_sel = X_train_raw[:, indices]
                X_test_sel = X_test_raw[:, indices]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train_sel)
                X_test = scaler.transform(X_test_sel)

                val_split = int(len(X_train) * 0.8)
                X_tr, X_va = X_train[:val_split], X_train[val_split:]
                y_tr, y_va = y_train[:val_split], y_train[val_split:]

                model = XGBoostHourlyModel(params=horizon_params or None)
                model.fit(X_tr, y_tr, X_va, y_va, feature_names=group_features)
                y_pred = model.predict(X_test)
                metrics = compute_metrics(y_test, y_pred)

                rows.append(
                    {
                        "model": "XGBoost",
                        "task": task,
                        "horizon_hours": horizon,
                        "feature_group": group_name,
                        "n_features": len(group_features),
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "mape": metrics["mape"],
                        "n_samples": int(metrics["n_samples"]),
                        "notes": "best_params_based",
                    }
                )

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "feature_ablation.csv", index=False)


def collect_prediction_rows() -> pd.DataFrame:
    """Read every available prediction file into one analysis table."""
    files = [
        "arima_predictions_h1.csv",
        "arima_predictions_seq6.csv",
        "prophet_predictions_h1.csv",
        "prophet_predictions_seq6.csv",
        "xgboost_predictions_h1.csv",
        "xgboost_predictions_seq6.csv",
        "lstm_predictions_h1.csv",
        "lstm_predictions_seq6.csv",
        "transformer_predictions_h1.csv",
        "transformer_predictions_seq6.csv",
    ]

    frames: list[pd.DataFrame] = []
    for name in files:
        path = OUTPUT_DIR / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "split" in df.columns:
            df = df[df["split"] == "test"].copy()
        if df.empty:
            continue
        if "task" not in df.columns:
            df["task"] = "seq6" if "seq6" in name else "h1"
        df["target_time"] = pd.to_datetime(df["target_time"])
        df["abs_error"] = (df["y_true"] - df["y_pred"]).abs()
        df["sq_error"] = (df["y_true"] - df["y_pred"]) ** 2
        frames.append(df)

    if not frames:
        raise RuntimeError("No prediction files found for analysis")
    return pd.concat(frames, ignore_index=True)


def run_error_analyses() -> None:
    """Produce AQI bucket, high-pollution, by-hour and day-period analysis tables."""
    df = collect_prediction_rows()

    bins = [-np.inf, 35, 75, 115, 150, np.inf]
    labels = ["0-35", "35-75", "75-115", "115-150", ">150"]
    df["aqi_bucket"] = pd.cut(df["y_true"], bins=bins, labels=labels, right=True)

    aqi_rows = []
    for (model, task, horizon, bucket), group in df.groupby(["model", "task", "horizon_hours", "aqi_bucket"], dropna=False):
        metrics = compute_metrics(group["y_true"].values, group["y_pred"].values)
        aqi_rows.append(
            {
                "model": model,
                "task": task,
                "horizon_hours": int(horizon),
                "aqi_bucket": str(bucket),
                "n_samples": int(metrics["n_samples"]),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "mape": metrics["mape"],
            }
        )
    pd.DataFrame(aqi_rows).to_csv(OUTPUT_DIR / "aqi_bucket_metrics.csv", index=False)

    high_pollution = df[df["y_true"] > 150].copy()
    hp_rows = []
    for (model, task, horizon), group in high_pollution.groupby(["model", "task", "horizon_hours"]):
        metrics = compute_metrics(group["y_true"].values, group["y_pred"].values)
        hp_rows.append(
            {
                "model": model,
                "task": task,
                "horizon_hours": int(horizon),
                "n_samples": int(metrics["n_samples"]),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "mape": metrics["mape"],
                "mean_abs_error": float(group["abs_error"].mean()),
            }
        )
    pd.DataFrame(hp_rows).to_csv(OUTPUT_DIR / "high_pollution_error_analysis.csv", index=False)

    df["hour"] = df["target_time"].dt.hour
    hour_rows = []
    for (model, task, horizon, hour), group in df.groupby(["model", "task", "horizon_hours", "hour"]):
        metrics = compute_metrics(group["y_true"].values, group["y_pred"].values)
        hour_rows.append(
            {
                "model": model,
                "task": task,
                "horizon_hours": int(horizon),
                "hour": int(hour),
                "n_samples": int(metrics["n_samples"]),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "mape": metrics["mape"],
                "mean_abs_error": float(group["abs_error"].mean()),
            }
        )
    pd.DataFrame(hour_rows).to_csv(OUTPUT_DIR / "hourly_error_by_hour.csv", index=False)

    def period_from_hour(hour: int) -> str:
        if 0 <= hour <= 5:
            return "overnight"
        if 6 <= hour <= 9:
            return "morning_peak"
        if 10 <= hour <= 16:
            return "daytime"
        if 17 <= hour <= 20:
            return "evening_peak"
        return "night"

    df["dayperiod"] = df["hour"].map(period_from_hour)
    dp_rows = []
    for (model, task, horizon, dayperiod), group in df.groupby(["model", "task", "horizon_hours", "dayperiod"]):
        metrics = compute_metrics(group["y_true"].values, group["y_pred"].values)
        dp_rows.append(
            {
                "model": model,
                "task": task,
                "horizon_hours": int(horizon),
                "dayperiod": dayperiod,
                "n_samples": int(metrics["n_samples"]),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "mape": metrics["mape"],
                "mean_abs_error": float(group["abs_error"].mean()),
            }
        )
    pd.DataFrame(dp_rows).to_csv(OUTPUT_DIR / "hourly_error_by_dayperiod.csv", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_feature_ablation()
    run_error_analyses()
    print("Generated feature_ablation and error analysis outputs.")


if __name__ == "__main__":
    main()
