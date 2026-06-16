"""Train an enhanced Prophet baseline for hourly PM2.5 h1 and seq6 tasks."""

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

from utils.feature_engineering import build_hourly_feature_frame
from utils.hourly_experiment import make_prediction_frame, metric_rows_from_predictions
from utils.hourly_output_paths import artifact_path, ensure_hourly_output_dirs
from utils.io import PROCESSED_DIR

OUTPUT_DIR = ROOT / "outputs" / "hourly"


def load_direct_data(horizon: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build a direct supervised dataset for one horizon.

    The target remains the PM2.5 value at `t + horizon`, while all regressors are
    computed from the origin timestamp `t`. This mirrors the direct multi-horizon
    setup used by the tree models and avoids the degenerate "time-only" Prophet fit.
    """

    df = pd.read_csv(PROCESSED_DIR / "beijing_hourly.csv")
    frame, feature_names = build_hourly_feature_frame(df)

    target_col = f"pm25_target_h{horizon}"
    frame[target_col] = frame["pm25"].shift(-horizon)
    frame["target_time"] = pd.to_datetime(frame["datetime"]) + pd.Timedelta(hours=horizon)
    valid = frame.dropna(subset=feature_names + [target_col]).reset_index(drop=True)

    prophet_df = valid[["datetime", "target_time", target_col] + feature_names].copy()
    prophet_df = prophet_df.rename(
        columns={
            "datetime": "forecast_origin_time",
            "target_time": "ds",
            target_col: "y",
        }
    )

    split_idx = int(len(prophet_df) * 0.8)
    train_df = prophet_df.iloc[:split_idx].copy()
    test_df = prophet_df.iloc[split_idx:].copy()
    return train_df, test_df, feature_names


def fit_prophet(train_df: pd.DataFrame, feature_names: list[str]):
    """Fit one enhanced Prophet model on a direct supervised split."""
    from prophet import Prophet

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=5.0,
        seasonality_mode="additive",
    )
    for feature in feature_names:
        model.add_regressor(feature, standardize="auto")

    t0 = time.time()
    model.fit(train_df[["ds", "y"] + feature_names].copy())
    fit_time = time.time() - t0
    return model, fit_time


def _predict(model, df: pd.DataFrame, feature_names: list[str]) -> tuple[np.ndarray, float]:
    """Generate clipped Prophet predictions with elapsed time."""
    t0 = time.time()
    forecast = model.predict(df[["ds"] + feature_names].copy())
    pred_time = time.time() - t0
    y_pred = np.clip(forecast["yhat"].to_numpy(dtype=np.float64), 0.0, None)
    return y_pred, pred_time


def run_h1() -> list[dict]:
    """Train and save enhanced h1 Prophet outputs."""
    train_df, test_df, feature_names = load_direct_data(horizon=1)
    model, fit_time = fit_prophet(train_df, feature_names)
    y_pred, pred_time = _predict(model, test_df, feature_names)

    pred_df = make_prediction_frame(
        forecast_times=test_df["forecast_origin_time"].to_numpy(),
        y_true=test_df["y"].to_numpy(dtype=np.float64),
        y_pred=y_pred,
        horizon=1,
        model_name="Prophet",
        task="h1",
    )
    pred_path = artifact_path("prophet_predictions_h1.csv", ensure_parent=True)
    config_path = artifact_path("prophet_config_h1.json", ensure_parent=True)
    pred_df.to_csv(pred_path, index=False)

    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "task": "h1",
                "horizon_hours": 1,
                "mode": "direct_supervised_with_regressors",
                "daily_seasonality": True,
                "weekly_seasonality": True,
                "yearly_seasonality": False,
                "changepoint_prior_scale": 0.1,
                "seasonality_prior_scale": 5.0,
                "seasonality_mode": "additive",
                "n_regressors": len(feature_names),
                "regressors": feature_names,
                "fit_time_s": round(fit_time, 1),
                "pred_time_s": round(pred_time, 1),
                "status": "done",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    notes = (
        f"direct_with_48_features,fit_time_s={fit_time:.1f},pred_time_s={pred_time:.1f},"
        "yearly=False,changepoint=0.1,seasonality=5.0"
    )
    return metric_rows_from_predictions(pred_df, model_name="Prophet", task="h1", notes=notes)


def run_seq6() -> list[dict]:
    """Train six direct-horizon Prophet models and assemble seq6 outputs."""
    pred_frames: list[pd.DataFrame] = []
    fit_times: dict[str, float] = {}
    pred_times: dict[str, float] = {}
    feature_names_used: list[str] | None = None

    for horizon in range(1, 7):
        train_df, test_df, feature_names = load_direct_data(horizon=horizon)
        feature_names_used = feature_names
        model, fit_time = fit_prophet(train_df, feature_names)
        y_pred, pred_time = _predict(model, test_df, feature_names)

        pred_frames.append(
            make_prediction_frame(
                forecast_times=test_df["forecast_origin_time"].to_numpy(),
                y_true=test_df["y"].to_numpy(dtype=np.float64),
                y_pred=y_pred,
                horizon=horizon,
                model_name="Prophet",
                task="seq6",
            )
        )
        fit_times[f"h{horizon}"] = round(fit_time, 1)
        pred_times[f"h{horizon}"] = round(pred_time, 1)

    pred_df = pd.concat(pred_frames, ignore_index=True)
    pred_path = artifact_path("prophet_predictions_seq6.csv", ensure_parent=True)
    config_path = artifact_path("prophet_config_seq6.json", ensure_parent=True)
    pred_df.to_csv(pred_path, index=False)

    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "task": "seq6",
                "horizons": [1, 2, 3, 4, 5, 6],
                "mode": "direct_multi_horizon_with_regressors",
                "daily_seasonality": True,
                "weekly_seasonality": True,
                "yearly_seasonality": False,
                "changepoint_prior_scale": 0.1,
                "seasonality_prior_scale": 5.0,
                "seasonality_mode": "additive",
                "n_regressors": len(feature_names_used or []),
                "regressors": feature_names_used or [],
                "fit_time_s_by_horizon": fit_times,
                "pred_time_s_by_horizon": pred_times,
                "status": "done",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    notes = (
        "direct_multi_horizon_with_48_features,"
        + ",".join(f"{key}:fit={value:.1f}" for key, value in fit_times.items())
        + ","
        + ",".join(f"{key}:pred={value:.1f}" for key, value in pred_times.items())
    )
    return metric_rows_from_predictions(pred_df, model_name="Prophet", task="seq6", notes=notes)


def save_metrics(rows: list[dict]) -> None:
    """Persist combined metrics."""
    path = artifact_path("prophet_metrics.csv", ensure_parent=True)
    df = pd.DataFrame(rows)
    cols = ["model", "task", "horizon_hours", "rmse", "mae", "mape", "n_samples", "notes"]
    for column in cols:
        if column not in df.columns:
            df[column] = ""
    df = df[cols].sort_values(["task", "horizon_hours"]).reset_index(drop=True)
    df.to_csv(path, index=False)


def write_blocked_outputs(reason: str) -> None:
    """Write blocked configs and empty metrics when Prophet cannot run."""
    rows = []
    for task, horizons in [("h1", [1]), ("seq6", [1, 2, 3, 4, 5, 6])]:
        for horizon in horizons:
            rows.append(
                {
                    "model": "Prophet",
                    "task": task,
                    "horizon_hours": horizon,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "mape": np.nan,
                    "n_samples": 0,
                    "notes": f"blocked:{reason}",
                }
            )
        config_name = "prophet_config_h1.json" if task == "h1" else "prophet_config_seq6.json"
        config_path = artifact_path(config_name, ensure_parent=True)
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump({"task": task, "status": "blocked", "reason": reason}, handle, ensure_ascii=False, indent=2)
    save_metrics(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train enhanced Prophet for hourly h1 and seq6 tasks")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually run Prophet training (default: write blocked outputs)",
    )
    parser.add_argument(
        "--task",
        default="both",
        choices=["h1", "seq6", "both"],
        help="Select which Prophet task to train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_hourly_output_dirs()

    if not args.run:
        write_blocked_outputs(reason="skipped_by_request")
        print("Prophet skipped by request. Use --run to enable real Prophet training.")
        return

    try:
        from prophet import Prophet  # noqa: F401
    except Exception as exc:
        write_blocked_outputs(reason=f"prophet_import_failed:{exc.__class__.__name__}")
        print("Prophet blocked: prophet package is unavailable in current environment.")
        return

    rows: list[dict] = []
    if args.task in ("h1", "both"):
        rows.extend(run_h1())
    if args.task in ("seq6", "both"):
        rows.extend(run_seq6())
    save_metrics(rows)
    print("Enhanced Prophet training complete.")


if __name__ == "__main__":
    main()
