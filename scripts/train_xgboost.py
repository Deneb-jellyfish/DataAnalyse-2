"""Train XGBoost for hourly PM2.5 prediction tasks (+1h and +12h)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.metrics import compute_metrics
from models.xgboost_model import XGBoostHourlyModel
from utils.feature_engineering import StandardScaler, build_hourly_feature_frame
from utils.io import PROCESSED_DIR

OUTPUT_DIR = ROOT / "outputs" / "hourly"
FEATURES_DIR = PROCESSED_DIR / "features_hourly"


def load_h1_data():
    """Build +1h data by shifting pm25 target by -1 hour."""
    df = pd.read_csv(PROCESSED_DIR / "beijing_hourly.csv")
    frame, feature_names = build_hourly_feature_frame(df)
    frame["pm25_target_h1"] = frame["pm25"].shift(-1)
    valid = frame.dropna(subset=feature_names + ["pm25_target_h1"]).reset_index(drop=True)

    split_idx = int(len(valid) * 0.8)
    train = valid.iloc[:split_idx]
    test = valid.iloc[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_names].values)
    X_test = scaler.transform(test[feature_names].values)
    y_train = train["pm25_target_h1"].values.astype(np.float64)
    y_test = test["pm25_target_h1"].values.astype(np.float64)
    dates_train = train["datetime"].values
    dates_test = test["datetime"].values

    return X_train, X_test, y_train, y_test, dates_train, dates_test, feature_names


def load_h12_data():
    df = pd.read_csv(PROCESSED_DIR / "beijing_hourly.csv")
    frame, feature_names = build_hourly_feature_frame(df)
    frame["pm25_target_h12"] = frame["pm25"].shift(-12)
    valid = frame.dropna(subset=feature_names + ["pm25_target_h12"]).reset_index(drop=True)

    split_idx = int(len(valid) * 0.8)
    train = valid.iloc[:split_idx]
    test = valid.iloc[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_names].values)
    X_test = scaler.transform(test[feature_names].values)
    y_train = train["pm25_target_h12"].values.astype(np.float64)
    y_test = test["pm25_target_h12"].values.astype(np.float64)
    dates_train = train["datetime"].values
    dates_test = test["datetime"].values

    return X_train, X_test, y_train, y_test, dates_train, dates_test, feature_names


def make_prediction_csv(
    forecast_times: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    model_name: str,
    city: str = "Beijing",
    split: str = "test",
) -> pd.DataFrame:
    forecast_dt = pd.to_datetime(forecast_times)
    target_dt = forecast_dt + pd.Timedelta(hours=int(horizon))
    return pd.DataFrame(
        {
            "forecast_origin_time": forecast_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "target_time": target_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "horizon_hours": int(horizon),
            "city": city,
            "model": model_name,
            "y_true": y_true,
            "y_pred": y_pred,
            "split": split,
        }
    )


def _sample_params(trial: optuna.Trial, task: str, tune_stage: str) -> dict:
    """Sample params by stage: first-round coarse grid, second-round local refinement."""
    if tune_stage == "first":
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [200, 400, 800]),
            "max_depth": trial.suggest_categorical("max_depth", [4, 6, 8]),
            "learning_rate": trial.suggest_categorical("learning_rate", [0.03, 0.05, 0.1]),
            "subsample": trial.suggest_categorical("subsample", [0.8, 1.0]),
            "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.8, 1.0]),
            "min_child_weight": trial.suggest_categorical("min_child_weight", [1, 3, 5]),
            "reg_lambda": trial.suggest_categorical("reg_lambda", [1, 3, 5]),
            "reg_alpha": trial.suggest_categorical("reg_alpha", [0.0, 0.1, 0.5]),
        }
    else:
        # Stage-2 local refinement for harder horizon (especially h12)
        if task == "h12":
            params = {
                "n_estimators": trial.suggest_categorical("n_estimators", [400, 600, 800, 1000, 1200]),
                "max_depth": trial.suggest_categorical("max_depth", [3, 4, 5, 6]),
                "learning_rate": trial.suggest_categorical("learning_rate", [0.015, 0.02, 0.03, 0.04, 0.05]),
                "subsample": trial.suggest_categorical("subsample", [0.8, 0.9, 1.0]),
                "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.7, 0.8, 0.9, 1.0]),
                "min_child_weight": trial.suggest_categorical("min_child_weight", [3, 5, 7, 9]),
                "reg_lambda": trial.suggest_categorical("reg_lambda", [3, 5, 7, 9]),
                "reg_alpha": trial.suggest_categorical("reg_alpha", [0.0, 0.05, 0.1, 0.2]),
                "gamma": trial.suggest_categorical("gamma", [0.0, 0.1, 0.3]),
            }
        else:
            params = {
                "n_estimators": trial.suggest_categorical("n_estimators", [300, 400, 500, 700]),
                "max_depth": trial.suggest_categorical("max_depth", [3, 4, 5, 6]),
                "learning_rate": trial.suggest_categorical("learning_rate", [0.02, 0.03, 0.04, 0.05]),
                "subsample": trial.suggest_categorical("subsample", [0.8, 0.9, 1.0]),
                "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.7, 0.8, 0.9, 1.0]),
                "min_child_weight": trial.suggest_categorical("min_child_weight", [1, 3, 5, 7]),
                "reg_lambda": trial.suggest_categorical("reg_lambda", [2, 3, 5, 7]),
                "reg_alpha": trial.suggest_categorical("reg_alpha", [0.0, 0.05, 0.1, 0.2]),
                "gamma": trial.suggest_categorical("gamma", [0.0, 0.1, 0.2]),
            }

    params["early_stopping_rounds"] = 100
    return params


def _tune_task(
    task: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    n_trials: int = 30,
    tune_stage: str = "first",
) -> dict:
    """Optuna tuning aligned with plan: first coarse grid, then local refinement."""
    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, task=task, tune_stage=tune_stage)

        fold_scores = []
        for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
            X_tr_fold, X_va_fold = X_train[tr_idx], X_train[va_idx]
            y_tr_fold, y_va_fold = y_train[tr_idx], y_train[va_idx]
            model = XGBoostHourlyModel(params=params)
            model.fit(X_tr_fold, y_tr_fold, X_va_fold, y_va_fold, feature_names=feature_names)
            y_pred_fold = model.predict(X_va_fold)
            rmse_fold = float(np.sqrt(np.mean((y_va_fold - y_pred_fold) ** 2)))
            fold_scores.append(rmse_fold)
            trial.report(rmse_fold, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="minimize",
        study_name=f"xgb_{task}_{tune_stage}_round",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


def _run_task(task: str, tune: bool = False, n_trials: int = 30, tune_stage: str = "first") -> dict:
    if task == "h1":
        horizon = 1
        X_train, X_test, y_train, y_test, _, dates_test, feature_names = load_h1_data()
    else:
        horizon = 12
        X_train, X_test, y_train, y_test, _, dates_test, feature_names = load_h12_data()

    best_params = (
        _tune_task(task, X_train, y_train, feature_names, n_trials=n_trials, tune_stage=tune_stage)
        if tune
        else None
    )

    val_split = int(len(X_train) * 0.8)
    X_tr, X_va = X_train[:val_split], X_train[val_split:]
    y_tr, y_va = y_train[:val_split], y_train[val_split:]

    train_params = dict(best_params) if best_params else None
    if train_params is not None:
        train_params["early_stopping_rounds"] = 100

    t0 = time.time()
    model = XGBoostHourlyModel(params=train_params)
    meta = model.fit(X_tr, y_tr, X_va, y_va, feature_names=feature_names)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    pred_df = make_prediction_csv(dates_test, y_test, y_pred, horizon=horizon, model_name="XGBoost")
    pred_df.to_csv(OUTPUT_DIR / f"xgboost_predictions_{task}.csv", index=False)

    imp_df = model.get_feature_importance()
    imp_df.to_csv(OUTPUT_DIR / f"xgboost_feature_importance_{task}.csv", index=False)

    model.save(OUTPUT_DIR / f"xgboost_{task}.pkl")

    params_to_save = dict(model.params)
    if best_params:
        params_to_save.update(best_params)
    with open(OUTPUT_DIR / f"xgboost_{task}_best_params.json", "w", encoding="utf-8") as f:
        json.dump(params_to_save, f, ensure_ascii=False, indent=2)

    return {
        "model": "XGBoost",
        "horizon_hours": int(horizon),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mape": metrics["mape"],
        "n_samples": int(metrics["n_samples"]),
        "notes": f"tuned={tune},stage={tune_stage},best_iteration={meta['best_iteration']},train_time_s={train_time:.1f}",
    }


def save_metrics(rows: list[dict]) -> None:
    path = OUTPUT_DIR / "xgboost_metrics.csv"
    df = pd.DataFrame(rows)
    cols = ["model", "horizon_hours", "rmse", "mae", "mape", "n_samples", "notes"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost for hourly PM2.5")
    parser.add_argument("--task", default="both", choices=["h1", "h12", "both"])
    parser.add_argument("--tune", action="store_true", help="Run Optuna tuning for selected task(s)")
    parser.add_argument("--tune-stage", default="first", choices=["first", "second"], help="Tuning stage")
    parser.add_argument("--trials", type=int, default=24)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    if args.task in ("h1", "both"):
        all_rows.append(_run_task("h1", tune=args.tune, n_trials=args.trials, tune_stage=args.tune_stage))
    if args.task in ("h12", "both"):
        all_rows.append(_run_task("h12", tune=args.tune, n_trials=args.trials, tune_stage=args.tune_stage))

    save_metrics(all_rows)


if __name__ == "__main__":
    main()
