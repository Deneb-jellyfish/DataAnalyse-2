"""Train XGBoost for hourly PM2.5 h1 and seq6 tasks."""

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

from models.xgboost_model import XGBoostHourlyModel
from utils.feature_engineering import StandardScaler, build_hourly_feature_frame
from utils.hourly_experiment import make_prediction_frame, metric_rows_from_predictions
from utils.hourly_output_paths import artifact_path, ensure_hourly_output_dirs
from utils.io import PROCESSED_DIR

OUTPUT_DIR = ROOT / "outputs" / "hourly"


def load_direct_data(horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build a direct supervised dataset for one forecast horizon."""
    df = pd.read_csv(PROCESSED_DIR / "beijing_hourly.csv")
    frame, feature_names = build_hourly_feature_frame(df)
    target_col = f"pm25_target_h{horizon}"
    frame[target_col] = frame["pm25"].shift(-horizon)
    valid = frame.dropna(subset=feature_names + [target_col]).reset_index(drop=True)

    split_idx = int(len(valid) * 0.8)
    train = valid.iloc[:split_idx]
    test = valid.iloc[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_names].values.astype(np.float64))
    X_test = scaler.transform(test[feature_names].values.astype(np.float64))
    y_train = train[target_col].values.astype(np.float64)
    y_test = test[target_col].values.astype(np.float64)
    dates_test = test["datetime"].values
    return X_train, X_test, y_train, y_test, dates_test, feature_names


def _sample_params(trial: optuna.Trial, task: str, tune_stage: str) -> dict:
    """Sample XGBoost hyperparameters for coarse or refined tuning."""
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
    elif task == "seq6":
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
    """Run Optuna with rolling CV and return the best parameter set."""
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
        study_name=f"xgb_{task}_{tune_stage}",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


def _fit_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    params: dict | None,
) -> tuple[XGBoostHourlyModel, dict, float]:
    val_split = int(len(X_train) * 0.8)
    X_tr, X_va = X_train[:val_split], X_train[val_split:]
    y_tr, y_va = y_train[:val_split], y_train[val_split:]

    train_params = dict(params) if params else None
    if train_params is not None:
        train_params["early_stopping_rounds"] = 100

    model = XGBoostHourlyModel(params=train_params)
    t0 = time.time()
    meta = model.fit(X_tr, y_tr, X_va, y_va, feature_names=feature_names)
    train_time = time.time() - t0
    return model, meta, train_time


def run_h1(tune: bool = False, n_trials: int = 30, tune_stage: str = "first") -> list[dict]:
    """Train and save h1 outputs."""
    X_train, X_test, y_train, y_test, dates_test, feature_names = load_direct_data(horizon=1)
    best_params = (
        _tune_task("h1", X_train, y_train, feature_names, n_trials=n_trials, tune_stage=tune_stage)
        if tune
        else None
    )

    model, meta, train_time = _fit_model(X_train, y_train, feature_names, best_params)
    y_pred = model.predict(X_test)

    pred_df = make_prediction_frame(
        forecast_times=dates_test,
        y_true=y_test,
        y_pred=y_pred,
        horizon=1,
        model_name="XGBoost",
        task="h1",
    )
    pred_df.to_csv(artifact_path("xgboost_predictions_h1.csv", ensure_parent=True), index=False)
    model.get_feature_importance().to_csv(artifact_path("xgboost_feature_importance_h1.csv", ensure_parent=True), index=False)
    model.save(artifact_path("xgboost_h1.pkl", ensure_parent=True))

    params_to_save = dict(model.params)
    if best_params:
        params_to_save.update(best_params)
    with open(artifact_path("xgboost_h1_best_params.json", ensure_parent=True), "w", encoding="utf-8") as handle:
        json.dump(params_to_save, handle, ensure_ascii=False, indent=2)

    notes = f"best_iteration={meta['best_iteration']},train_time_s={train_time:.1f},tuned={tune},stage={tune_stage}"
    return metric_rows_from_predictions(pred_df, model_name="XGBoost", task="h1", notes=notes)


def run_seq6(tune: bool = False, n_trials: int = 30, tune_stage: str = "first") -> list[dict]:
    """Train six direct-horizon models and assemble seq6 outputs."""
    pred_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []
    param_map: dict[str, dict] = {}
    note_chunks: list[str] = []

    for horizon in range(1, 7):
        X_train, X_test, y_train, y_test, dates_test, feature_names = load_direct_data(horizon=horizon)
        best_params = (
            _tune_task("seq6", X_train, y_train, feature_names, n_trials=n_trials, tune_stage=tune_stage)
            if tune
            else None
        )

        model, meta, train_time = _fit_model(X_train, y_train, feature_names, best_params)
        y_pred = model.predict(X_test)

        pred_frames.append(
            make_prediction_frame(
                forecast_times=dates_test,
                y_true=y_test,
                y_pred=y_pred,
                horizon=horizon,
                model_name="XGBoost",
                task="seq6",
            )
        )

        feature_df = model.get_feature_importance().copy()
        feature_df["horizon_hours"] = horizon
        feature_frames.append(feature_df)

        params_to_save = dict(model.params)
        if best_params:
            params_to_save.update(best_params)
        param_map[f"h{horizon}"] = params_to_save
        model.save(artifact_path(f"xgboost_seq6_h{horizon}.pkl", ensure_parent=True))
        note_chunks.append(
            f"h{horizon}:best_iteration={meta['best_iteration']},train_time_s={train_time:.1f}"
        )

    pred_df = pd.concat(pred_frames, ignore_index=True)
    pred_df.to_csv(artifact_path("xgboost_predictions_seq6.csv", ensure_parent=True), index=False)

    feature_all = pd.concat(feature_frames, ignore_index=True)
    feature_summary = (
        feature_all.groupby("feature", as_index=False)[["gain", "gain_norm"]]
        .mean()
        .sort_values("gain_norm", ascending=False)
        .reset_index(drop=True)
    )
    feature_summary.to_csv(artifact_path("xgboost_feature_importance_seq6.csv", ensure_parent=True), index=False)

    with open(artifact_path("xgboost_seq6_best_params.json", ensure_parent=True), "w", encoding="utf-8") as handle:
        json.dump(param_map, handle, ensure_ascii=False, indent=2)

    notes = f"direct_multi_horizon,tuned={tune},stage={tune_stage}," + ";".join(note_chunks)
    return metric_rows_from_predictions(pred_df, model_name="XGBoost", task="seq6", notes=notes)


def save_metrics(rows: list[dict]) -> None:
    """Persist combined metrics."""
    path = artifact_path("xgboost_metrics.csv", ensure_parent=True)
    df = pd.DataFrame(rows)
    cols = ["model", "task", "horizon_hours", "rmse", "mae", "mape", "n_samples", "notes"]
    for column in cols:
        if column not in df.columns:
            df[column] = ""
    df = df[cols].sort_values(["task", "horizon_hours"]).reset_index(drop=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost for hourly PM2.5")
    parser.add_argument("--task", default="both", choices=["h1", "seq6", "both"])
    parser.add_argument("--tune", action="store_true", help="Run Optuna tuning for selected task(s)")
    parser.add_argument("--tune-stage", default="first", choices=["first", "second"], help="Tuning stage")
    parser.add_argument("--trials", type=int, default=24)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_hourly_output_dirs()

    rows: list[dict] = []
    if args.task in ("h1", "both"):
        rows.extend(run_h1(tune=args.tune, n_trials=args.trials, tune_stage=args.tune_stage))
    if args.task in ("seq6", "both"):
        rows.extend(run_seq6(tune=args.tune, n_trials=args.trials, tune_stage=args.tune_stage))
    save_metrics(rows)


if __name__ == "__main__":
    main()
