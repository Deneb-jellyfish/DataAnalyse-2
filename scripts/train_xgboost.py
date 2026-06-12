"""Train XGBoost for hourly PM2.5 prediction tasks (+1h and +24h).

Usage:
  python scripts/train_xgboost.py --task h1
  python scripts/train_xgboost.py --task h24
  python scripts/train_xgboost.py --task both
"""

from __future__ import annotations

import argparse
import json
import pickle
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
from evaluation.metrics import compute_metrics
from utils.feature_engineering import build_hourly_feature_frame, StandardScaler
from utils.io import PROCESSED_DIR

OUTPUT_DIR = ROOT / "outputs" / "hourly"
FEATURES_DIR = PROCESSED_DIR / "features_hourly"


def load_h1_data():
    """Load pre-built +1h feature matrices (y = current pm25)."""
    X_train = np.load(FEATURES_DIR / "X_train.npy")
    X_test = np.load(FEATURES_DIR / "X_test.npy")
    y_train = np.load(FEATURES_DIR / "y_train.npy")
    y_test = np.load(FEATURES_DIR / "y_test.npy")
    dates_train = np.load(FEATURES_DIR / "dates_train.npy", allow_pickle=True)
    dates_test = np.load(FEATURES_DIR / "dates_test.npy", allow_pickle=True)
    with open(FEATURES_DIR / "feature_names.json") as f:
        feature_names = json.load(f)

    return X_train, X_test, y_train, y_test, dates_train, dates_test, feature_names


def load_h24_data():
    """Build +24h data by shifting pm25 target by -24 hours.

    Uses enhanced features: the standard 20 + pm25(t) + longer lags.
    For +24h, pm25(t) is NOT look-ahead — we know the current value
    when forecasting 24h into the future.
    """
    df = pd.read_csv(PROCESSED_DIR / "beijing_hourly.csv")
    frame, base_features = build_hourly_feature_frame(df)

    # Add pm25(t) — legitimate for +24h (no look-ahead)
    frame["pm25_current"] = frame["pm25"]

    # Add longer lags for daily-cycle patterns
    frame["pm25_lag48"] = frame["pm25"].shift(48)    # 2 days ago same hour
    frame["pm25_lag72"] = frame["pm25"].shift(72)    # 3 days ago same hour
    frame["pm25_lag168"] = frame["pm25"].shift(168)   # 7 days ago same hour

    # Extended rolling stats
    frame["pm25_roll_mean_48"] = frame["pm25"].shift(1).rolling(48).mean()
    frame["pm25_roll_std_48"] = frame["pm25"].shift(1).rolling(48).std()

    # Interaction: current PM2.5 × wind speed (ventilation effect)
    frame["pm25_x_wind_speed"] = frame["pm25"] * frame["wind_speed"].fillna(0)

    h24_extra = [
        "pm25_current",
        "pm25_lag48", "pm25_lag72", "pm25_lag168",
        "pm25_roll_mean_48", "pm25_roll_std_48",
        "pm25_x_wind_speed",
    ]
    feature_names = base_features + h24_extra

    # Target: PM2.5 at t+24
    frame["pm25_target_h24"] = frame["pm25"].shift(-24)

    # Drop rows with NaN in features or target
    valid = frame.dropna(subset=feature_names + ["pm25_target_h24"]).reset_index(drop=True)
    print(f"  h24 valid rows: {len(valid)} (dropped {len(frame) - len(valid)})")

    # Chronological 80/20 split
    split_idx = int(len(valid) * 0.8)
    train = valid.iloc[:split_idx]
    test = valid.iloc[split_idx:]

    # Scale features using train stats
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_names].values)
    X_test = scaler.transform(test[feature_names].values)
    y_train = train["pm25_target_h24"].values.astype(np.float64)
    y_test = test["pm25_target_h24"].values.astype(np.float64)
    dates_train = train["datetime"].values
    dates_test = test["datetime"].values

    print(f"  h24 train: {len(train)}, test: {len(test)}, features: {len(feature_names)}")
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
    """Build unified-format predictions DataFrame."""
    forecast_dt = pd.to_datetime(forecast_times)
    target_dt = forecast_dt + pd.Timedelta(hours=horizon)

    return pd.DataFrame({
        "forecast_origin_time": forecast_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "target_time": target_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "horizon_hours": horizon,
        "city": city,
        "model": model_name,
        "y_true": y_true,
        "y_pred": y_pred,
        "split": split,
    })


def run_h1() -> dict:
    """Train and evaluate XGBoost for +1h prediction."""
    print("=" * 60)
    print("XGBoost +1h")
    print("=" * 60)

    X_train, X_test, y_train, y_test, dates_train, dates_test, feature_names = load_h1_data()

    # Split train into train/val (80/20 of train, chronologically)
    val_split = int(len(X_train) * 0.8)
    X_tr, X_va = X_train[:val_split], X_train[val_split:]
    y_tr, y_va = y_train[:val_split], y_train[val_split:]

    print(f"  Train: {len(X_tr)}, Val: {len(X_va)}, Test: {len(X_test)}")
    print(f"  Features: {len(feature_names)}")

    # Train
    t0 = time.time()
    model = XGBoostHourlyModel()
    meta = model.fit(X_tr, y_tr, X_va, y_va, feature_names=feature_names)
    train_time = time.time() - t0
    print(f"  Best iteration: {meta['best_iteration']}, best score: {meta['best_score']:.4f}")
    print(f"  Train time: {train_time:.1f}s")

    # Predict
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    print(f"  Test RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.1f}%")

    # Save predictions
    pred_df = make_prediction_csv(dates_test, y_test, y_pred, horizon=1, model_name="XGBoost")
    pred_path = OUTPUT_DIR / "xgboost_predictions_h1.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"  Saved: {pred_path}")

    # Save feature importance
    imp_df = model.get_feature_importance()
    imp_path = OUTPUT_DIR / "xgboost_feature_importance_h1.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"  Saved: {imp_path}")

    # Save model
    model.save(OUTPUT_DIR / "xgboost_h1.pkl")

    return {
        "model": "XGBoost",
        "horizon_hours": 1,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mape": metrics["mape"],
        "n_train": len(X_tr),
        "n_val": len(X_va),
        "n_test": len(X_test),
        "best_iteration": meta["best_iteration"],
        "train_time_s": round(train_time, 1),
    }


def _tune_h24(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    n_trials: int = 30,
    n_splits: int = 5,
) -> dict:
    """Run Optuna hyperparameter search with TimeSeriesSplit CV.

    Returns best hyperparameter dict.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
        }

        fold_scores = []
        for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
            X_tr_fold = X_train[tr_idx]
            X_va_fold = X_train[va_idx]
            y_tr_fold = y_train[tr_idx]
            y_va_fold = y_train[va_idx]

            model = XGBoostHourlyModel(params=params)
            model.fit(X_tr_fold, y_tr_fold, X_va_fold, y_va_fold,
                      feature_names=feature_names)
            y_pred_fold = model.predict(X_va_fold)
            rmse_fold = float(np.sqrt(np.mean((y_va_fold - y_pred_fold) ** 2)))
            fold_scores.append(rmse_fold)

            trial.report(rmse_fold, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best CV RMSE: {study.best_value:.2f}")
    print(f"Best params: {study.best_params}")

    return study.best_params


def run_h24(tune: bool = False, n_trials: int = 30) -> dict:
    """Train and evaluate XGBoost for +24h prediction."""
    print("=" * 60)
    print("XGBoost +24h")
    if tune:
        print(f"Hyperparameter tuning enabled (Optuna + TimeSeriesSplit CV, {n_trials} trials)")
    print("=" * 60)

    X_train, X_test, y_train, y_test, dates_train, dates_test, feature_names = load_h24_data()

    # --- Hyperparameter tuning ---
    best_params = None
    if tune:
        t0_tune = time.time()
        best_params = _tune_h24(X_train, y_train, feature_names, n_trials=n_trials)
        tune_time = time.time() - t0_tune
        print(f"  Tuning time: {tune_time:.1f}s")

        # Save best params
        params_path = OUTPUT_DIR / "xgboost_h24_best_params.json"
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"  Best params saved: {params_path}")

    # Split train into train/val (use 90/10 when tuned for more training data)
    val_ratio = 0.9 if tune else 0.8
    val_split = int(len(X_train) * val_ratio)
    X_tr, X_va = X_train[:val_split], X_train[val_split:]
    y_tr, y_va = y_train[:val_split], y_train[val_split:]

    print(f"  Train: {len(X_tr)}, Val: {len(X_va)}, Test: {len(X_test)}")
    print(f"  Features: {len(feature_names)}")

    # Train (use longer early stopping when tuned since best learning rates are lower)
    train_params = dict(best_params) if best_params else None
    if tune and train_params is not None:
        train_params["early_stopping_rounds"] = 100
    t0 = time.time()
    model = XGBoostHourlyModel(params=train_params)
    meta = model.fit(X_tr, y_tr, X_va, y_va, feature_names=feature_names)
    train_time = time.time() - t0
    print(f"  Best iteration: {meta['best_iteration']}, best score: {meta['best_score']:.4f}")
    print(f"  Train time: {train_time:.1f}s")

    # Predict
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    print(f"  Test RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.1f}%")

    # Save predictions
    pred_df = make_prediction_csv(dates_test, y_test, y_pred, horizon=24, model_name="XGBoost")
    pred_path = OUTPUT_DIR / "xgboost_predictions_h24.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"  Saved: {pred_path}")

    # Save feature importance
    imp_df = model.get_feature_importance()
    imp_path = OUTPUT_DIR / "xgboost_feature_importance_h24.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"  Saved: {imp_path}")

    # Save model
    model.save(OUTPUT_DIR / "xgboost_h24.pkl")

    return {
        "model": "XGBoost",
        "horizon_hours": 24,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mape": metrics["mape"],
        "n_train": len(X_tr),
        "n_val": len(X_va),
        "n_test": len(X_test),
        "best_iteration": meta["best_iteration"],
        "train_time_s": round(train_time, 1),
    }


def save_metrics(rows: list[dict]) -> None:
    """Save combined metrics CSV."""
    path = OUTPUT_DIR / "xgboost_metrics.csv"
    df = pd.DataFrame(rows)
    # Add notes column if not present
    if "notes" not in df.columns:
        df["notes"] = ""
    df = df[["model", "horizon_hours", "rmse", "mae", "mape", "n_train", "n_val",
             "n_test", "best_iteration", "train_time_s", "notes"]]
    df.to_csv(path, index=False)
    print(f"Metrics saved: {path}")


def append_registry(rows: list[dict]) -> None:
    """Append experiment entries to registry."""
    path = OUTPUT_DIR / "experiment_registry.csv"
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    records = []
    for r in rows:
        records.append({
            "experiment_id": f"xgb_h{r['horizon_hours']}_{now.replace(' ', 'T')}",
            "model": r["model"],
            "horizon_hours": r["horizon_hours"],
            "feature_set": "full_20",
            "train_range": "",
            "val_range": "",
            "test_range": "",
            "params": "default",
            "best_score": r.get("best_iteration", ""),
            "run_time": r.get("train_time_s", ""),
            "status": "done",
            "author": "auto",
            "notes": f"RMSE={r['rmse']:.2f}",
        })

    df_new = pd.DataFrame(records)
    if path.exists():
        df_old = pd.read_csv(path)
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    df_new.to_csv(path, index=False)
    print(f"Registry updated: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost for hourly PM2.5")
    parser.add_argument("--task", default="both",
                        choices=["h1", "h24", "both"],
                        help="Prediction horizon task")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter tuning for +24h task")
    parser.add_argument("--trials", type=int, default=30,
                        help="Number of Optuna trials (default: 30)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    if args.task in ("h1", "both"):
        r = run_h1()
        all_rows.append(r)

    if args.task in ("h24", "both"):
        r = run_h24(tune=args.tune, n_trials=args.trials)
        all_rows.append(r)

    save_metrics(all_rows)
    append_registry(all_rows)

    print("\n" + "=" * 60)
    print("XGBoost Summary")
    print("=" * 60)
    for r in all_rows:
        print(f"  +{r['horizon_hours']}h: RMSE={r['rmse']:.2f}, MAE={r['mae']:.2f}, MAPE={r['mape']:.1f}%")


if __name__ == "__main__":
    main()
