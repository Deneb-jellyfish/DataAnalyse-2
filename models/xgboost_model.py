"""XGBoost model for Beijing PM2.5 daily prediction (B-3).

Uses the 18-dimension feature matrix prepared by Member A with
Optuna-driven Bayesian hyper-parameter tuning and time-series
cross-validation.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io import FEATURES_DIR, PROCESSED_DIR

log = logging.getLogger(__name__)

OPTUNA_TRIALS = 80
CV_SPLITS = 5
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
def _load_matrices() -> dict[str, Any]:
    """Load feature matrices and metadata produced by Member A."""
    data: dict[str, Any] = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        data[name] = np.load(FEATURES_DIR / f"{name}.npy", allow_pickle=False)
    for name in ["dates_train", "dates_test"]:
        data[name] = np.load(FEATURES_DIR / f"{name}.npy", allow_pickle=True)
    with open(FEATURES_DIR / "feature_names.json", encoding="utf-8") as fh:
        data["feature_names"] = json.load(fh)
    with open(FEATURES_DIR / "scaler.pkl", "rb") as fh:
        data["scaler"] = pickle.load(fh)
    return data


# ---------------------------------------------------------------------------
def _objective(trial: optuna.Trial, X_train: np.ndarray,
               y_train: np.ndarray) -> float:
    """Optuna objective — minimise CV RMSE."""
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "verbosity": 0,
        "seed": RANDOM_SEED,
        "n_jobs": 1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    rmses: list[float] = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        pred = model.predict(X_val)
        rmses.append(float(np.sqrt(np.mean((y_val - pred) ** 2))))
    return float(np.mean(rmses))


# ---------------------------------------------------------------------------
def run() -> dict[str, Any]:
    """Full XGBoost pipeline."""
    data = _load_matrices()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    log.info("XGBoost: X_train %s  X_test %s  %d features",
             X_train.shape, X_test.shape, X_train.shape[1])

    # --- hyper-param tuning ---
    log.info("Optuna tuning (%d trials, %d-fold TSCV) ...", OPTUNA_TRIALS, CV_SPLITS)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(
        lambda trial: _objective(trial, X_train, y_train),
        n_trials=OPTUNA_TRIALS, show_progress_bar=False,
    )
    best_params = study.best_params
    log.info("  best CV-RMSE=%.2f  params=%s", study.best_value, best_params)

    # --- final fit on all training data ---
    final = xgb.XGBRegressor(
        objective="reg:squarederror", eval_metric="rmse",
        verbosity=0, seed=RANDOM_SEED, n_jobs=1, **best_params,
    )
    final.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = final.predict(X_test)

    # --- metrics ---
    residuals = y_test - y_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    ape = np.abs(residuals / np.maximum(y_test, 1e-6)) * 100
    mape = float(np.mean(np.minimum(ape, 200.0)))

    # --- predictions ---
    test_dates = data["dates_test"]
    rows: list[dict[str, Any]] = []
    for d, t, p in zip(test_dates, y_test, y_pred):
        rows.append({
            "date": str(d),
            "city": "Beijing",
            "model": "XGBoost",
            "y_true": round(float(t), 4),
            "y_pred": round(max(0.0, float(p)), 4),
        })

    log.info("XGBoost  RMSE=%.2f  MAE=%.2f  MAPE=%.1f%%", rmse, mae, mape)
    return {
        "model": "XGBoost",
        "rmse": rmse, "mae": mae, "mape": mape,
        "best_params": best_params,
        "cv_rmse": float(study.best_value),
        "n_train": int(X_train.shape[0]), "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "predictions": rows,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run()
    print(f"\n{'='*50}")
    print(
        f"XGBoost | RMSE={result['rmse']:.2f}  MAE={result['mae']:.2f}  "
        f"MAPE={result['mape']:.1f}%  (n={result['n_test']})"
    )
