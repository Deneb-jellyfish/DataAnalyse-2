"""XGBoost model for hourly PM2.5 prediction (+1h and +24h)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb


class XGBoostHourlyModel:
    """XGBoost regressor wrapped for hourly PM2.5 prediction.

    Supports fit/predict/feature_importance/save/load.
    """

    def __init__(self, params: Optional[dict] = None):
        self.params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": "cpu",
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 50,
        }
        if params:
            self.params.update(params)

        self._model: Optional[xgb.XGBRegressor] = None
        self._feature_names: list[str] = []
        self._best_iteration: int = 0
        self._best_score: float = 0.0

    @property
    def model(self) -> xgb.XGBRegressor:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
    ) -> dict:
        """Fit XGBoost with optional early stopping on validation set.

        Returns training metadata dict.
        """
        self._feature_names = feature_names or [
            f"f{i}" for i in range(X_train.shape[1])
        ]

        fit_params = {k: v for k, v in self.params.items()
                      if k not in ("early_stopping_rounds", "n_estimators")}

        model = xgb.XGBRegressor(
            n_estimators=self.params["n_estimators"],
            early_stopping_rounds=self.params["early_stopping_rounds"],
            **fit_params,
        )

        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train, verbose=False)

        self._model = model
        self._best_iteration = model.best_iteration if model.best_iteration else self.params["n_estimators"]
        self._best_score = model.best_score if model.best_score else 0.0

        return {
            "best_iteration": self._best_iteration,
            "best_score": float(self._best_score),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1-D array of predictions."""
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return DataFrame with feature importance (gain-based)."""
        booster = self.model.get_booster()
        importance = booster.get_score(importance_type="gain")

        records = []
        for i, name in enumerate(self._feature_names):
            records.append({
                "feature": name,
                "gain": importance.get(f"f{i}", 0.0),
            })

        df = pd.DataFrame(records)
        df = df.sort_values("gain", ascending=False).reset_index(drop=True)
        # Normalize
        total = df["gain"].sum()
        if total > 0:
            df["gain_norm"] = df["gain"] / total
        else:
            df["gain_norm"] = 0.0
        return df

    def save(self, path: str | Path) -> None:
        """Pickle the model and metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self._model,
            "feature_names": self._feature_names,
            "params": self.params,
            "best_iteration": self._best_iteration,
            "best_score": self._best_score,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostHourlyModel":
        """Load a pickled model."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        obj = cls(params=state["params"])
        obj._model = state["model"]
        obj._feature_names = state["feature_names"]
        obj._best_iteration = state["best_iteration"]
        obj._best_score = state["best_score"]
        return obj
