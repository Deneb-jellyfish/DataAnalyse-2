"""ARIMA model for hourly PM2.5 prediction.

Uses statsmodels.tsa.arima.model.ARIMA (not pmdarima).
"""

from __future__ import annotations

from datetime import datetime
import warnings
from typing import Optional

import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from statsmodels.tsa.arima.model import ARIMA


class ARIMAHourlyModel:
    """ARIMA wrapped for hourly PM2.5 prediction.

    Supports rolling one-step-ahead and rolling multi-step forecasting
    using statsmodels ARIMA.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (2, 1, 2),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 24),
    ):
        """Initialise ARIMA model.

        Parameters
        ----------
        order : (p, d, q)
            Non-seasonal ARIMA order.
        seasonal_order : (P, D, Q, s)
            Seasonal order; s=24 captures a daily cycle in hourly data.
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self._results = None
        self._train_length: int = 0

    def _log(self, message: str) -> None:
        """Print a lightweight timestamped progress message."""
        print(f"[ARIMA {datetime.now():%H:%M:%S}] {message}", flush=True)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, series: np.ndarray) -> dict:
        """Fit ARIMA on training PM2.5 time series.

        Parameters
        ----------
        series : 1D array of PM2.5 values in chronological order.

        Returns
        -------
        dict with keys: n_train, aic, bic, order, seasonal_order.
        """
        series = np.asarray(series, dtype=np.float64)
        valid = series[~np.isnan(series)]
        if len(valid) < len(series):
            series = valid.copy()

        self._train_length = len(series)
        self._log(
            f"start fit: n_train={self._train_length}, "
            f"order={self.order}, seasonal_order={self.seasonal_order}"
        )

        # If all seasonal terms are zero, skip seasonal_order to avoid
        # potential issues with s=0 in older statsmodels versions.
        seas = self.seasonal_order
        use_seasonal = not (seas[0] == 0 and seas[1] == 0 and seas[2] == 0 and seas[3] == 0)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            if use_seasonal:
                model = ARIMA(
                    series,
                    order=self.order,
                    seasonal_order=seas,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
            else:
                model = ARIMA(
                    series,
                    order=self.order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
            self._results = model.fit(method_kwargs={"maxiter": 500})

        self._log(
            f"fit complete: aic={self._results.aic:.2f}, bic={self._results.bic:.2f}"
        )

        return {
            "n_train": self._train_length,
            "aic": float(self._results.aic),
            "bic": float(self._results.bic),
            "order": list(self.order),
            "seasonal_order": list(self.seasonal_order),
        }

    # ------------------------------------------------------------------
    # +1h rolling one-step-ahead forecast
    # ------------------------------------------------------------------
    def predict_h1(self, test_series: np.ndarray, log_every: Optional[int] = None) -> np.ndarray:
        """Rolling one-step-ahead forecast on test series.

        At each step:
          - forecast(steps=1) yields the +1h prediction
          - append the TRUE observation, then advance

        Parameters
        ----------
        test_series : 1D array of true PM2.5 test values.

        Returns
        -------
        predictions : 1D array same length as test_series.
        """
        if self._results is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        test_series = np.asarray(test_series, dtype=np.float64)
        n_test = len(test_series)
        predictions = np.full(n_test, np.nan, dtype=np.float64)
        log_every = log_every or max(1, n_test // 10)
        self._log(f"start rolling h1 forecast: n_test={n_test}, log_every={log_every}")

        results = self._results

        for i in range(n_test):
            true_val = test_series[i]
            if np.isnan(true_val):
                predictions[i] = np.nan
                if (i + 1) % log_every == 0 or (i + 1) == n_test:
                    self._log(f"h1 progress: {i + 1}/{n_test} ({(i + 1) / n_test:.1%})")
                continue

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    fc = results.forecast(steps=1)
                predictions[i] = float(fc[0])
            except Exception:
                predictions[i] = np.nan

            # Append the true observation without re-estimating parameters
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    results = results.append([true_val], refit=False)
            except Exception:
                # If append fails, keep using the old results
                pass

            if (i + 1) % log_every == 0 or (i + 1) == n_test:
                self._log(f"h1 progress: {i + 1}/{n_test} ({(i + 1) / n_test:.1%})")

        self._log("h1 forecast complete")

        return predictions

    # ------------------------------------------------------------------
    # Rolling multi-step forecast
    # ------------------------------------------------------------------
    def predict_steps(
        self,
        test_series: np.ndarray,
        steps: int,
        log_every: Optional[int] = None,
    ) -> np.ndarray:
        """Rolling multi-step forecast on test series.

        At each forecast origin i:
          - forecast(steps=steps) and keep all forecast steps
          - append the TRUE observation at origin i, then advance

        Later horizons near the test-set tail cannot be evaluated, so they
        remain NaN.

        Parameters
        ----------
        test_series : 1D array of true PM2.5 test values.

        Returns
        -------
        predictions : 2D array of shape ``(len(test_series), steps)``.
        """
        if self._results is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if steps < 1:
            raise ValueError("steps must be at least 1")

        test_series = np.asarray(test_series, dtype=np.float64)
        n_test = len(test_series)
        predictions = np.full((n_test, steps), np.nan, dtype=np.float64)

        results = self._results
        max_i = n_test - steps + 1
        log_every = log_every or max(1, max_i // 10)
        self._log(
            f"start rolling seq forecast: n_test={n_test}, steps={steps}, "
            f"forecast_origins={max_i}, log_every={log_every}"
        )

        for i in range(max_i):
            true_val = test_series[i]
            if np.isnan(true_val):
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        results = results.append([np.nan_to_num(true_val, nan=0.0)], refit=False)
                except Exception:
                    pass
                if (i + 1) % log_every == 0 or (i + 1) == max_i:
                    self._log(
                        f"seq{steps} progress: {i + 1}/{max_i} ({(i + 1) / max_i:.1%})"
                    )
                continue

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    fc = results.forecast(steps=steps)
                predictions[i, :steps] = np.asarray(fc, dtype=np.float64)
            except Exception:
                predictions[i, :steps] = np.nan

            # Append the true observation at origin i
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    results = results.append([true_val], refit=False)
            except Exception:
                pass

            if (i + 1) % log_every == 0 or (i + 1) == max_i:
                self._log(f"seq{steps} progress: {i + 1}/{max_i} ({(i + 1) / max_i:.1%})")

        self._log(f"seq{steps} forecast complete")

        return predictions

    def predict_h12(self, test_series: np.ndarray) -> np.ndarray:
        """Compatibility wrapper that returns only the +12h step."""
        return self.predict_steps(test_series, steps=12)[:, -1]

    # ------------------------------------------------------------------
    # Convenience: full evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        test_series: np.ndarray,
        horizon: int = 1,
    ) -> np.ndarray:
        """Dispatch to predict_h1 or predict_h12 based on horizon."""
        if horizon == 1:
            return self.predict_h1(test_series)
        elif horizon > 1:
            return self.predict_steps(test_series, steps=horizon)[:, horizon - 1]
        raise ValueError(f"horizon must be a positive integer, got {horizon}")
