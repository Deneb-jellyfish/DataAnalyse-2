"""Prophet model for hourly PM2.5 prediction.

Uses Facebook Prophet for trend + seasonality decomposition.
If Prophet import fails, falls back to statsmodels STL decomposition
for a simple trend + seasonality model.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd


class ProphetHourlyModel:
    """Prophet-based hourly PM2.5 prediction model.

    Wraps Facebook Prophet with automatic fallback to statsmodels STL
    if Prophet is not available.

    Usage:
        model = ProphetHourlyModel()
        model.fit(df)               # df has columns: ds (datetime), y (pm25)
        preds = model.predict(ds_list)  # list/array of future datetimes
    """

    def __init__(self) -> None:
        self._prophet_model: Optional[object] = None
        self._fallback_stats: dict = {}
        self._use_fallback: bool = False

        # Fallback: decomposed components from statsmodels STL
        self._trend_interpolator = None
        self._seasonal_components: dict[str, np.ndarray] = {}
        self._seasonal_period: int = 24  # 24-hour daily seasonality
        self._weekly_period: int = 168  # 168-hour weekly seasonality
        self._train_end: Optional[pd.Timestamp] = None
        self._train_values: Optional[np.ndarray] = None
        self._train_times: Optional[np.ndarray] = None

    def _try_import_prophet(self) -> bool:
        """Attempt to import and create a Prophet model.

        Returns True if successful, False otherwise.
        """
        try:
            from prophet import Prophet  # type: ignore
            self._prophet_model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                seasonality_mode='additive',
            )
            # Add custom daily (period=1 day) and weekly (period=7 days) seasonalities
            # Periods are in days per Prophet convention
            self._prophet_model.add_seasonality(
                name='daily', period=1.0, fourier_order=5
            )
            self._prophet_model.add_seasonality(
                name='weekly', period=7.0, fourier_order=5
            )
            return True
        except ImportError:
            return False

    def fit(self, df: pd.DataFrame) -> dict:
        """Fit Prophet (or fallback) on hourly PM2.5 data.

        Parameters
        ----------
        df : pd.DataFrame
            Must have columns 'ds' (datetime) and 'y' (pm25 values).

        Returns
        -------
        dict
            Metadata dict with keys:
            - model: 'Prophet' or 'STL-Fallback'
            - n_train: number of training samples
            - train_start, train_end: date range
            - fit_time_s: fitting time in seconds
            - fallback_reason: reason for fallback (if applicable)
        """
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)

        n_train = len(df)
        train_start = df["ds"].iloc[0]
        train_end = df["ds"].iloc[-1]

        t0 = time.time()

        if self._try_import_prophet():
            # Prophet path
            self._use_fallback = False
            self._prophet_model.fit(df[["ds", "y"]])
            fit_time = time.time() - t0
            return {
                "model": "Prophet",
                "n_train": n_train,
                "train_start": str(train_start),
                "train_end": str(train_end),
                "fit_time_s": round(fit_time, 1),
                "fallback_reason": "",
            }
        else:
            # Fallback path using STL
            self._use_fallback = True
            result = self._fit_stl_fallback(df)
            result["fit_time_s"] = round(time.time() - t0, 1)
            return result

    def _fit_stl_fallback(self, df: pd.DataFrame) -> dict:
        """Fallback: decompose PM2.5 series using STL and extrapolate."""
        from statsmodels.tsa.seasonal import STL

        n_train = len(df)
        train_start = df["ds"].iloc[0]
        train_end = df["ds"].iloc[-1]

        self._train_end = train_end
        self._train_values = df["y"].values.astype(np.float64)
        self._train_times = df["ds"].values

        # Fit STL decomposition
        # STL requires periodic > 1 and < len(series)
        period = min(self._seasonal_period, n_train // 2)
        if period < 2:
            period = 2

        try:
            stl = STL(
                self._train_values,
                period=period,
                seasonal=7,  # 7 iterations, robust
                trend=13,    # Default trend filter
            )
            result = stl.fit()
            self._trend = result.trend
            self._seasonal = result.seasonal
            self._resid = result.resid

            # Extrapolate trend using linear regression on the last portion
            # Use the last 7 days (168 hours) for trend extrapolation
            trend_window = min(168, n_train)
            last_trend = self._trend[-trend_window:]
            x_trend = np.arange(len(last_trend)).reshape(-1, 1)
            y_trend = last_trend

            from sklearn.linear_model import LinearRegression
            self._trend_model = LinearRegression()
            self._trend_model.fit(x_trend, y_trend.nan_to_num(0.0))

            # Store seasonal pattern for extrapolation (24-hour cycle)
            if len(self._seasonal) >= self._seasonal_period:
                self._seasonal_pattern = self._seasonal[-self._seasonal_period:]
            else:
                self._seasonal_pattern = self._seasonal

            self._fallback_stats = {
                "trend_strength": float(
                    1.0 - np.nanvar(result.resid) / np.nanvar(result.trend + result.resid)
                    if np.nanvar(result.trend + result.resid) > 0 else 0
                ),
                "seasonal_strength": float(
                    1.0 - np.nanvar(result.resid) / np.nanvar(result.seasonal + result.resid)
                    if np.nanvar(result.seasonal + result.resid) > 0 else 0
                ),
            }
        except Exception:
            # Ultra-simple fallback: just use the mean of recent values
            self._trend = self._train_values.copy()
            self._seasonal = np.zeros_like(self._train_values)
            self._resid = np.zeros_like(self._train_values)
            self._seasonal_pattern = np.zeros(self._seasonal_period)
            from sklearn.linear_model import LinearRegression
            window = min(168, n_train)
            x_trend = np.arange(window).reshape(-1, 1)
            y_trend = self._train_values[-window:]
            self._trend_model = LinearRegression()
            self._trend_model.fit(x_trend, y_trend)

        return {
            "model": "STL-Fallback",
            "n_train": n_train,
            "train_start": str(train_start),
            "train_end": str(train_end),
            "fit_time_s": 0.0,
            "fallback_reason": "Prophet not available; using statsmodels STL decomposition",
        }

    def predict(self, future_ds: list) -> np.ndarray:
        """Generate PM2.5 predictions for a list of future datetimes.

        Parameters
        ----------
        future_ds : list
            List of pd.Timestamp or datetime-like strings for prediction.

        Returns
        -------
        np.ndarray
            Array of predicted PM2.5 values (yhat).
        """
        if self._use_fallback:
            return self._predict_fallback(future_ds)

        future_df = pd.DataFrame({"ds": pd.to_datetime(future_ds)})
        forecast = self._prophet_model.predict(future_df)
        return forecast["yhat"].values.astype(np.float64)

    def _predict_fallback(self, future_ds: list) -> np.ndarray:
        """Predict using the STL fallback decomposition."""
        future_times = pd.to_datetime(future_ds)

        if self._train_end is None or self._train_values is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        seasonal_period = len(self._seasonal_pattern)
        predictions = np.zeros(len(future_times), dtype=np.float64)

        for i, ft in enumerate(future_times):
            # Trend: extrapolate linearly from the last training point
            hours_ahead = (ft - self._train_end).total_seconds() / 3600.0
            hour_offset = max(0, hours_ahead)

            # Predict trend continuation
            trend_x = np.array([[len(self._trend) + hour_offset - 168]])
            trend_val = float(self._trend_model.predict(trend_x)[0])

            # Seasonal: cycle through the 24h pattern
            if seasonal_period > 0:
                hour_of_day = ft.hour
                # Find closest seasonal index
                # Map hour to index in seasonal_pattern
                # The seasonal_pattern[-24:] covers the last 24 hours of training
                # We need to figure out the mapping
                last_train_hour = self._train_end.hour
                target_hour = ft.hour
                # The seasonal pattern at index -24 corresponds to
                # (last_train_hour - 23) hour, and index -1 corresponds to
                # last_train_hour
                # So target_hour maps to pattern index based on offset
                hour_diff = target_hour - last_train_hour
                if hour_diff <= 0:
                    hour_diff += 24
                pattern_idx = (seasonal_period - 24 + hour_diff) % seasonal_period
                seasonal_val = float(self._seasonal_pattern[pattern_idx % seasonal_period])
                if np.isnan(seasonal_val):
                    seasonal_val = 0.0
            else:
                seasonal_val = 0.0

            predictions[i] = max(0.0, trend_val + seasonal_val)

        return predictions
