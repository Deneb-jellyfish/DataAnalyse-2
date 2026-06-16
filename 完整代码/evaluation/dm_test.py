"""Diebold-Mariano test for pairwise forecast comparison."""

from __future__ import annotations

import numpy as np
from scipy import stats


def dm_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    h: int = 1,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Diebold-Mariano test for equal predictive accuracy.

    H0: E[L(e_a)] = E[L(e_b)]  (equal predictive accuracy)
    H1: models differ in predictive accuracy

    Parameters
    ----------
    errors_a, errors_b : ndarray
        Forecast errors from two models (same length).
    h : int
        Forecast horizon for autocorrelation correction.
        Uses h-1 Newey-West truncation lag.
    alternative : str
        'two-sided', 'less' (a better than b), 'greater' (b better than a).

    Returns
    -------
    dict with keys: dm_statistic, p_value, horizon.
    """
    errors_a = np.asarray(errors_a, dtype=np.float64)
    errors_b = np.asarray(errors_b, dtype=np.float64)

    # Remove NaN
    mask = ~(np.isnan(errors_a) | np.isnan(errors_b))
    errors_a = errors_a[mask]
    errors_b = errors_b[mask]
    n = len(errors_a)

    if n < 2:
        return {"dm_statistic": np.nan, "p_value": np.nan, "horizon": h, "n": n}

    # Squared error loss differential
    d = errors_a**2 - errors_b**2
    d_mean = np.mean(d)

    # Long-run variance estimation with Newey-West (truncation lag = h-1)
    max_lag = max(0, h - 1)
    gamma0 = np.var(d, ddof=0)
    long_run_var = gamma0
    for lag in range(1, min(max_lag + 1, n - 1)):
        autocov = np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean))
        weight = 1.0 - lag / (max_lag + 1)
        long_run_var += 2 * weight * autocov

    long_run_var = max(long_run_var, 1e-12)
    dm_stat = d_mean / np.sqrt(long_run_var / n)

    # p-value
    if alternative == "two-sided":
        p_value = 2 * stats.norm.sf(np.abs(dm_stat))
    elif alternative == "less":
        p_value = stats.norm.cdf(dm_stat)
    elif alternative == "greater":
        p_value = stats.norm.sf(dm_stat)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return {
        "dm_statistic": float(dm_stat),
        "p_value": float(p_value),
        "horizon": h,
        "n": n,
    }
