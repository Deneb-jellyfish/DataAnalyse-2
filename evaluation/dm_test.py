"""Diebold-Mariano test for pairwise model comparison (B-8)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def dm_test(
    errors1: np.ndarray,
    errors2: np.ndarray,
    h: int = 1,
    alternative: str = "two_sided",
) -> dict[str, float]:
    """Diebold-Mariano test for equal predictive accuracy.

    Parameters
    ----------
    errors1, errors2 : 1-D arrays of prediction errors (y_true - y_pred)
    h : forecast horizon (1 for 1-step-ahead)
    alternative : 'two_sided', 'greater' (model 2 worse), 'less' (model 2 better)

    Returns
    -------
    dict with ``dm_stat`` and ``p_value``.
    """
    d = errors1 ** 2 - errors2 ** 2  # squared-error loss differential
    n = len(d)

    # long-run variance (Newey-West style, simple for h=1)
    if h == 1:
        var_d = np.var(d, ddof=1) / n
    else:
        # Bartlett kernel for h > 1
        gamma = [np.mean((d[h:] - d[:-h].mean()) * (d[:-h] - d[:-h].mean()))]
        for lag in range(1, h):
            gamma.append(
                np.mean(
                    (d[lag:] - d[lag:].mean()) * (d[:-lag] - d[:-lag].mean())
                )
            )
        var_d = (gamma[0] + 2 * sum(gamma[1:])) / n

    dm_stat = float(np.mean(d) / np.sqrt(max(var_d, 1e-12)))

    if alternative == "two_sided":
        p_value = 2 * stats.norm.sf(abs(dm_stat))
    elif alternative == "greater":
        p_value = stats.norm.sf(dm_stat)
    else:  # less
        p_value = stats.norm.cdf(dm_stat)

    return {"dm_stat": dm_stat, "p_value": float(p_value)}


def pairwise_dm(
    model_errors: dict[str, np.ndarray],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Run pairwise DM tests on all model pairs.

    ``model_errors`` maps model name → array of errors on the common test set.
    """
    names = sorted(model_errors.keys())
    results: list[dict[str, Any]] = []

    for i, m1 in enumerate(names):
        for m2 in names[i + 1:]:
            # align on common test points
            n = min(len(model_errors[m1]), len(model_errors[m2]))
            e1, e2 = model_errors[m1][:n], model_errors[m2][:n]
            res = dm_test(e1, e2)
            sig = "**" if res["p_value"] < 0.01 else ("*" if res["p_value"] < alpha else "")
            results.append({
                "model1": m1, "model2": m2,
                "dm_stat": res["dm_stat"],
                "p_value": res["p_value"],
                "significant": sig,
                "better": m1 if res["dm_stat"] < 0 and res["p_value"] < alpha else
                          (m2 if res["dm_stat"] > 0 and res["p_value"] < alpha else "none"),
            })

    return {"pairs": results, "alpha": alpha, "models": names}


if __name__ == "__main__":
    # quick smoke test
    np.random.seed(42)
    e1 = np.random.randn(100)
    e2 = np.random.randn(100) * 1.2
    res = dm_test(e1, e2)
    print(f"DM smoke test: stat={res['dm_stat']:.3f}  p={res['p_value']:.4f}")
