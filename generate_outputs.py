"""Generate predictions.csv, experiment_log.csv and run DM tests.

Single entry-point: re-runs all models, collects predictions, computes
pairwise DM statistics, and writes final deliverables for Member C.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluation.dm_test import pairwise_dm
from evaluation.metrics import compute_metrics, aqi_level

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
def collect() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Run every model and gather predictions + metadata."""
    import importlib

    all_rows: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}

    tasks = [
        ("models.arima_model", "ARIMA"),
        ("models.prophet_model", "Prophet"),
        ("models.xgboost_model", "XGBoost"),
        ("models.lstm_model", "LSTM"),
        ("models.transformer_model", "Informer"),
        ("models.transformer_standard", "Transformer"),
    ]

    for mod_name, model_name in tasks:
        log.info("running %s ...", model_name)
        mod = importlib.import_module(mod_name)
        result = mod.run()
        all_rows.extend(result["predictions"])
        metadata[model_name] = {
            k: v for k, v in result.items() if k != "predictions"
        }
        log.info("  %s RMSE=%.2f MAE=%.2f n=%d",
                 model_name, result["rmse"], result["mae"],
                 len(result["predictions"]))

    return all_rows, metadata


# ---------------------------------------------------------------------------
def compute_dm(all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Run pairwise DM tests on the *common* test dates of all models."""
    df = pd.DataFrame(all_rows)

    # find the common date set where all 6 models have predictions
    date_sets = [set(df[df["model"] == m]["date"]) for m in df["model"].unique()]
    common = date_sets[0]
    for ds in date_sets[1:]:
        common = common & ds
    common = sorted(common)
    log.info("DM test on %d common dates", len(common))

    # extract errors
    model_errors: dict[str, np.ndarray] = {}
    for m in sorted(df["model"].unique()):
        sub = df[(df["model"] == m) & (df["date"].isin(common))]
        sub = sub.sort_values("date")
        model_errors[m] = (sub["y_true"].values - sub["y_pred"].values).astype(np.float64)

    return pairwise_dm(model_errors)


# ---------------------------------------------------------------------------
def save_predictions_csv(all_rows: list[dict[str, Any]], path: Path) -> None:
    """Write predictions.csv in the format expected by Member C."""
    df = pd.DataFrame(all_rows)
    df = df.sort_values(["model", "date"])
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log.info("predictions.csv: %d rows → %s", len(df), path)


def save_experiment_log(
    metadata: dict[str, dict[str, Any]],
    dm_result: dict[str, Any],
    path: Path,
) -> None:
    """Write experiment_log.csv with one row per model experiment."""
    rows = []
    for model_name, meta in metadata.items():
        rows.append({
            "model": model_name,
            "rmse": meta.get("rmse"),
            "mae": meta.get("mae"),
            "mape": meta.get("mape"),
            "n_train": meta.get("n_train"),
            "n_test": meta.get("n_test"),
            "params": json.dumps(meta, default=str),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log.info("experiment_log.csv: %d models → %s", len(rows), path)


# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    log.info("=" * 60)
    log.info("AQI Project — Member B Final Output Generation")
    log.info("=" * 60)

    # 1. collect all predictions
    log.info("\n[1/4] Running all models ...")
    all_rows, metadata = collect()

    # 2. save predictions.csv
    log.info("\n[2/4] Saving predictions.csv ...")
    save_predictions_csv(all_rows, ROOT / "predictions.csv")

    # 3. DM tests
    log.info("\n[3/4] Computing Diebold-Mariano tests ...")
    dm_result = compute_dm(all_rows)

    # print DM matrix
    log.info("\nPairwise DM test results:")
    for pair in dm_result["pairs"]:
        sig = pair["significant"]
        if sig:
            log.info(
                "  %-14s vs %-14s  DM=%-7.2f p=%.4f %s  (%s better)",
                pair["model1"], pair["model2"],
                pair["dm_stat"], pair["p_value"], sig, pair["better"],
            )

    # 4. experiment log
    log.info("\n[4/4] Saving experiment_log.csv ...")
    save_experiment_log(metadata, dm_result, ROOT / "docs" / "experiment_log.csv")

    # final summary
    log.info("\n" + "=" * 60)
    log.info("Deliverables ready:")
    log.info("  predictions.csv       — %d rows", len(all_rows))
    log.info("  docs/experiment_log.csv")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
