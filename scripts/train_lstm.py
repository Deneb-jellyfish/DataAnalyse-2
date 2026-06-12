"""Train LSTM for hourly PM2.5 prediction tasks (+1h and +24h).

Usage:
  python scripts/train_lstm.py --task h1
  python scripts/train_lstm.py --task h24
  python scripts/train_lstm.py --task both
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.lstm_model import LSTMModel, LSTMTrainer
from evaluation.metrics import compute_metrics
from utils.feature_engineering import build_hourly_feature_frame, StandardScaler
from utils.io import PROCESSED_DIR

OUTPUT_DIR = ROOT / "outputs" / "hourly"
FEATURES_DIR = PROCESSED_DIR / "features_hourly"

torch.set_num_threads(4)


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------
def create_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window inputs and targets.

    Parameters
    ----------
    X : np.ndarray, shape (n_timesteps, n_features)
        Chronologically ordered feature matrix.
    y : np.ndarray, shape (n_timesteps,)
        Raw PM2.5 at each timestep (NOT pre-shifted).
    seq_len : int
        Number of past timesteps per input window.
    horizon : int
        Steps ahead to predict relative to the *end* of the input window.

    Returns
    -------
    X_seq : np.ndarray, shape (n_windows, seq_len, n_features), float32
    y_seq : np.ndarray, shape (n_windows,), float32
    """
    n = len(X)
    assert n == len(y), f"X and y length mismatch: {n} vs {len(y)}"
    n_windows = n - seq_len - horizon + 1
    if n_windows <= 0:
        raise ValueError(
            f"Not enough samples for seq_len={seq_len}, horizon={horizon} "
            f"(n={n})"
        )

    X_seq = np.zeros((n_windows, seq_len, X.shape[1]), dtype=np.float32)
    y_seq = np.zeros(n_windows, dtype=np.float32)

    for i in range(n_windows):
        X_seq[i] = X[i : i + seq_len]
        y_seq[i] = y[i + seq_len + horizon - 1]

    return X_seq, y_seq


def forecast_dates_for_sequences(
    dates: np.ndarray, seq_len: int, horizon: int
) -> np.ndarray:
    """Extract *forecast_origin_time* datetimes for each window.

    Returns the datetime of the last input timestep in each sliding window.
    target_time = forecast_origin_time + horizon hours.
    """
    n = len(dates)
    n_windows = n - seq_len - horizon + 1
    return dates[seq_len - 1 : seq_len - 1 + n_windows]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_h1_data():
    """Load pre-built +1h matrices from ``features_hourly/`` and build sequences.

    y is the *raw* PM2.5; ``create_sequences(..., horizon=1)`` handles the
    forward shift.
    """
    X_train = np.load(FEATURES_DIR / "X_train.npy")
    X_test = np.load(FEATURES_DIR / "X_test.npy")
    y_train = np.load(FEATURES_DIR / "y_train.npy")
    y_test = np.load(FEATURES_DIR / "y_test.npy")
    dates_train = np.load(FEATURES_DIR / "dates_train.npy", allow_pickle=True)
    dates_test = np.load(FEATURES_DIR / "dates_test.npy", allow_pickle=True)
    with open(FEATURES_DIR / "feature_names.json") as f:
        feature_names = json.load(f)

    seq_len = 48
    horizon = 1

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len, horizon)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_len, horizon)

    fct_train = forecast_dates_for_sequences(dates_train, seq_len, horizon)
    fct_test = forecast_dates_for_sequences(dates_test, seq_len, horizon)

    return (
        X_train_seq, X_test_seq,
        y_train_seq, y_test_seq,
        fct_train, fct_test,
        feature_names, seq_len, horizon,
    )


def load_h24_data():
    """Build +24h data from ``beijing_hourly.csv`` and create sequences.

    Uses raw PM2.5 as target; ``create_sequences(..., horizon=24)`` shifts
    the target 24 hours forward.
    """
    df = pd.read_csv(PROCESSED_DIR / "beijing_hourly.csv")
    frame, feature_names = build_hourly_feature_frame(df)

    # Keep rows with complete features AND raw pm25
    valid = frame.dropna(subset=feature_names + ["pm25"]).reset_index(drop=True)
    print(f"  h24 valid rows: {len(valid)} (dropped {len(frame) - len(valid)})")

    # Chronological 80 / 20 split
    split_idx = int(len(valid) * 0.8)
    train = valid.iloc[:split_idx]
    test = valid.iloc[split_idx:]

    # Scale (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_names].values.astype(np.float64))
    X_test = scaler.transform(test[feature_names].values.astype(np.float64))
    y_train_raw = train["pm25"].values.astype(np.float64)
    y_test_raw = test["pm25"].values.astype(np.float64)
    dates_train = train["datetime"].values
    dates_test = test["datetime"].values

    seq_len = 48
    horizon = 24

    X_train_seq, y_train_seq = create_sequences(
        X_train, y_train_raw, seq_len, horizon,
    )
    X_test_seq, y_test_seq = create_sequences(
        X_test, y_test_raw, seq_len, horizon,
    )

    fct_train = forecast_dates_for_sequences(dates_train, seq_len, horizon)
    fct_test = forecast_dates_for_sequences(dates_test, seq_len, horizon)

    print(
        f"  h24 train seqs: {len(X_train_seq)}, test seqs: {len(X_test_seq)}"
    )

    return (
        X_train_seq, X_test_seq,
        y_train_seq, y_test_seq,
        fct_train, fct_test,
        feature_names, seq_len, horizon,
    )


# ---------------------------------------------------------------------------
# Prediction CSV
# ---------------------------------------------------------------------------
def make_prediction_csv(
    forecast_times: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    model_name: str,
    city: str = "Beijing",
    split_label: str = "test",
) -> pd.DataFrame:
    """Build predictions DataFrame in the unified output format."""
    forecast_dt = pd.to_datetime(forecast_times)
    target_dt = forecast_dt + pd.Timedelta(hours=int(horizon))

    return pd.DataFrame({
        "forecast_origin_time": forecast_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "target_time": target_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "horizon_hours": int(horizon),
        "city": city,
        "model": model_name,
        "y_true": y_true,
        "y_pred": y_pred,
        "split": split_label,
    })


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------
def run_task(task: str) -> tuple[dict, dict]:
    """Train LSTM for one task and return (metrics_row, train_history)."""
    label = task.upper().replace("H", "+") + "h"
    print(f"\n{'=' * 60}")
    print(f"LSTM {label}")
    print("=" * 60)

    if task == "h1":
        (
            X_train, X_test,
            y_train, y_test,
            fct_train, fct_test,
            feature_names, seq_len, horizon,
        ) = load_h1_data()
        # Optimized params for +1h
        hidden_dim = 128
        num_layers = 3
        dropout = 0.3
        lr = 3e-4
        batch_size = 32
        patience = 25
        epochs = 150
    else:
        (
            X_train, X_test,
            y_train, y_test,
            fct_train, fct_test,
            feature_names, seq_len, horizon,
        ) = load_h24_data()
        # Light params for +24h
        hidden_dim = 64
        num_layers = 2
        dropout = 0.2
        lr = 1e-3
        batch_size = 64
        patience = 15
        epochs = 100

    # Chronological 80 / 20 train / val split on training sequences
    val_split = int(len(X_train) * 0.8)
    X_tr, X_va = X_train[:val_split], X_train[val_split:]
    y_tr, y_va = y_train[:val_split], y_train[val_split:]

    print(
        f"  Train: {len(X_tr)}, Val: {len(X_va)}, Test: {len(X_test)}"
    )
    print(
        f"  Features: {len(feature_names)}, "
        f"Seq len: {seq_len}, Horizon: {horizon}h"
    )

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=batch_size, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=batch_size, shuffle=False,
    )

    # Model
    model = LSTMModel(
        input_dim=X_train.shape[2],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    trainer = LSTMTrainer(model, lr=lr)

    t0 = time.time()
    history = trainer.train(train_loader, val_loader, epochs=epochs, patience=patience)
    train_time = time.time() - t0
    print(f"  Train time: {train_time:.1f}s")

    # Evaluate on test set
    y_pred = trainer.predict(test_loader)
    metrics = compute_metrics(y_test, y_pred)
    print(
        f"  Test RMSE={metrics['rmse']:.2f}, "
        f"MAE={metrics['mae']:.2f}, "
        f"MAPE={metrics['mape']:.1f}%"
    )

    # Save prediction CSV
    pred_df = make_prediction_csv(
        fct_test, y_test, y_pred,
        horizon=horizon, model_name="LSTM",
    )
    pred_path = OUTPUT_DIR / f"lstm_predictions_{task}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"  Predictions saved: {pred_path}")

    # Save model checkpoint
    trainer.save(OUTPUT_DIR / f"lstm_{task}.pkl")
    print(f"  Model saved: {OUTPUT_DIR / f'lstm_{task}.pkl'}")

    row = {
        "model": "LSTM",
        "horizon_hours": int(horizon),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mape": metrics["mape"],
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "n_test": int(len(X_test)),
        "best_val_loss": float(trainer.best_val_loss),
        "train_time_s": round(train_time, 1),
    }
    return row, history


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_metrics(rows: list[dict]) -> None:
    """Write ``outputs/hourly/lstm_metrics.csv``."""
    path = OUTPUT_DIR / "lstm_metrics.csv"
    df = pd.DataFrame(rows)
    columns = [
        "model", "horizon_hours", "rmse", "mae", "mape",
        "n_train", "n_val", "n_test", "best_val_loss", "train_time_s",
    ]
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    df = df[columns]
    df.to_csv(path, index=False)
    print(f"\nMetrics saved: {path}")


def save_train_history(all_histories: dict[str, dict]) -> None:
    """Write ``outputs/hourly/lstm_train_history.json``."""
    path = OUTPUT_DIR / "lstm_train_history.json"
    serializable: dict[str, dict] = {}
    for task, hist in all_histories.items():
        serializable[task] = {
            "train_loss": [float(v) for v in hist.get("train_loss", [])],
            "val_loss": [float(v) for v in hist.get("val_loss", [])],
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"Training history saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM for hourly PM2.5")
    parser.add_argument(
        "--task", default="both",
        choices=["h1", "h24", "both"],
        help="Prediction horizon task (default: both)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    all_rows: list[dict] = []
    all_histories: dict[str, dict] = {}

    if args.task in ("h1", "both"):
        row, history = run_task("h1")
        all_rows.append(row)
        all_histories["h1"] = history

    if args.task in ("h24", "both"):
        row, history = run_task("h24")
        all_rows.append(row)
        all_histories["h24"] = history

    save_metrics(all_rows)
    save_train_history(all_histories)

    print("\n" + "=" * 60)
    print("LSTM Summary")
    print("=" * 60)
    for r in all_rows:
        print(
            f"  +{r['horizon_hours']}h: "
            f"RMSE={r['rmse']:.2f}, "
            f"MAE={r['mae']:.2f}, "
            f"MAPE={r['mape']:.1f}%"
        )


if __name__ == "__main__":
    main()
