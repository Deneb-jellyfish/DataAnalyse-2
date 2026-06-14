"""Train LSTM for hourly PM2.5 h1 and seq6 tasks."""

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
from utils.hourly_experiment import (
    make_multi_horizon_prediction_frame,
    make_prediction_frame,
    metric_rows_from_predictions,
)
from utils.io import FEATURES_HOURLY_DIR

OUTPUT_DIR = ROOT / "outputs" / "hourly"
FEATURES_DIR = FEATURES_HOURLY_DIR

torch.set_num_threads(4)
np.random.seed(42)
torch.manual_seed(42)


def load_base_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load the standardized hourly feature matrices."""
    X_train = np.load(FEATURES_DIR / "X_train.npy")
    X_test = np.load(FEATURES_DIR / "X_test.npy")
    y_train = np.load(FEATURES_DIR / "y_train.npy")
    y_test = np.load(FEATURES_DIR / "y_test.npy")
    dates_train = np.load(FEATURES_DIR / "dates_train.npy", allow_pickle=True)
    dates_test = np.load(FEATURES_DIR / "dates_test.npy", allow_pickle=True)
    with open(FEATURES_DIR / "feature_names.json", encoding="utf-8") as handle:
        feature_names = json.load(handle)
    return X_train, X_test, y_train, y_test, dates_train, dates_test, feature_names


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int, output_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Create single-output or multi-output sequence windows."""
    n_windows = len(X) - seq_len - output_len + 1
    if n_windows <= 0:
        raise ValueError(f"Not enough samples for seq_len={seq_len}, output_len={output_len}")

    X_seq = np.zeros((n_windows, seq_len, X.shape[1]), dtype=np.float32)
    if output_len == 1:
        y_seq = np.zeros(n_windows, dtype=np.float32)
    else:
        y_seq = np.zeros((n_windows, output_len), dtype=np.float32)

    for index in range(n_windows):
        X_seq[index] = X[index : index + seq_len]
        target_slice = y[index + seq_len : index + seq_len + output_len]
        if output_len == 1:
            y_seq[index] = target_slice[0]
        else:
            y_seq[index] = target_slice
    return X_seq, y_seq


def forecast_dates_for_sequences(dates: np.ndarray, seq_len: int, output_len: int) -> np.ndarray:
    """Return the forecast-origin timestamps aligned with generated windows."""
    n_windows = len(dates) - seq_len - output_len + 1
    return dates[seq_len - 1 : seq_len - 1 + n_windows]


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def run_task(task: str, profile: str = "full") -> tuple[list[dict], dict]:
    """Train one task and write its prediction/config artifacts."""
    X_train, X_test, y_train, y_test, _, dates_test, _ = load_base_data()
    if task == "h1":
        seq_len, output_len = 48, 1
        if profile == "handoff":
            hidden_dim, num_layers, dropout, lr = 96, 2, 0.2, 5e-4
            batch_size, patience, epochs = 64, 8, 40
        else:
            hidden_dim, num_layers, dropout, lr = 128, 3, 0.3, 3e-4
            batch_size, patience, epochs = 32, 20, 120
    else:
        seq_len, output_len = 72, 6
        if profile == "handoff":
            hidden_dim, num_layers, dropout, lr = 96, 2, 0.2, 5e-4
            batch_size, patience, epochs = 64, 8, 50
        else:
            hidden_dim, num_layers, dropout, lr = 128, 3, 0.3, 3e-4
            batch_size, patience, epochs = 32, 20, 140

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len, output_len)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_len, output_len)
    fct_test = forecast_dates_for_sequences(dates_test, seq_len, output_len)

    val_split = int(len(X_train_seq) * 0.8)
    X_tr, X_va = X_train_seq[:val_split], X_train_seq[val_split:]
    y_tr, y_va = y_train_seq[:val_split], y_train_seq[val_split:]

    train_loader = _make_loader(X_tr, y_tr, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(X_va, y_va, batch_size=batch_size, shuffle=False)
    test_loader = _make_loader(X_test_seq, y_test_seq, batch_size=batch_size, shuffle=False)

    model = LSTMModel(
        input_dim=X_train_seq.shape[2],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        output_dim=output_len,
    )
    trainer = LSTMTrainer(model, lr=lr)

    t0 = time.time()
    history = trainer.train(train_loader, val_loader, epochs=epochs, patience=patience)
    train_time = time.time() - t0

    y_pred = trainer.predict(test_loader)
    if output_len == 1:
        pred_df = make_prediction_frame(
            forecast_times=fct_test,
            y_true=y_test_seq,
            y_pred=y_pred,
            horizon=1,
            model_name="LSTM",
            task="h1",
        )
        pred_path = OUTPUT_DIR / "lstm_predictions_h1.csv"
        config_path = OUTPUT_DIR / "lstm_h1_config.json"
        model_path = OUTPUT_DIR / "lstm_h1.pkl"
        task_name = "h1"
    else:
        pred_df = make_multi_horizon_prediction_frame(
            forecast_times=fct_test,
            y_true=y_test_seq,
            y_pred=y_pred,
            horizons=range(1, 7),
            model_name="LSTM",
            task="seq6",
        )
        pred_path = OUTPUT_DIR / "lstm_predictions_seq6.csv"
        config_path = OUTPUT_DIR / "lstm_seq6_config.json"
        model_path = OUTPUT_DIR / "lstm_seq6.pkl"
        task_name = "seq6"
    pred_df.to_csv(pred_path, index=False)
    trainer.save(model_path)

    config = {
        "task": task_name,
        "output_len": int(output_len),
        "seq_len": int(seq_len),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "patience": int(patience),
        "epochs": int(epochs),
        "profile": profile,
        "train_time_s": round(train_time, 1),
        "best_val_loss": float(trainer.best_val_loss),
    }
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    notes = f"seq_len={seq_len},output_len={output_len},best_val_loss={float(trainer.best_val_loss):.4f},train_time_s={train_time:.1f}"
    rows = metric_rows_from_predictions(pred_df, model_name="LSTM", task=task_name, notes=notes)
    return rows, history


def save_metrics(rows: list[dict]) -> None:
    """Save combined metrics to CSV."""
    path = OUTPUT_DIR / "lstm_metrics.csv"
    df = pd.DataFrame(rows)
    cols = ["model", "task", "horizon_hours", "rmse", "mae", "mape", "n_samples", "notes"]
    for column in cols:
        if column not in df.columns:
            df[column] = ""
    df = df[cols].sort_values(["task", "horizon_hours"]).reset_index(drop=True)
    df.to_csv(path, index=False)


def save_train_history(all_histories: dict[str, dict]) -> None:
    """Persist per-task training curves."""
    path = OUTPUT_DIR / "lstm_train_history.json"
    serializable = {
        task: {
            "train_loss": [float(v) for v in hist.get("train_loss", [])],
            "val_loss": [float(v) for v in hist.get("val_loss", [])],
        }
        for task, hist in all_histories.items()
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM for hourly PM2.5")
    parser.add_argument("--task", default="both", choices=["h1", "seq6", "both"])
    parser.add_argument("--profile", default="full", choices=["full", "handoff"])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_histories: dict[str, dict] = {}
    if args.task in ("h1", "both"):
        rows, history = run_task("h1", profile=args.profile)
        all_rows.extend(rows)
        all_histories["h1"] = history
    if args.task in ("seq6", "both"):
        rows, history = run_task("seq6", profile=args.profile)
        all_rows.extend(rows)
        all_histories["seq6"] = history

    save_metrics(all_rows)
    save_train_history(all_histories)


if __name__ == "__main__":
    main()
