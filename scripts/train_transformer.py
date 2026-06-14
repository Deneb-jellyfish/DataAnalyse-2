"""Train Transformer for hourly PM2.5 h1 and seq6 tasks."""

from __future__ import annotations

import argparse
from datetime import datetime
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

from models.transformer_model import TimeSeriesTransformer, TransformerTrainer
from utils.hourly_experiment import (
    make_multi_horizon_prediction_frame,
    make_prediction_frame,
    metric_rows_from_predictions,
)
from utils.io import FEATURES_HOURLY_DIR

OUTPUT_DIR = ROOT / "outputs" / "hourly"
FEATURES_DIR = FEATURES_HOURLY_DIR

np.random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(4)


def log(message: str) -> None:
    """Print timestamped script progress messages."""
    print(f"[train_transformer {datetime.now():%H:%M:%S}] {message}", flush=True)


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
    """Create single-output or multi-output windows."""
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


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def _forecast_dates(dates: np.ndarray, seq_len: int, output_len: int) -> np.ndarray:
    n_windows = len(dates) - seq_len - output_len + 1
    return dates[seq_len - 1 : seq_len - 1 + n_windows]


def run_task(task: str, batch_size: int, epochs: int, patience: int, lr: float) -> tuple[list[dict], TransformerTrainer]:
    """Train one task and write its prediction/config artifacts."""
    log(
        f"start task={task}: batch_size={batch_size}, epochs={epochs}, "
        f"patience={patience}, lr={lr}"
    )
    X_train, X_test, y_train, y_test, dates_train, dates_test, feature_names = load_base_data()
    if task == "h1":
        seq_len, output_len = 48, 1
    else:
        seq_len, output_len = 72, 6
    log(
        f"loaded base data for task={task}: "
        f"X_train={X_train.shape}, X_test={X_test.shape}, "
        f"y_train={y_train.shape}, y_test={y_test.shape}"
    )

    val_split = int(len(X_train) * 0.8)
    X_tr, X_va = X_train[:val_split], X_train[val_split:]
    y_tr, y_va = y_train[:val_split], y_train[val_split:]
    dates_va = dates_train[val_split:]

    X_tr_seq, y_tr_seq = create_sequences(X_tr, y_tr, seq_len, output_len)
    X_va_seq, y_va_seq = create_sequences(X_va, y_va, seq_len, output_len)
    X_te_seq, y_te_seq = create_sequences(X_test, y_test, seq_len, output_len)
    log(
        f"sequence windows for task={task}: "
        f"train={X_tr_seq.shape}, val={X_va_seq.shape}, test={X_te_seq.shape}"
    )

    forecast_va = _forecast_dates(dates_va, seq_len, output_len)
    forecast_te = _forecast_dates(dates_test, seq_len, output_len)

    train_loader = _make_loader(X_tr_seq, y_tr_seq, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(X_va_seq, y_va_seq, batch_size=batch_size, shuffle=False)
    test_loader = _make_loader(X_te_seq, y_te_seq, batch_size=batch_size, shuffle=False)

    model = TimeSeriesTransformer(
        input_dim=len(feature_names),
        seq_len=seq_len,
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        dim_feedforward=128,
        dropout=0.1,
        output_dim=output_len,
    )
    trainer = TransformerTrainer(model, lr=lr)

    t0 = time.time()
    log(f"begin trainer.train for task={task}")
    train_result = trainer.train(train_loader, val_loader, epochs=epochs, patience=patience, verbose=True)
    train_time = time.time() - t0
    log(
        f"finished trainer.train for task={task}: "
        f"best_epoch={train_result['best_epoch']}, "
        f"best_val_loss={train_result['best_val_loss']:.4f}, "
        f"train_time_s={train_time:.1f}"
    )

    log(f"start test prediction for task={task}")
    y_pred = trainer.predict(test_loader)
    if output_len == 1:
        pred_df = make_prediction_frame(
            forecast_times=forecast_te,
            y_true=y_te_seq,
            y_pred=y_pred,
            horizon=1,
            model_name="Transformer",
            task="h1",
        )
        val_df = make_prediction_frame(
            forecast_times=forecast_va,
            y_true=y_va_seq,
            y_pred=trainer.predict(val_loader),
            horizon=1,
            model_name="Transformer",
            task="h1",
            split="val",
        )
        pred_df.to_csv(OUTPUT_DIR / "transformer_predictions_h1.csv", index=False)
        val_df.to_csv(OUTPUT_DIR / "transformer_predictions_h1_val.csv", index=False)
        config_path = OUTPUT_DIR / "transformer_h1_config.json"
        model_path = OUTPUT_DIR / "transformer_h1.pt"
        task_name = "h1"
        log(
            f"saved prediction files for task={task}: "
            f"test_rows={len(pred_df)}, val_rows={len(val_df)}"
        )
    else:
        pred_df = make_multi_horizon_prediction_frame(
            forecast_times=forecast_te,
            y_true=y_te_seq,
            y_pred=y_pred,
            horizons=range(1, 7),
            model_name="Transformer",
            task="seq6",
        )
        val_df = make_multi_horizon_prediction_frame(
            forecast_times=forecast_va,
            y_true=y_va_seq,
            y_pred=trainer.predict(val_loader),
            horizons=range(1, 7),
            model_name="Transformer",
            task="seq6",
            split="val",
        )
        pred_df.to_csv(OUTPUT_DIR / "transformer_predictions_seq6.csv", index=False)
        val_df.to_csv(OUTPUT_DIR / "transformer_predictions_seq6_val.csv", index=False)
        config_path = OUTPUT_DIR / "transformer_seq6_config.json"
        model_path = OUTPUT_DIR / "transformer_seq6.pt"
        task_name = "seq6"
        log(
            f"saved prediction files for task={task}: "
            f"test_rows={len(pred_df)}, val_rows={len(val_df)}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config,
            "history": trainer.get_history(),
        },
        model_path,
    )
    log(f"saved model checkpoint for task={task}: path={model_path}")

    config = {
        "task": task_name,
        "seq_len": int(seq_len),
        "output_len": int(output_len),
        "d_model": 64,
        "nhead": 4,
        "num_encoder_layers": 3,
        "dim_feedforward": 128,
        "dropout": 0.1,
        "batch_size": int(batch_size),
        "lr": float(lr),
        "epochs": int(epochs),
        "patience": int(patience),
        "train_time_s": round(train_time, 1),
        "best_epoch": int(train_result["best_epoch"]),
        "best_val_loss": float(train_result["best_val_loss"]),
    }
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    log(f"saved config for task={task}: path={config_path}")

    notes = (
        f"seq_len={seq_len},output_len={output_len},best_epoch={train_result['best_epoch']},"
        f"best_val_loss={train_result['best_val_loss']:.4f},train_time_s={train_time:.1f}"
    )
    rows = metric_rows_from_predictions(pred_df, model_name="Transformer", task=task_name, notes=notes)
    log(f"task complete={task}: metric_rows={len(rows)}")
    return rows, trainer


def save_metrics(rows: list[dict]) -> None:
    """Persist combined metrics."""
    path = OUTPUT_DIR / "transformer_metrics.csv"
    df = pd.DataFrame(rows)
    cols = ["model", "task", "horizon_hours", "rmse", "mae", "mape", "n_samples", "notes"]
    for column in cols:
        if column not in df.columns:
            df[column] = ""
    df = df[cols].sort_values(["task", "horizon_hours"]).reset_index(drop=True)
    df.to_csv(path, index=False)
    log(f"saved metrics: rows={len(df)}, path={path}")


def save_history(all_trainers: list[TransformerTrainer], tasks: list[str]) -> None:
    """Persist per-task training curves."""
    path = OUTPUT_DIR / "transformer_train_history.json"
    history_data = {
        task: {
            "train_loss": [float(v) for v in trainer.get_history()["train_loss"]],
            "val_loss": [float(v) for v in trainer.get_history()["val_loss"]],
        }
        for trainer, task in zip(all_trainers, tasks)
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(history_data, handle, ensure_ascii=False, indent=2)
    log(f"saved train history: tasks={tasks}, path={path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Transformer for hourly PM2.5")
    parser.add_argument("--task", default="both", choices=["h1", "seq6", "both"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log(
        f"run started: task={args.task}, epochs={args.epochs}, "
        f"patience={args.patience}, batch_size={args.batch_size}, lr={args.lr}"
    )

    all_rows: list[dict] = []
    all_trainers: list[TransformerTrainer] = []
    all_tasks: list[str] = []
    if args.task in ("h1", "both"):
        rows, trainer = run_task("h1", args.batch_size, args.epochs, args.patience, args.lr)
        all_rows.extend(rows)
        all_trainers.append(trainer)
        all_tasks.append("h1")
    if args.task in ("seq6", "both"):
        rows, trainer = run_task("seq6", args.batch_size, args.epochs + 20, args.patience, args.lr)
        all_rows.extend(rows)
        all_trainers.append(trainer)
        all_tasks.append("seq6")

    save_metrics(all_rows)
    save_history(all_trainers, all_tasks)
    log("all requested tasks complete")


if __name__ == "__main__":
    main()
