"""Train Transformer for hourly PM2.5 prediction tasks (+1h and +12h)."""

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

from evaluation.metrics import compute_metrics
from models.transformer_model import TimeSeriesTransformer, TransformerTrainer
from utils.feature_engineering import StandardScaler, build_hourly_feature_frame
from utils.io import PROCESSED_DIR

OUTPUT_DIR = ROOT / "outputs" / "hourly"
FEATURES_DIR = PROCESSED_DIR / "features_hourly"

np.random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(4)


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    n_windows = len(X) - seq_len - horizon + 1
    if n_windows <= 0:
        raise ValueError(f"Not enough samples for seq_len={seq_len}, horizon={horizon}")
    X_seq = np.zeros((n_windows, seq_len, X.shape[1]), dtype=np.float32)
    y_seq = np.zeros(n_windows, dtype=np.float32)
    for i in range(n_windows):
        X_seq[i] = X[i : i + seq_len]
        y_seq[i] = y[i + seq_len + horizon - 1]
    return X_seq, y_seq


def load_h1_data():
    X_train = np.load(FEATURES_DIR / "X_train.npy")
    X_test = np.load(FEATURES_DIR / "X_test.npy")
    y_train = np.load(FEATURES_DIR / "y_train.npy")
    y_test = np.load(FEATURES_DIR / "y_test.npy")
    dates_train = np.load(FEATURES_DIR / "dates_train.npy", allow_pickle=True)
    dates_test = np.load(FEATURES_DIR / "dates_test.npy", allow_pickle=True)
    with open(FEATURES_DIR / "feature_names.json", encoding="utf-8") as f:
        feature_names = json.load(f)
    return X_train, X_test, y_train, y_test, dates_train, dates_test, feature_names


def load_h12_data():
    df = pd.read_csv(PROCESSED_DIR / "beijing_hourly.csv")
    frame, feature_names = build_hourly_feature_frame(df)
    valid = frame.dropna(subset=feature_names + ["pm25"]).reset_index(drop=True)
    split_idx = int(len(valid) * 0.8)
    train, test = valid.iloc[:split_idx], valid.iloc[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_names].values)
    X_test = scaler.transform(test[feature_names].values)
    y_train = train["pm25"].values.astype(np.float64)
    y_test = test["pm25"].values.astype(np.float64)
    dates_train = train["datetime"].values
    dates_test = test["datetime"].values
    return X_train, X_test, y_train, y_test, dates_train, dates_test, feature_names


def make_prediction_csv(
    forecast_times: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    model_name: str,
    city: str = "Beijing",
    split: str = "test",
) -> pd.DataFrame:
    forecast_dt = pd.to_datetime(forecast_times)
    target_dt = forecast_dt + pd.Timedelta(hours=int(horizon))
    return pd.DataFrame(
        {
            "forecast_origin_time": forecast_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "target_time": target_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "horizon_hours": int(horizon),
            "city": city,
            "model": model_name,
            "y_true": y_true,
            "y_pred": y_pred,
            "split": split,
        }
    )


def _run(task: str, batch_size: int, epochs: int, patience: int, lr: float) -> tuple[dict, TransformerTrainer]:
    if task == "h1":
        X_train, X_test, y_train, y_test, dates_train, dates_test, feature_names = load_h1_data()
        seq_len, horizon = 48, 1
    else:
        X_train, X_test, y_train, y_test, dates_train, dates_test, feature_names = load_h12_data()
        seq_len, horizon = 72, 12

    val_split = int(len(X_train) * 0.8)
    X_tr, X_va = X_train[:val_split], X_train[val_split:]
    y_tr, y_va = y_train[:val_split], y_train[val_split:]
    dates_va = dates_train[val_split:]

    X_tr_seq, y_tr_seq = create_sequences(X_tr, y_tr, seq_len, horizon)
    X_va_seq, y_va_seq = create_sequences(X_va, y_va, seq_len, horizon)
    X_te_seq, y_te_seq = create_sequences(X_test, y_test, seq_len, horizon)

    forecast_va = dates_va[seq_len - 1 : seq_len - 1 + len(y_va_seq)]
    forecast_te = dates_test[seq_len - 1 : seq_len - 1 + len(y_te_seq)]

    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr_seq, dtype=torch.float32), torch.tensor(y_tr_seq, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_va_seq, dtype=torch.float32), torch.tensor(y_va_seq, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_te_seq, dtype=torch.float32), torch.tensor(y_te_seq, dtype=torch.float32)), batch_size=batch_size, shuffle=False)

    model = TimeSeriesTransformer(
        input_dim=len(feature_names),
        seq_len=seq_len,
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        dim_feedforward=128,
        dropout=0.1,
    )
    trainer = TransformerTrainer(model, lr=lr)

    t0 = time.time()
    train_result = trainer.train(train_loader, val_loader, epochs=epochs, patience=patience, verbose=True)
    train_time = time.time() - t0

    y_pred = trainer.predict(test_loader)
    metrics = compute_metrics(y_te_seq, y_pred)

    pred_df = make_prediction_csv(forecast_te, y_te_seq, y_pred, horizon=horizon, model_name="Transformer")
    pred_df.to_csv(OUTPUT_DIR / f"transformer_predictions_{task}.csv", index=False)

    val_pred = trainer.predict(val_loader)
    val_df = make_prediction_csv(forecast_va, y_va_seq, val_pred, horizon=horizon, model_name="Transformer", split="val")
    val_df.to_csv(OUTPUT_DIR / f"transformer_predictions_{task}_val.csv", index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config,
            "history": trainer.get_history(),
        },
        OUTPUT_DIR / f"transformer_{task}.pt",
    )

    config = {
        "task": task,
        "horizon_hours": int(horizon),
        "seq_len": int(seq_len),
        "d_model": 64,
        "nhead": 4,
        "num_encoder_layers": 3,
        "dim_feedforward": 128,
        "dropout": 0.1,
        "batch_size": int(batch_size),
        "lr": float(lr),
        "epochs": int(epochs),
        "patience": int(patience),
    }
    with open(OUTPUT_DIR / f"transformer_{task}_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    row = {
        "model": "Transformer",
        "horizon_hours": int(horizon),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mape": metrics["mape"],
        "n_samples": int(metrics["n_samples"]),
        "notes": f"seq_len={seq_len},best_epoch={train_result['best_epoch']},best_val_loss={train_result['best_val_loss']:.4f},train_time_s={train_time:.1f}",
    }
    return row, trainer


def save_metrics(all_rows: list[dict]) -> None:
    path = OUTPUT_DIR / "transformer_metrics.csv"
    df = pd.DataFrame(all_rows)
    cols = ["model", "horizon_hours", "rmse", "mae", "mape", "n_samples", "notes"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]
    df.to_csv(path, index=False)


def save_history(all_trainers: list[TransformerTrainer], tasks: list[str]) -> None:
    path = OUTPUT_DIR / "transformer_train_history.json"
    history_data = {
        task: {
            "train_loss": trainer.get_history()["train_loss"],
            "val_loss": trainer.get_history()["val_loss"],
        }
        for trainer, task in zip(all_trainers, tasks)
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Transformer for hourly PM2.5")
    parser.add_argument("--task", default="both", choices=["h1", "h12", "both"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_trainers: list[TransformerTrainer] = []
    all_tasks: list[str] = []

    if args.task in ("h1", "both"):
        r, trainer = _run("h1", args.batch_size, args.epochs, args.patience, args.lr)
        all_rows.append(r)
        all_trainers.append(trainer)
        all_tasks.append("h1")

    if args.task in ("h12", "both"):
        r, trainer = _run("h12", args.batch_size, args.epochs, args.patience, args.lr)
        all_rows.append(r)
        all_trainers.append(trainer)
        all_tasks.append("h12")

    save_metrics(all_rows)
    save_history(all_trainers, all_tasks)


if __name__ == "__main__":
    main()
