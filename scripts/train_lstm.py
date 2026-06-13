"""Train LSTM for hourly PM2.5 prediction tasks (+1h and +12h)."""

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
from models.lstm_model import LSTMModel, LSTMTrainer
from utils.feature_engineering import StandardScaler, build_hourly_feature_frame
from utils.io import PROCESSED_DIR

OUTPUT_DIR = ROOT / "outputs" / "hourly"
FEATURES_DIR = PROCESSED_DIR / "features_hourly"

torch.set_num_threads(4)
np.random.seed(42)
torch.manual_seed(42)


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


def forecast_dates_for_sequences(dates: np.ndarray, seq_len: int, horizon: int) -> np.ndarray:
    n_windows = len(dates) - seq_len - horizon + 1
    return dates[seq_len - 1 : seq_len - 1 + n_windows]


def load_h1_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X_train = np.load(FEATURES_DIR / "X_train.npy")
    X_test = np.load(FEATURES_DIR / "X_test.npy")
    y_train = np.load(FEATURES_DIR / "y_train.npy")
    y_test = np.load(FEATURES_DIR / "y_test.npy")
    dates_train = np.load(FEATURES_DIR / "dates_train.npy", allow_pickle=True)
    dates_test = np.load(FEATURES_DIR / "dates_test.npy", allow_pickle=True)
    with open(FEATURES_DIR / "feature_names.json", encoding="utf-8") as f:
        feature_names = json.load(f)
    return X_train, X_test, y_train, y_test, dates_train, dates_test, feature_names


def load_h12_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(PROCESSED_DIR / "beijing_hourly.csv")
    frame, feature_names = build_hourly_feature_frame(df)
    valid = frame.dropna(subset=feature_names + ["pm25"]).reset_index(drop=True)
    split_idx = int(len(valid) * 0.8)
    train, test = valid.iloc[:split_idx], valid.iloc[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_names].values.astype(np.float64))
    X_test = scaler.transform(test[feature_names].values.astype(np.float64))
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
    split_label: str = "test",
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
            "split": split_label,
        }
    )


def run_task(task: str) -> tuple[dict, dict]:
    if task == "h1":
        X_train, X_test, y_train, y_test, dates_train, dates_test, feature_names = load_h1_data()
        seq_len, horizon = 48, 1
        hidden_dim, num_layers, dropout, lr = 128, 3, 0.3, 3e-4
        batch_size, patience, epochs = 32, 20, 120
    else:
        X_train, X_test, y_train, y_test, dates_train, dates_test, feature_names = load_h12_data()
        seq_len, horizon = 72, 12
        hidden_dim, num_layers, dropout, lr = 64, 2, 0.2, 1e-3
        batch_size, patience, epochs = 64, 12, 100

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len, horizon)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_len, horizon)
    fct_test = forecast_dates_for_sequences(dates_test, seq_len, horizon)

    val_split = int(len(X_train_seq) * 0.8)
    X_tr, X_va = X_train_seq[:val_split], X_train_seq[val_split:]
    y_tr, y_va = y_train_seq[:val_split], y_train_seq[val_split:]

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test_seq), torch.from_numpy(y_test_seq)), batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_dim=X_train_seq.shape[2], hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    trainer = LSTMTrainer(model, lr=lr)

    t0 = time.time()
    history = trainer.train(train_loader, val_loader, epochs=epochs, patience=patience)
    train_time = time.time() - t0

    y_pred = trainer.predict(test_loader)
    metrics = compute_metrics(y_test_seq, y_pred)

    pred_df = make_prediction_csv(fct_test, y_test_seq, y_pred, horizon=horizon, model_name="LSTM")
    pred_path = OUTPUT_DIR / f"lstm_predictions_{task}.csv"
    pred_df.to_csv(pred_path, index=False)

    trainer.save(OUTPUT_DIR / f"lstm_{task}.pkl")

    config = {
        "task": task,
        "horizon_hours": int(horizon),
        "seq_len": int(seq_len),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "patience": int(patience),
        "epochs": int(epochs),
    }
    with open(OUTPUT_DIR / f"lstm_{task}_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    row = {
        "model": "LSTM",
        "horizon_hours": int(horizon),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mape": metrics["mape"],
        "n_samples": int(metrics["n_samples"]),
        "notes": f"seq_len={seq_len},best_val_loss={float(trainer.best_val_loss):.4f},train_time_s={train_time:.1f}",
    }
    return row, history


def save_metrics(rows: list[dict]) -> None:
    path = OUTPUT_DIR / "lstm_metrics.csv"
    df = pd.DataFrame(rows)
    cols = ["model", "horizon_hours", "rmse", "mae", "mape", "n_samples", "notes"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]
    df.to_csv(path, index=False)


def save_train_history(all_histories: dict[str, dict]) -> None:
    path = OUTPUT_DIR / "lstm_train_history.json"
    serializable = {
        task: {
            "train_loss": [float(v) for v in hist.get("train_loss", [])],
            "val_loss": [float(v) for v in hist.get("val_loss", [])],
        }
        for task, hist in all_histories.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM for hourly PM2.5")
    parser.add_argument("--task", default="both", choices=["h1", "h12", "both"])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_histories: dict[str, dict] = {}

    if args.task in ("h1", "both"):
        row, history = run_task("h1")
        all_rows.append(row)
        all_histories["h1"] = history

    if args.task in ("h12", "both"):
        row, history = run_task("h12")
        all_rows.append(row)
        all_histories["h12"] = history

    save_metrics(all_rows)
    save_train_history(all_histories)


if __name__ == "__main__":
    main()
