"""Train Transformer for hourly PM2.5 prediction tasks (+1h and +24h).

Uses the same data pipeline and sequence construction as LSTM for fair comparison.

Usage:
  python scripts/train_transformer.py --task h1
  python scripts/train_transformer.py --task h24
  python scripts/train_transformer.py --task both
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

from models.transformer_model import TimeSeriesTransformer, TransformerTrainer
from evaluation.metrics import compute_metrics
from utils.feature_engineering import build_hourly_feature_frame, StandardScaler
from utils.io import PROCESSED_DIR

torch.manual_seed(42)

OUTPUT_DIR = ROOT / "outputs" / "hourly"
FEATURES_DIR = PROCESSED_DIR / "features_hourly"
BATCH_SIZE = 64


# ==============================================================================
# Sequence construction (same as LSTM)
# ==============================================================================

def create_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window sequences.

    Args:
        X: (n_samples, n_features) feature matrix
        y: (n_samples,) target array (unshifted pm25)
        seq_len: length of input window
        horizon: forecast steps beyond the end of the sequence

    Returns:
        X_seq: (n_sequences, seq_len, n_features)
        y_seq: (n_sequences,)
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len - horizon + 1):
        X_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len + horizon - 1])
    return np.array(X_seq), np.array(y_seq)


# ==============================================================================
# Data loading
# ==============================================================================

def load_h1_data():
    """Load pre-built +1h feature matrices and create sequences.

    Features are from features_hourly/ (y = current pm25, unshifted).
    Sequences: 24-hour input -> predict +1h.
    """
    X_train = np.load(FEATURES_DIR / "X_train.npy")
    X_test = np.load(FEATURES_DIR / "X_test.npy")
    y_train = np.load(FEATURES_DIR / "y_train.npy")
    y_test = np.load(FEATURES_DIR / "y_test.npy")
    dates_train = np.load(FEATURES_DIR / "dates_train.npy", allow_pickle=True)
    dates_test = np.load(FEATURES_DIR / "dates_test.npy", allow_pickle=True)
    with open(FEATURES_DIR / "feature_names.json") as f:
        feature_names = json.load(f)

    return X_train, X_test, y_train, y_test, dates_train, dates_test, feature_names


def load_h24_data():
    """Build +24h data by reading raw CSV and creating sequences.

    Builds features from beijing_hourly.csv.
    Target is unshifted pm25; the horizon=24 in create_sequences
    handles the 24h-ahead offset.
    Sequences: 48-hour input -> predict +24h.
    """
    df = pd.read_csv(PROCESSED_DIR / "beijing_hourly.csv")
    frame, feature_names = build_hourly_feature_frame(df)

    # Keep datetime and pm25 columns for later use
    frame = frame.copy()

    # Drop rows with NaN in features
    valid = frame.dropna(subset=feature_names).reset_index(drop=True)
    print(f"  h24 valid rows: {len(valid)} (dropped {len(frame) - len(valid)})")

    # Chronological 80/20 split
    split_idx = int(len(valid) * 0.8)
    train = valid.iloc[:split_idx]
    test = valid.iloc[split_idx:]

    # Scale features using train stats
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_names].values)
    X_test = scaler.transform(test[feature_names].values)

    # Target: unshifted pm25 (horizon handled by create_sequences)
    y_train = train["pm25"].values.astype(np.float64)
    y_test = test["pm25"].values.astype(np.float64)

    dates_train = train["datetime"].values
    dates_test = test["datetime"].values

    print(f"  h24 train: {len(train)}, test: {len(test)}")
    return X_train, X_test, y_train, y_test, dates_train, dates_test, feature_names


# ==============================================================================
# Prediction CSV builder
# ==============================================================================

def make_prediction_csv(
    forecast_times: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    model_name: str,
    city: str = "Beijing",
    split: str = "test",
) -> pd.DataFrame:
    """Build unified-format predictions DataFrame."""
    forecast_dt = pd.to_datetime(forecast_times)
    target_dt = forecast_dt + pd.Timedelta(hours=horizon)

    return pd.DataFrame({
        "forecast_origin_time": forecast_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "target_time": target_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "horizon_hours": horizon,
        "city": city,
        "model": model_name,
        "y_true": y_true,
        "y_pred": y_pred,
        "split": split,
    })


# ==============================================================================
# Dataset helper
# ==============================================================================

def create_dataloader(
    X_seq: np.ndarray, y_seq: np.ndarray, batch_size: int = BATCH_SIZE, shuffle: bool = True
) -> DataLoader:
    """Create a DataLoader from numpy arrays."""
    dataset = TensorDataset(
        torch.tensor(X_seq, dtype=torch.float32),
        torch.tensor(y_seq, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ==============================================================================
# Run tasks
# ==============================================================================

def run_h1() -> dict:
    """Train and evaluate Transformer for +1h prediction."""
    print("=" * 60)
    print("Transformer +1h")
    print("=" * 60)

    # Load data
    X_train, X_test, y_train, y_test, dates_train, dates_test, feature_names = load_h1_data()

    # Split train into train/val (80/20 of train, chronologically)
    val_split = int(len(X_train) * 0.8)
    X_tr, X_va = X_train[:val_split], X_train[val_split:]
    y_tr, y_va = y_train[:val_split], y_train[val_split:]
    dates_tr = dates_train[:val_split]
    dates_va = dates_train[val_split:]

    seq_len = 24
    horizon = 1

    # Create sequences
    X_tr_seq, y_tr_seq = create_sequences(X_tr, y_tr, seq_len, horizon)
    X_va_seq, y_va_seq = create_sequences(X_va, y_va, seq_len, horizon)
    X_te_seq, y_te_seq = create_sequences(X_test, y_test, seq_len, horizon)

    # Forecast origin times (datetime of the LAST observation in each sequence)
    forecast_va = dates_va[seq_len - 1 : seq_len - 1 + len(y_va_seq)]
    forecast_te = dates_test[seq_len - 1 : seq_len - 1 + len(y_te_seq)]

    print(f"  Train seqs: {len(X_tr_seq)}, Val seqs: {len(X_va_seq)}, Test seqs: {len(X_te_seq)}")
    print(f"  Features: {len(feature_names)}, input_dim: {X_tr_seq.shape[2]}")

    # Create dataloaders
    train_loader = create_dataloader(X_tr_seq, y_tr_seq, shuffle=False)
    val_loader = create_dataloader(X_va_seq, y_va_seq, shuffle=False)
    test_loader = create_dataloader(X_te_seq, y_te_seq, shuffle=False)

    # Build model
    model = TimeSeriesTransformer(
        input_dim=len(feature_names),
        seq_len=seq_len,
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        dim_feedforward=128,
        dropout=0.1,
    )
    print(f"  Model params: {sum(p.numel() for p in model.parameters())}")

    # Train
    trainer = TransformerTrainer(model, lr=1e-3)
    t0 = time.time()
    train_result = trainer.train(
        train_loader, val_loader, epochs=100, patience=15, verbose=True
    )
    train_time = time.time() - t0
    print(f"  Best epoch: {train_result['best_epoch']}, "
          f"best val loss: {train_result['best_val_loss']:.4f}")
    print(f"  Train time: {train_time:.1f}s")

    # Predict
    y_pred = trainer.predict(test_loader)
    metrics = compute_metrics(y_te_seq, y_pred)
    print(f"  Test RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, "
          f"MAPE={metrics['mape']:.1f}%")

    # Save predictions
    pred_df = make_prediction_csv(
        forecast_te, y_te_seq, y_pred, horizon=horizon, model_name="Transformer"
    )
    pred_path = OUTPUT_DIR / "transformer_predictions_h1.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"  Saved: {pred_path}")

    # Save validation predictions
    val_pred = trainer.predict(val_loader)
    val_df = make_prediction_csv(
        forecast_va, y_va_seq, val_pred, horizon=horizon,
        model_name="Transformer", split="val"
    )

    # Save model
    model_path = OUTPUT_DIR / "transformer_h1.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config,
            "history": trainer.get_history(),
        },
        model_path,
    )
    print(f"  Saved: {model_path}")

    result = {
        "model": "Transformer",
        "horizon_hours": horizon,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mape": metrics["mape"],
        "n_train": len(X_tr_seq),
        "n_val": len(X_va_seq),
        "n_test": len(X_te_seq),
        "best_epoch": train_result["best_epoch"],
        "best_val_loss": train_result["best_val_loss"],
        "train_time_s": round(train_time, 1),
        "model_params": sum(p.numel() for p in model.parameters()),
    }
    return result, trainer


def run_h24() -> dict:
    """Train and evaluate Transformer for +24h prediction."""
    print("=" * 60)
    print("Transformer +24h")
    print("=" * 60)

    # Load data
    X_train, X_test, y_train, y_test, dates_train, dates_test, feature_names = load_h24_data()

    # Split train into train/val (80/20 of train, chronologically)
    val_split = int(len(X_train) * 0.8)
    X_tr, X_va = X_train[:val_split], X_train[val_split:]
    y_tr, y_va = y_train[:val_split], y_train[val_split:]
    dates_tr = dates_train[:val_split]
    dates_va = dates_train[val_split:]

    seq_len = 48
    horizon = 24

    # Create sequences
    X_tr_seq, y_tr_seq = create_sequences(X_tr, y_tr, seq_len, horizon)
    X_va_seq, y_va_seq = create_sequences(X_va, y_va, seq_len, horizon)
    X_te_seq, y_te_seq = create_sequences(X_test, y_test, seq_len, horizon)

    # Forecast origin times (datetime of the LAST observation in each sequence)
    forecast_va = dates_va[seq_len - 1 : seq_len - 1 + len(y_va_seq)]
    forecast_te = dates_test[seq_len - 1 : seq_len - 1 + len(y_te_seq)]

    print(f"  Train seqs: {len(X_tr_seq)}, Val seqs: {len(X_va_seq)}, Test seqs: {len(X_te_seq)}")
    print(f"  Features: {len(feature_names)}, input_dim: {X_tr_seq.shape[2]}")

    # Create dataloaders
    train_loader = create_dataloader(X_tr_seq, y_tr_seq, shuffle=False)
    val_loader = create_dataloader(X_va_seq, y_va_seq, shuffle=False)
    test_loader = create_dataloader(X_te_seq, y_te_seq, shuffle=False)

    # Build model
    model = TimeSeriesTransformer(
        input_dim=len(feature_names),
        seq_len=seq_len,
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        dim_feedforward=128,
        dropout=0.1,
    )
    print(f"  Model params: {sum(p.numel() for p in model.parameters())}")

    # Train
    trainer = TransformerTrainer(model, lr=1e-3)
    t0 = time.time()
    train_result = trainer.train(
        train_loader, val_loader, epochs=100, patience=15, verbose=True
    )
    train_time = time.time() - t0
    print(f"  Best epoch: {train_result['best_epoch']}, "
          f"best val loss: {train_result['best_val_loss']:.4f}")
    print(f"  Train time: {train_time:.1f}s")

    # Predict
    y_pred = trainer.predict(test_loader)
    metrics = compute_metrics(y_te_seq, y_pred)
    print(f"  Test RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, "
          f"MAPE={metrics['mape']:.1f}%")

    # Save predictions
    pred_df = make_prediction_csv(
        forecast_te, y_te_seq, y_pred, horizon=horizon, model_name="Transformer"
    )
    pred_path = OUTPUT_DIR / "transformer_predictions_h24.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"  Saved: {pred_path}")

    # Save validation predictions
    val_pred = trainer.predict(val_loader)
    val_df = make_prediction_csv(
        forecast_va, y_va_seq, val_pred, horizon=horizon,
        model_name="Transformer", split="val"
    )

    # Save model
    model_path = OUTPUT_DIR / "transformer_h24.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config,
            "history": trainer.get_history(),
        },
        model_path,
    )
    print(f"  Saved: {model_path}")

    result = {
        "model": "Transformer",
        "horizon_hours": horizon,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mape": metrics["mape"],
        "n_train": len(X_tr_seq),
        "n_val": len(X_va_seq),
        "n_test": len(X_te_seq),
        "best_epoch": train_result["best_epoch"],
        "best_val_loss": train_result["best_val_loss"],
        "train_time_s": round(train_time, 1),
        "model_params": sum(p.numel() for p in model.parameters()),
    }
    return result, trainer


# ==============================================================================
# Save helpers
# ==============================================================================

def save_metrics(all_rows: list[dict]) -> None:
    """Save combined metrics CSV."""
    path = OUTPUT_DIR / "transformer_metrics.csv"
    df = pd.DataFrame(all_rows)
    if "notes" not in df.columns:
        df["notes"] = ""
    df = df[[
        "model", "horizon_hours", "rmse", "mae", "mape",
        "n_train", "n_val", "n_test",
        "best_epoch", "best_val_loss", "train_time_s", "model_params", "notes",
    ]]
    df.to_csv(path, index=False)
    print(f"Metrics saved: {path}")


def save_history(all_trainers: list[TransformerTrainer], horizons: list[int]) -> None:
    """Save training history as JSON."""
    path = OUTPUT_DIR / "transformer_train_history.json"
    history_data = {}
    for trainer, h in zip(all_trainers, horizons):
        history_data[f"h{h}"] = {
            "train_loss": trainer.get_history()["train_loss"],
            "val_loss": trainer.get_history()["val_loss"],
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)
    print(f"History saved: {path}")


def append_registry(rows: list[dict]) -> None:
    """Append experiment entries to registry."""
    path = OUTPUT_DIR / "experiment_registry.csv"
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    records = []
    for r in rows:
        records.append({
            "experiment_id": f"transformer_h{r['horizon_hours']}_{now.replace(' ', 'T')}",
            "model": r["model"],
            "horizon_hours": r["horizon_hours"],
            "feature_set": "full_20",
            "train_range": "",
            "val_range": "",
            "test_range": "",
            "params": f"d_model=64,nhead=4,layers=3,dim_ff=128,lr=1e-3,params={r.get('model_params', '')}",
            "best_score": r.get("best_val_loss", ""),
            "run_time": r.get("train_time_s", ""),
            "status": "done",
            "author": "auto",
            "notes": f"RMSE={r['rmse']:.2f}",
        })

    df_new = pd.DataFrame(records)
    if path.exists():
        df_old = pd.read_csv(path)
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    df_new.to_csv(path, index=False)
    print(f"Registry updated: {path}")


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Transformer for hourly PM2.5")
    parser.add_argument(
        "--task", default="both", choices=["h1", "h24", "both"],
        help="Prediction horizon task",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_trainers: list[TransformerTrainer] = []
    all_horizons: list[int] = []

    if args.task in ("h1", "both"):
        r, trainer = run_h1()
        all_rows.append(r)
        all_trainers.append(trainer)
        all_horizons.append(1)

    if args.task in ("h24", "both"):
        r, trainer = run_h24()
        all_rows.append(r)
        all_trainers.append(trainer)
        all_horizons.append(24)

    save_metrics(all_rows)
    save_history(all_trainers, all_horizons)
    append_registry(all_rows)

    print("\n" + "=" * 60)
    print("Transformer Summary")
    print("=" * 60)
    for r in all_rows:
        print(f"  +{r['horizon_hours']}h: RMSE={r['rmse']:.2f}, "
              f"MAE={r['mae']:.2f}, MAPE={r['mape']:.1f}%")


if __name__ == "__main__":
    main()
