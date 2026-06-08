"""LSTM model for Beijing PM2.5 daily prediction (B-4).

Builds sliding-window sequences from the multivariate daily table.
Architecture: 2× LSTM → Dropout → Dense(1), trained with
Early Stopping and a small hyper-parameter sweep.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io import PROCESSED_DIR

log = logging.getLogger(__name__)

# -- config ----------------------------------------------------------------
LOOKBACK = 30
TEST_RATIO = 0.2
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 150
PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
                "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------
def _load_multivariate() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return df with dates, scaled feature matrix, target (PM2.5)."""
    df = pd.read_csv(
        PROCESSED_DIR / "beijing" / "beijing_daily_city.csv",
        parse_dates=["date"],
    ).sort_values("date").reset_index(drop=True)

    raw = df[FEATURE_COLS].values.astype(np.float64)
    # fill any remaining NaNs with column mean
    col_means = np.nanmean(raw, axis=0)
    raw = np.where(np.isnan(raw), col_means, raw)

    # MinMaxScaler: LSTM converges better with [0,1] inputs
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(raw)
    target_idx = FEATURE_COLS.index("PM2.5")
    return df, scaled, scaler, target_idx


def _build_windows(
    data: np.ndarray, target_idx: int, lookback: int = LOOKBACK
) -> tuple[np.ndarray, np.ndarray]:
    """Convert (T, F) matrix into (T-lookback, lookback, F) windows."""
    X_list, y_list = [], []
    for i in range(len(data) - lookback):
        X_list.append(data[i:i + lookback])
        y_list.append(data[i + lookback, target_idx])
    return np.stack(X_list), np.array(y_list)


def _split_windows(
    X: np.ndarray, y: np.ndarray, test_ratio: float = TEST_RATIO
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronological split on sliding-window data."""
    n = len(X)
    split = int(n * (1 - test_ratio))
    return X[:split], X[split:], y[:split], y[split:]


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)          # (B, L, H)
        return self.fc(out[:, -1, :])  # last timestep → scalar


def _train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = LR,
    epochs: int = EPOCHS,
) -> list[float]:
    """Train with Adam + MSELoss, early-stopping on val loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val = float("inf")
    patience_left = PATIENCE
    best_state = None
    history: list[float] = []

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb).squeeze(-1), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(Xb).squeeze(-1), yb).item() * Xb.size(0)
        val_loss /= len(val_loader.dataset)
        history.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience_left = PATIENCE
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int,
                 shuffle: bool) -> DataLoader:
    tX = torch.tensor(X, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(tX, ty), batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# hyper-param sweep
# ---------------------------------------------------------------------------
def _sweep(
    X_train: np.ndarray, y_train: np.ndarray,
    input_dim: int,
) -> dict[str, Any]:
    """Lightweight grid search for LSTM hyper-parameters."""
    # hold out a validation portion from train (chronological)
    val_size = int(len(X_train) * 0.15)
    X_tr, X_va = X_train[:-val_size], X_train[-val_size:]
    y_tr, y_va = y_train[:-val_size], y_train[-val_size:]

    best_rmse = float("inf")
    best_cfg: dict[str, Any] = {}

    for lr in [1e-3, 5e-4, 1e-4]:
        for hidden in [32, 64, 128]:
            for bs in [16, 32]:
                model = LSTMPredictor(input_dim, hidden=hidden).to(DEVICE)
                tr_loader = _make_loader(X_tr, y_tr, bs, shuffle=False)
                va_loader = _make_loader(X_va, y_va, bs, shuffle=False)
                _train_model(model, tr_loader, va_loader, lr=lr, epochs=100)

                model.eval()
                with torch.no_grad():
                    p = model(torch.tensor(X_va, dtype=torch.float32).to(DEVICE))
                    rmse = float(np.sqrt(np.mean((y_va - p.cpu().numpy().squeeze()) ** 2)))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_cfg = {"lr": lr, "hidden": hidden, "batch_size": bs}
    log.info("  best val RMSE=%.2f  cfg=%s", best_rmse, best_cfg)
    return best_cfg


# ---------------------------------------------------------------------------
def run() -> dict[str, Any]:
    df, scaled, scaler, target_idx = _load_multivariate()
    X, y = _build_windows(scaled, target_idx)
    X_train, X_test, y_train, y_test = _split_windows(X, y)

    # inverse-transform target for metric computation (MinMaxScaler)
    pm25_min = scaler.data_min_[target_idx]
    pm25_range = scaler.data_range_[target_idx]
    y_test_orig = y_test * pm25_range + pm25_min

    log.info("LSTM: train %s  test %s  device=%s  lookback=%d",
             X_train.shape, X_test.shape, DEVICE, LOOKBACK)

    cfg = _sweep(X_train, y_train, X_train.shape[2])

    # final train on full training set
    val_size = int(len(X_train) * 0.15)
    X_tr_final = X_train[:-val_size]; y_tr_final = y_train[:-val_size]
    X_val_final = X_train[-val_size:]; y_val_final = y_train[-val_size:]

    model = LSTMPredictor(X_train.shape[2], hidden=cfg["hidden"]).to(DEVICE)
    tr_loader = _make_loader(X_tr_final, y_tr_final, cfg["batch_size"], shuffle=False)
    va_loader = _make_loader(X_val_final, y_val_final, cfg["batch_size"], shuffle=False)
    history = _train_model(model, tr_loader, va_loader, lr=cfg["lr"])

    # predict
    model.eval()
    with torch.no_grad():
        preds_scaled = model(
            torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        ).cpu().numpy().squeeze()
    y_pred = preds_scaled * pm25_range + pm25_min

    # metrics
    residuals = y_test_orig - y_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    ape = np.abs(residuals / np.maximum(y_test_orig, 1e-6)) * 100
    mape = float(np.mean(np.minimum(ape, 200.0)))

    # predictions
    window_offset = LOOKBACK + int(len(X) * (1 - TEST_RATIO))
    test_dates = df["date"].iloc[window_offset:].values
    rows: list[dict[str, Any]] = []
    for d, t, p in zip(test_dates, y_test_orig, y_pred):
        rows.append({
            "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
            "city": "Beijing",
            "model": "LSTM",
            "y_true": round(float(t), 4),
            "y_pred": round(max(0.0, float(p)), 4),
        })

    log.info("LSTM  RMSE=%.2f  MAE=%.2f  MAPE=%.1f%%", rmse, mae, mape)
    return {
        "model": "LSTM",
        "rmse": rmse, "mae": mae, "mape": mape,
        "cfg": cfg, "history": history,
        "n_train": int(X_train.shape[0]), "n_test": int(X_test.shape[0]),
        "predictions": rows,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run()
    print(f"\n{'='*50}")
    print(
        f"LSTM  | RMSE={result['rmse']:.2f}  MAE={result['mae']:.2f}  "
        f"MAPE={result['mape']:.1f}%  (n={result['n_test']})"
    )
