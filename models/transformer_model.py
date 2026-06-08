"""Informer model for Beijing PM2.5 daily prediction (B-5).

Implements the core Informer architecture (Zhou et al., AAAI 2021):
- ProbSparse self-attention
- Self-attention distilling (conv pooling between encoder layers)
- Generative-style decoder (one-shot prediction)

Reference: https://github.com/zhouhaoyi/Informer2020
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io import PROCESSED_DIR

log = logging.getLogger(__name__)

# -- config ----------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

SEQ_LEN = 30        # encoder input length (lookback)
LABEL_LEN = 15      # decoder known (start) token length
PRED_LEN = 1        # forecast horizon
TEST_RATIO = 0.2

D_MODEL = 128
N_HEADS = 4
E_LAYERS = 2
D_LAYERS = 1
D_FF = 512
DROPOUT = 0.2
FACTOR = 5          # ProbSparse sampling factor

BATCH_SIZE = 32
LR = 5e-4
EPOCHS = 200
PATIENCE = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
                "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
TARGET_IDX = 0  # PM2.5 is first column


# ===================================================================
# ProbSparse Attention
# ===================================================================
class ProbSparseAttention(nn.Module):
    """ProbSparse self-attention layer.

    For each query, the sparsity measure is:
        M(q_i, K) = max_j(q_i·k_j/√d) - mean_j(q_i·k_j/√d)

    Only the top-u queries (with highest max-mean difference) compute
    full attention; the rest use the mean of V.
    """

    def __init__(self, factor: int = FACTOR, scale: float | None = None):
        super().__init__()
        self.factor = factor
        self.scale = scale

    def _sparsity_measure(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """M(q_i, K) = max(qk) - mean(qk) using sampled keys."""
        B, H, L_Q, D = Q.shape
        L_K = K.shape[2]
        n_sample = min(self.factor * int(math.log(L_K)), L_K)

        K_sample = K[:, :, torch.randperm(L_K)[:n_sample], :]  # (B,H,n,D)
        scores = torch.matmul(Q, K_sample.transpose(-2, -1))      # (B,H,L_Q,n)
        scale = self.scale or (D ** -0.5)
        scores = scores * scale
        return scores.max(dim=-1)[0] - scores.mean(dim=-1)         # (B,H,L_Q)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, H, L_Q, D = Q.shape
        L_K = K.shape[2]
        scale = self.scale or (D ** -0.5)

        # sparsity measure & select top-u
        M = self._sparsity_measure(Q, K)
        u = max(1, min(int(self.factor * math.log(L_Q)), L_Q))
        _, top_u_idx = torch.topk(M, u, dim=-1)  # (B, H, u)

        # gather top-u queries
        top_u_idx_exp = top_u_idx.unsqueeze(-1).expand(B, H, u, D)
        Q_top = torch.gather(Q, dim=-2, index=top_u_idx_exp)  # (B, H, u, D)

        # sparse attention for top-u queries
        scores_top = torch.matmul(Q_top, K.transpose(-2, -1)) * scale  # (B,H,u,L_K)
        attn_top = F.softmax(scores_top, dim=-1)
        ctx_top = torch.matmul(attn_top, V)  # (B, H, u, D)

        # mean-pooling for remaining queries
        V_mean = V.mean(dim=-2, keepdim=True).expand(B, H, L_Q, D)  # (B,H,L_Q,D)

        # scatter sparse context into full context
        ctx = V_mean.clone()
        ctx.scatter_(dim=-2, index=top_u_idx_exp, src=ctx_top)

        return ctx, None


class AttentionLayer(nn.Module):
    """Multi-head ProbSparse attention block with residual + layernorm."""

    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 factor: int = FACTOR, dropout: float = DROPOUT):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention = ProbSparseAttention(factor=factor)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
                ) -> torch.Tensor:
        B, L_Q, _ = Q.shape; B, L_K, _ = K.shape
        residual = Q

        def _reshape(x: torch.Tensor, L: int) -> torch.Tensor:
            return (x.view(B, L, self.n_heads, self.d_k)
                     .transpose(1, 2))  # (B, H, L, d_k)

        q, k, v = self.w_q(Q), self.w_k(K), self.w_v(V)
        q, k, v = _reshape(q, L_Q), _reshape(k, L_K), _reshape(v, L_K)

        ctx, _ = self.attention(q, k, v)
        ctx = ctx.transpose(1, 2).contiguous().view(B, L_Q, self.d_model)
        return self.norm(residual + self.dropout(self.out_proj(ctx)))


# ===================================================================
# Embedding
# ===================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class DataEmbedding(nn.Module):
    """Value embedding + positional encoding + optional temporal features."""

    def __init__(self, c_in: int, d_model: int = D_MODEL, dropout: float = DROPOUT):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, c_in)
        return self.dropout(self.position_encoding(self.value_embedding(x)))


# ===================================================================
# Encoder
# ===================================================================
class ConvLayer(nn.Module):
    """1-D conv distillation to halve sequence length between encoder layers."""

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, stride=1)
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        y = x.transpose(1, 2)                   # (B, D, L)
        y = self.pool(self.activation(self.conv(y)))  # (B, D, L//2)
        y = y.transpose(1, 2)                   # (B, L//2, D)
        return self.norm(y)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 d_ff: int = D_FF, dropout: float = DROPOUT,
                 factor: int = FACTOR):
        super().__init__()
        self.attn = AttentionLayer(d_model, n_heads, factor)
        self.conv = ConvLayer(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x, x, x)
        return self.conv(attn_out)


class Encoder(nn.Module):
    def __init__(self, attn_layers: list[EncoderLayer]):
        super().__init__()
        self.layers = nn.ModuleList(attn_layers)
        self.norm = nn.LayerNorm(D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ===================================================================
# Decoder
# ===================================================================
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 d_ff: int = D_FF, dropout: float = DROPOUT,
                 factor: int = FACTOR):
        super().__init__()
        self.self_attn = AttentionLayer(d_model, n_heads, factor)
        self.cross_attn = AttentionLayer(d_model, n_heads, factor)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        # self-attention on decoder input
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x)))
        # cross-attention with encoder output
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out)))
        # feed-forward
        return self.norm3(x + self.dropout(self.ff(x)))


class Decoder(nn.Module):
    def __init__(self, layers: list[DecoderLayer]):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(D_MODEL)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)


# ===================================================================
# Full Informer Model
# ===================================================================
class Informer(nn.Module):
    """Informer for univariate/multivariate time-series forecasting."""

    def __init__(self, enc_in: int, dec_in: int, c_out: int,
                 seq_len: int = SEQ_LEN, label_len: int = LABEL_LEN,
                 pred_len: int = PRED_LEN, d_model: int = D_MODEL,
                 n_heads: int = N_HEADS, e_layers: int = E_LAYERS,
                 d_layers: int = D_LAYERS, d_ff: int = D_FF,
                 dropout: float = DROPOUT, factor: int = FACTOR):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        # embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)

        # encoder
        enc_layers = []
        for _ in range(e_layers):
            enc_layers.append(EncoderLayer(d_model, n_heads, d_ff, dropout, factor))
        self.encoder = Encoder(enc_layers)

        # decoder
        dec_layers = [DecoderLayer(d_model, n_heads, d_ff, dropout, factor)
                      for _ in range(d_layers)]
        self.decoder = Decoder(dec_layers)

        # output projection
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc: torch.Tensor, x_dec: torch.Tensor
                ) -> torch.Tensor:
        # embed
        enc_out = self.enc_embedding(x_enc)
        enc_out = self.encoder(enc_out)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out)

        # project
        return self.projection(dec_out)[:, -self.pred_len:, :]  # (B, pred_len, c_out)


# ===================================================================
# Data pipeline
# ===================================================================
def _load_data() -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    df = pd.read_csv(
        PROCESSED_DIR / "beijing" / "beijing_daily_city.csv",
        parse_dates=["date"],
    ).sort_values("date").reset_index(drop=True)

    raw = df[FEATURE_COLS].values.astype(np.float64)
    col_means = np.nanmean(raw, axis=0)
    raw = np.where(np.isnan(raw), col_means, raw)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw)
    return df, scaled, scaler


def _build_windows(data: np.ndarray, seq_len: int, label_len: int, pred_len: int
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build encoder and decoder inputs with targets.

    Returns:
        x_enc: (N, seq_len, C) — encoder input
        x_dec: (N, label_len+pred_len, C) — decoder input
        y:     (N, pred_len, 1) — target PM2.5
    """
    enc_inputs, dec_inputs, targets = [], [], []
    total = seq_len + pred_len

    for i in range(len(data) - total):
        enc_inputs.append(data[i:i + seq_len])
        # decoder: last label_len known steps + pred_len placeholders (zeros)
        dec_in = np.zeros((label_len + pred_len, data.shape[1]), dtype=np.float32)
        dec_in[:label_len] = data[i + seq_len - label_len:i + seq_len]
        dec_inputs.append(dec_in)
        # target: PM2.5 at position seq_len
        targets.append(data[i + seq_len, TARGET_IDX])

    return (
        np.stack(enc_inputs).astype(np.float32),
        np.stack(dec_inputs).astype(np.float32),
        np.array(targets, dtype=np.float32)[:, np.newaxis],  # (N, 1)
    )


def _split(*arrays: np.ndarray
           ) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(arrays[0])
    split = int(n * (1 - TEST_RATIO))
    return [(a[:split], a[split:]) for a in arrays]


def _make_loader(x_enc: np.ndarray, x_dec: np.ndarray, y: np.ndarray,
                 batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.from_numpy(x_enc), torch.from_numpy(x_dec), torch.from_numpy(y)),
        batch_size=batch_size, shuffle=shuffle)


# ===================================================================
def run() -> dict[str, Any]:
    df, scaled, scaler = _load_data()
    c_in = scaled.shape[1]

    x_enc, x_dec, y = _build_windows(scaled, SEQ_LEN, LABEL_LEN, PRED_LEN)
    (x_enc_tr, x_enc_te), (x_dec_tr, x_dec_te), (y_tr, y_te) = _split(x_enc, x_dec, y)

    # inverse-transform target
    pm25_mean = scaler.mean_[TARGET_IDX]
    pm25_scale = scaler.scale_[TARGET_IDX]
    y_test_orig = y_te[:, 0] * pm25_scale + pm25_mean

    log.info("Informer: train=%d test=%d seq_len=%d label_len=%d pred_len=%d",
             len(x_enc_tr), len(x_enc_te), SEQ_LEN, LABEL_LEN, PRED_LEN)
    log.info("  d_model=%d heads=%d e_layers=%d d_layers=%d device=%s",
             D_MODEL, N_HEADS, E_LAYERS, D_LAYERS, DEVICE)

    # validation split from train
    val_size = int(len(x_enc_tr) * 0.15)
    tr_loader = _make_loader(x_enc_tr[:-val_size], x_dec_tr[:-val_size],
                             y_tr[:-val_size], BATCH_SIZE, shuffle=False)
    va_loader = _make_loader(x_enc_tr[-val_size:], x_dec_tr[-val_size:],
                             y_tr[-val_size:], BATCH_SIZE, shuffle=False)

    # model
    model = Informer(
        enc_in=c_in, dec_in=c_in, c_out=1,
        seq_len=SEQ_LEN, label_len=LABEL_LEN, pred_len=PRED_LEN,
    ).to(DEVICE)
    log.info("  params: %.1fK", sum(p.numel() for p in model.parameters())/1000)

    # training
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    best_val = float("inf")
    patience_left = PATIENCE
    best_state = None

    for ep in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for Xe, Xd, Yb in tr_loader:
            Xe, Xd, Yb = Xe.to(DEVICE), Xd.to(DEVICE), Yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xe, Xd).squeeze(-1), Yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * Xe.size(0)
        train_loss /= len(tr_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xe, Xd, Yb in va_loader:
                Xe, Xd, Yb = Xe.to(DEVICE), Xd.to(DEVICE), Yb.to(DEVICE)
                val_loss += criterion(model(Xe, Xd).squeeze(-1), Yb).item() * Xe.size(0)
        val_loss /= len(va_loader.dataset)

        if val_loss < best_val:
            best_val = val_loss
            patience_left = PATIENCE
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left == 0:
                log.info("  early stop at epoch %d", ep + 1)
                break

    model.load_state_dict(best_state)

    # predict
    model.eval()
    with torch.no_grad():
        test_loader = _make_loader(x_enc_te, x_dec_te, y_te, BATCH_SIZE, shuffle=False)
        preds_list = []
        for Xe, Xd, _ in test_loader:
            Xe, Xd = Xe.to(DEVICE), Xd.to(DEVICE)
            preds_list.append(model(Xe, Xd).cpu().numpy())
        y_pred_scaled = np.concatenate(preds_list, axis=0).ravel()  # flatten to 1D

    y_pred = y_pred_scaled * pm25_scale + pm25_mean

    # metrics
    residuals = y_test_orig - y_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    ape = np.abs(residuals / np.maximum(y_test_orig, 1e-6)) * 100
    mape = float(np.mean(np.minimum(ape, 200.0)))

    # predictions
    window_offset = SEQ_LEN + int(len(x_enc) * (1 - TEST_RATIO))
    test_dates = df["date"].iloc[window_offset:window_offset + len(y_pred)].values
    rows: list[dict[str, Any]] = []
    for d, t, p in zip(test_dates, y_test_orig, y_pred):
        rows.append({
            "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
            "city": "Beijing",
            "model": "Informer",
            "y_true": round(float(t), 4),
            "y_pred": round(max(0.0, float(p)), 4),
        })

    log.info("Informer  RMSE=%.2f  MAE=%.2f  MAPE=%.1f%%", rmse, mae, mape)
    return {
        "model": "Informer",
        "rmse": rmse, "mae": mae, "mape": mape,
        "n_train": int(len(x_enc_tr)), "n_test": int(len(x_enc_te)),
        "predictions": rows,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run()
    print(f"\n{'='*50}")
    print(
        f"Informer | RMSE={result['rmse']:.2f}  MAE={result['mae']:.2f}  "
        f"MAPE={result['mape']:.1f}%  (n={result['n_test']})"
    )
