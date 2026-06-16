"""Transformer model for hourly PM2.5 prediction.

Uses FULL MultiheadAttention (not sparse/ProbSparse) for fair comparison with LSTM.
"""

from __future__ import annotations

import math
import copy
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Positional Encoding
# ==============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    pe[pos, 2i]   = sin(pos / 10000^(2i/d_model))
    pe[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer (not a parameter, but persisted in state_dict)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ==============================================================================
# TimeSeriesTransformer
# ==============================================================================

class TimeSeriesTransformer(nn.Module):
    """Full-attention Transformer for time-series regression.

    Architecture:
        1. Input projection: Linear(input_dim, d_model)
        2. Sinusoidal positional encoding
        3. TransformerEncoder (num_encoder_layers layers)
        4. Mean pooling over sequence dimension
        5. Output head: Linear(d_model, 1)
    """

    def __init__(
        self,
        input_dim: int = 20,
        seq_len: int = 24,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.output_dim = output_dim

        # 1. Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # 2. Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model, max_len=seq_len, dropout=dropout
        )

        # 3. TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # 4. Output head
        self.output_head = nn.Linear(d_model, output_dim)

        # Store config for serialization
        self._config = {
            "input_dim": input_dim,
            "seq_len": seq_len,
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "output_dim": output_dim,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            (batch,) when ``output_dim == 1`` else ``(batch, output_dim)``
        """
        # Project: (batch, seq_len, input_dim) -> (batch, seq_len, d_model)
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # TransformerEncoder: (batch, seq_len, d_model)
        x = self.transformer_encoder(x)

        # Mean pool over sequence dimension: (batch, d_model)
        x = x.mean(dim=1)

        # Output head: (batch, 1)
        x = self.output_head(x)
        if self.output_dim == 1:
            return x.squeeze(-1)
        return x

    @property
    def config(self) -> dict:
        return self._config


# ==============================================================================
# TransformerTrainer
# ==============================================================================

class TransformerTrainer:
    """Training harness for TimeSeriesTransformer.

    Handles optimizer, loss, training loop with early stopping,
    and prediction.
    """

    def __init__(
        self,
        model: TimeSeriesTransformer,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

    def _log(self, message: str) -> None:
        """Print lightweight timestamped training logs."""
        print(f"[Transformer {datetime.now():%H:%M:%S}] {message}", flush=True)

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 100,
        patience: int = 15,
        verbose: bool = True,
    ) -> dict:
        """Training loop with early stopping based on validation loss.

        Returns:
            dict with keys: best_epoch, best_val_loss, early_stop, epochs_run
        """
        best_val_loss = float("inf")
        best_model_state: Optional[dict] = None
        best_epoch = 0
        patience_counter = 0

        if verbose:
            self._log(
                f"start training: epochs={epochs}, patience={patience}, "
                f"train_batches={len(train_loader)}, val_batches={len(val_loader)}, "
                f"device={self.device}"
            )

        for epoch in range(1, epochs + 1):
            # --- Training ---
            self.model.train()
            train_losses: list[float] = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_losses))
            self.history["train_loss"].append(avg_train_loss)

            # --- Validation ---
            self.model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    preds = self.model(batch_x)
                    loss = self.criterion(preds, batch_y)
                    val_losses.append(loss.item())

            avg_val_loss = float(np.mean(val_losses))
            self.history["val_loss"].append(avg_val_loss)

            if verbose and (epoch % 10 == 0 or epoch == 1):
                self._log(
                    f"epoch {epoch:3d}/{epochs} "
                    f"train_loss={avg_train_loss:.4f} "
                    f"val_loss={avg_val_loss:.4f}"
                )

            # --- Early stopping ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
                if verbose:
                    self._log(
                        f"new best: epoch={epoch}, val_loss={avg_val_loss:.4f}"
                    )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    self._log(
                        f"early stopping at epoch={epoch}, best_epoch={best_epoch}"
                    )
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        if verbose:
            self._log(
                f"training complete: best_epoch={best_epoch}, "
                f"best_val_loss={best_val_loss:.4f}, epochs_run={epoch}"
            )

        result = {
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "early_stop": patience_counter >= patience,
            "epochs_run": epoch,
        }
        return result

    def predict(self, loader: torch.utils.data.DataLoader) -> np.ndarray:
        """Run inference and return predictions as a 1-D NumPy array."""
        self._log(f"start prediction: batches={len(loader)}")
        self.model.eval()
        all_preds: list[np.ndarray] = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                preds = self.model(batch_x)
                all_preds.append(preds.cpu().numpy())

        if all_preds:
            predictions = np.concatenate(all_preds)
            self._log(f"prediction complete: n_outputs={len(predictions)}")
            return predictions
        self._log("prediction complete: n_outputs=0")
        return np.array([])

    def get_history(self) -> dict[str, list[float]]:
        return self.history
