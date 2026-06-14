"""LSTM model and trainer for hourly PM2.5 prediction.

Architecture:
  - 2-layer LSTM with configurable hidden dim and dropout
  - Final linear projection to one or many forecast steps
  - Trainer with Adam, MSE loss, and early stopping
"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMModel(nn.Module):
    """2-layer LSTM for hourly PM2.5 regression.

    Parameters
    ----------
    input_dim : int
        Number of input features per timestep.
    hidden_dim : int
        LSTM hidden state dimension.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout applied between LSTM layers (ignored when num_layers == 1).
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Shape (batch,) when ``output_dim == 1`` else ``(batch, output_dim)``.
        """
        lstm_out, _ = self.lstm(x)                       # (batch, seq_len, hidden_dim)
        last_hidden = lstm_out[:, -1, :]                  # (batch, hidden_dim)
        output = self.fc(last_hidden)
        if self.output_dim == 1:
            return output.squeeze(-1)
        return output


class LSTMTrainer:
    """Training harness with Adam optimizer, MSE loss, and early stopping.

    Parameters
    ----------
    model : LSTMModel
        The model instance to train.
    lr : float
        Learning rate for Adam.
    device : str
        Torch device string (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        model: LSTMModel,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.best_model_state: Optional[dict] = None
        self.best_val_loss: float = float("inf")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 15,
    ) -> dict[str, list[float]]:
        """Run the training loop with early stopping.

        Parameters
        ----------
        train_loader : DataLoader
            Training batches.
        val_loader : DataLoader
            Validation batches.
        epochs : int
            Maximum number of epochs.
        patience : int
            Stop after this many epochs without validation improvement.

        Returns
        -------
        dict
            ``{"train_loss": [...], "val_loss": [...]}``.
        """
        self.model.train()
        patience_counter = 0

        for epoch in range(epochs):
            # ---- training ----
            self.model.train()
            train_batch_losses: list[float] = []
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_batch_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_batch_losses))
            self.history["train_loss"].append(avg_train_loss)

            # ---- validation ----
            self.model.eval()
            val_batch_losses: list[float] = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    y_pred = self.model(X_batch)
                    loss = self.criterion(y_pred, y_batch)
                    val_batch_losses.append(loss.item())

            avg_val_loss = float(np.mean(val_batch_losses))
            self.history["val_loss"].append(avg_val_loss)

            # ---- early stopping ----
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch + 1:>3d}/{epochs}: "
                    f"train_loss={avg_train_loss:.4f}, "
                    f"val_loss={avg_val_loss:.4f}"
                )

        # Restore best checkpoint
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self.history

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, loader: DataLoader) -> np.ndarray:
        """Return 1-D numpy array of predictions for all batches.

        Parameters
        ----------
        loader : DataLoader
            DataLoader yielding (X, y) tuples.  y is ignored.

        Returns
        -------
        np.ndarray
            Shape (n_samples,).
        """
        self.model.eval()
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                y_pred = self.model(X_batch)
                predictions.append(y_pred.cpu().numpy())
        return np.concatenate(predictions) if predictions else np.array([])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Pickle trainer state (model, optimizer, history)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_state_dict": self.model.state_dict(),
            "best_model_state_dict": self.best_model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "best_val_loss": self.best_val_loss,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(
        cls, path: str | Path, model: Optional[LSTMModel] = None
    ) -> "LSTMTrainer":
        """Restore a pickled trainer.

        Parameters
        ----------
        path : str or Path
            Pickle file path.
        model : LSTMModel, optional
            If provided, load weights into this instance; otherwise a default
            LSTMModel() is created.

        Returns
        -------
        LSTMTrainer
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        if model is None:
            model = LSTMModel()

        model.load_state_dict(
            state["best_model_state_dict"] or state["model_state_dict"]
        )

        trainer = cls(model)
        trainer.best_model_state = state["best_model_state_dict"]
        trainer.best_val_loss = state["best_val_loss"]
        trainer.history = state["history"]
        return trainer
