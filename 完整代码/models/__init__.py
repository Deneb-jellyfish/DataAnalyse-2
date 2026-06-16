"""Model implementations for hourly PM2.5 prediction.

Models:
- ARIMA: classical statistical baseline
- Prophet: trend + seasonality decomposition
- XGBoost: gradient-boosted trees (primary model)
- LSTM: recurrent neural network
- Transformer: self-attention based model
- Informer: optional long-sequence Transformer variant
"""

from models.transformer_model import (
    PositionalEncoding,
    TimeSeriesTransformer,
    TransformerTrainer,
)
