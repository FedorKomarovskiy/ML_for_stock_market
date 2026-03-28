from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import ModelConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


class VolatilityLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden)


@dataclass(slots=True)
class Standardizer:
    mean_: np.ndarray
    std_: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean_ = values.mean(axis=0)
        std_ = values.std(axis=0)
        std_[std_ == 0.0] = 1.0
        return cls(mean_=mean_, std_=std_)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean_) / self.std_


@dataclass(slots=True)
class LstmBundle:
    model: VolatilityLSTM
    scaler: Standardizer
    sequence_length: int
    feature_columns: list[str]


def _build_sequences(
    scaled_features: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_list: list[np.ndarray] = []
    y_list: list[float] = []
    for idx in indices:
        start = idx - sequence_length + 1
        if start < 0:
            continue
        window = scaled_features[start : idx + 1]
        if np.isnan(window).any() or np.isnan(targets[idx]):
            continue
        x_list.append(window)
        y_list.append(np.log(max(float(targets[idx]), 1e-8)))
    if not x_list:
        return np.empty((0, sequence_length, scaled_features.shape[1])), np.empty((0,))
    return np.stack(x_list), np.asarray(y_list, dtype=float)


def train_lstm_bundle(
    frame: pd.DataFrame,
    feature_columns: list[str],
    train_indices: np.ndarray,
    cfg: ModelConfig,
    sequence_length: int,
) -> LstmBundle:
    if train_indices.size < sequence_length + 50:
        raise ValueError("Not enough rows to train the LSTM model.")

    set_seed(cfg.seed)
    feature_values = frame[feature_columns].to_numpy(dtype=float)
    scaler = Standardizer.fit(feature_values[train_indices])
    scaled_features = scaler.transform(feature_values)
    target_values = frame["target_vol"].to_numpy(dtype=float)

    usable_indices = train_indices[train_indices >= sequence_length - 1]
    val_size = max(1, int(len(usable_indices) * 0.1))
    train_core = usable_indices[:-val_size]
    val_core = usable_indices[-val_size:]
    if train_core.size == 0:
        raise ValueError("Training split became empty after validation holdout.")

    x_train, y_train = _build_sequences(scaled_features, target_values, train_core, sequence_length)
    x_val, y_val = _build_sequences(scaled_features, target_values, val_core, sequence_length)
    if len(x_train) == 0 or len(x_val) == 0:
        raise ValueError("Failed to build train or validation sequences for LSTM.")

    device = torch.device("cpu")
    model = VolatilityLSTM(
        input_size=len(feature_columns),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(SequenceDataset(x_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(x_val, y_val), batch_size=cfg.batch_size, shuffle=False)

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    patience_left = cfg.patience

    for _ in range(cfg.epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                val_losses.append(float(loss_fn(model(batch_x), batch_y).item()))
        mean_val = float(np.mean(val_losses))
        if mean_val < best_val:
            best_val = mean_val
            best_state = copy.deepcopy(model.state_dict())
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        raise RuntimeError("LSTM training did not produce a valid state.")
    model.load_state_dict(best_state)
    model.eval()
    return LstmBundle(
        model=model,
        scaler=scaler,
        sequence_length=sequence_length,
        feature_columns=feature_columns,
    )


def predict_with_lstm(bundle: LstmBundle, frame: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    feature_values = frame[bundle.feature_columns].to_numpy(dtype=float)
    scaled_features = bundle.scaler.transform(feature_values)
    target_values = frame["target_vol"].to_numpy(dtype=float)
    x_test, _ = _build_sequences(scaled_features, target_values, indices, bundle.sequence_length)
    if len(x_test) == 0:
        return np.empty((0,), dtype=float)

    device = torch.device("cpu")
    loader = DataLoader(torch.tensor(x_test, dtype=torch.float32), batch_size=256, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch_x in loader:
            outputs = bundle.model(batch_x.to(device)).cpu().numpy().reshape(-1)
            preds.append(outputs)
    return np.exp(np.concatenate(preds))
