from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DataConfig:
    symbol: str
    interval: str
    start_at: str
    end_at: str
    cache_path: str


@dataclass(slots=True)
class FeatureConfig:
    forecast_horizon_hours: int
    sequence_length: int
    annualization_factor: int


@dataclass(slots=True)
class WalkForwardConfig:
    train_days: int
    test_days: int
    step_days: int


@dataclass(slots=True)
class GarchConfig:
    p: int
    q: int


@dataclass(slots=True)
class ModelConfig:
    hidden_size: int
    num_layers: int
    dropout: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    seed: int


@dataclass(slots=True)
class StrategyConfig:
    target_annual_vol: float
    max_leverage: float
    transaction_cost_bps: float


@dataclass(slots=True)
class MonteCarloConfig:
    paths: int
    horizon_hours: int
    seed: int


@dataclass(slots=True)
class OutputConfig:
    feature_frame_path: str
    predictions_path: str
    forecast_metrics_path: str
    strategy_metrics_path: str
    summary_path: str
    forecast_plot_path: str
    equity_plot_path: str
    monte_carlo_plot_path: str


@dataclass(slots=True)
class ProjectConfig:
    data: DataConfig
    features: FeatureConfig
    walk_forward: WalkForwardConfig
    garch: GarchConfig
    model: ModelConfig
    strategy: StrategyConfig
    monte_carlo: MonteCarloConfig
    outputs: OutputConfig


def load_config(path: str | Path) -> ProjectConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return ProjectConfig(
        data=DataConfig(**raw["data"]),
        features=FeatureConfig(**raw["features"]),
        walk_forward=WalkForwardConfig(**raw["walk_forward"]),
        garch=GarchConfig(**raw["garch"]),
        model=ModelConfig(**raw["model"]),
        strategy=StrategyConfig(**raw["strategy"]),
        monte_carlo=MonteCarloConfig(**raw["monte_carlo"]),
        outputs=OutputConfig(**raw["outputs"]),
    )
