from __future__ import annotations

import numpy as np
import pandas as pd

from ml_for_stock_market.config import StrategyConfig
from ml_for_stock_market.strategy import simulate_vol_targeting


def test_simulate_vol_targeting_returns_metrics() -> None:
    timestamps = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": np.linspace(40_000, 45_000, num=100),
            "forward_return_1h": np.random.default_rng(42).normal(0.0001, 0.01, size=100),
            "target_vol": np.full(100, 0.55),
            "garch_pred": np.full(100, 0.60),
            "lstm_pred": np.full(100, 0.50),
        }
    )
    strategy_frame, metrics = simulate_vol_targeting(
        frame,
        StrategyConfig(target_annual_vol=0.45, max_leverage=1.0, transaction_cost_bps=5.0),
    )
    assert len(metrics) == 3
    assert "lstm_return" in strategy_frame.columns
