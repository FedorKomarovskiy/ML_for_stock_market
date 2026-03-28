from __future__ import annotations

import numpy as np
import pandas as pd

from ml_for_stock_market.config import MonteCarloConfig
from ml_for_stock_market.monte_carlo import run_volatility_monte_carlo


def test_monte_carlo_produces_distribution() -> None:
    history = pd.Series(np.random.default_rng(7).normal(0.0, 0.01, size=500))
    frame, summary = run_volatility_monte_carlo(
        latest_annual_vol=0.55,
        historical_hourly_returns=history,
        cfg=MonteCarloConfig(paths=200, horizon_hours=48, seed=42),
    )
    assert len(frame) == 200
    assert summary["p95_annual_vol"] >= summary["p05_annual_vol"]
