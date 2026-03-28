from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import MonteCarloConfig


def run_volatility_monte_carlo(
    latest_annual_vol: float,
    historical_hourly_returns: pd.Series,
    cfg: MonteCarloConfig,
    annualization_factor: int = 8760,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(cfg.seed)
    hourly_sigma = max(latest_annual_vol, 1e-8) / math.sqrt(annualization_factor)
    drift = float(historical_hourly_returns.dropna().mean())

    simulations = rng.normal(
        loc=drift,
        scale=hourly_sigma,
        size=(cfg.paths, cfg.horizon_hours),
    )
    realized_annual_vol = np.sqrt(np.mean(simulations**2, axis=1) * annualization_factor)
    frame = pd.DataFrame({"simulated_annual_vol": realized_annual_vol})
    summary = {
        "mean_annual_vol": float(frame["simulated_annual_vol"].mean()),
        "median_annual_vol": float(frame["simulated_annual_vol"].median()),
        "p05_annual_vol": float(frame["simulated_annual_vol"].quantile(0.05)),
        "p95_annual_vol": float(frame["simulated_annual_vol"].quantile(0.95)),
    }
    return frame, summary
