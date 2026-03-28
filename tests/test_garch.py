from __future__ import annotations

import numpy as np

from ml_for_stock_market.config import FeatureConfig, GarchConfig
from ml_for_stock_market.garch import fit_garch_forecaster


def test_garch_forecaster_produces_positive_volatility() -> None:
    returns = np.random.default_rng(123).normal(0.0, 0.012, size=500)
    forecaster = fit_garch_forecaster(
        train_returns=returns,
        feature_cfg=FeatureConfig(forecast_horizon_hours=24, sequence_length=48, annualization_factor=8760),
        garch_cfg=GarchConfig(p=1, q=1),
    )
    forecaster.update(0.01)
    assert forecaster.forecast_annualized_vol() > 0
