from __future__ import annotations

import numpy as np
import pandas as pd

from ml_for_stock_market.config import FeatureConfig
from ml_for_stock_market.features import FEATURE_COLUMNS, build_feature_frame


def test_build_feature_frame_creates_target_and_features() -> None:
    timestamps = pd.date_range("2024-01-01", periods=400, freq="h", tz="UTC")
    close = np.linspace(100.0, 140.0, num=400) + np.sin(np.arange(400))
    raw = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1_000, 2_000, num=400),
            "turnover": np.linspace(100_000, 200_000, num=400),
        }
    )
    cfg = FeatureConfig(forecast_horizon_hours=24, sequence_length=48, annualization_factor=8760)
    frame = build_feature_frame(raw, cfg)
    assert not frame.empty
    assert "target_vol" in frame.columns
    assert set(FEATURE_COLUMNS).issubset(frame.columns)
    assert (frame["target_vol"] > 0).all()
