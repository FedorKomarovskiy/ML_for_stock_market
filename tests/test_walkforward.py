from __future__ import annotations

import pandas as pd

from ml_for_stock_market.config import WalkForwardConfig
from ml_for_stock_market.walkforward import make_walk_forward_splits


def test_walk_forward_splits_are_ordered() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2022-01-01", periods=24 * 800, freq="h", tz="UTC"),
        }
    )
    cfg = WalkForwardConfig(train_days=365, test_days=90, step_days=90)
    splits = make_walk_forward_splits(frame, cfg)
    assert splits
    for split in splits:
        assert split.train_end_time <= split.test_start_time
        assert split.test_start_time < split.test_end_time
