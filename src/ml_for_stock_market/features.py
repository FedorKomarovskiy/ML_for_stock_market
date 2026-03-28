from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import FeatureConfig


FEATURE_COLUMNS = [
    "log_return",
    "abs_return",
    "range_pct",
    "volume_zscore_24h",
    "momentum_6h",
    "momentum_24h",
    "rv_6h",
    "rv_24h",
    "rv_72h",
    "rv_168h",
    "vol_ratio_6_24",
    "vol_ratio_24_168",
    "ema_gap_12_48",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


def _future_realized_vol(returns: pd.Series, horizon: int, annualization_factor: int) -> pd.Series:
    future_sq = pd.Series(0.0, index=returns.index, dtype=float)
    for step in range(1, horizon + 1):
        future_sq = future_sq.add(returns.shift(-step).pow(2), fill_value=0.0)
    return np.sqrt((future_sq / horizon) * annualization_factor)


def build_feature_frame(raw_frame: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    frame = raw_frame.copy().sort_values("timestamp").reset_index(drop=True)

    frame["log_return"] = np.log(frame["close"]).diff()
    frame["forward_return_1h"] = frame["close"].pct_change().shift(-1)
    frame["abs_return"] = frame["log_return"].abs()
    frame["range_pct"] = (frame["high"] - frame["low"]) / frame["close"].replace(0.0, np.nan)
    frame["volume_log"] = np.log1p(frame["volume"])

    annual_sqrt = math.sqrt(cfg.annualization_factor)
    frame["rv_6h"] = frame["log_return"].rolling(6).std() * annual_sqrt
    frame["rv_24h"] = frame["log_return"].rolling(24).std() * annual_sqrt
    frame["rv_72h"] = frame["log_return"].rolling(72).std() * annual_sqrt
    frame["rv_168h"] = frame["log_return"].rolling(168).std() * annual_sqrt

    volume_mean = frame["volume_log"].rolling(24).mean()
    volume_std = frame["volume_log"].rolling(24).std()
    frame["volume_zscore_24h"] = (frame["volume_log"] - volume_mean) / volume_std.replace(0.0, np.nan)

    frame["momentum_6h"] = frame["close"].pct_change(6)
    frame["momentum_24h"] = frame["close"].pct_change(24)
    ema_fast = frame["close"].ewm(span=12, adjust=False).mean()
    ema_slow = frame["close"].ewm(span=48, adjust=False).mean()
    frame["ema_gap_12_48"] = (ema_fast - ema_slow) / ema_slow.replace(0.0, np.nan)
    frame["vol_ratio_6_24"] = frame["rv_6h"] / frame["rv_24h"].replace(0.0, np.nan)
    frame["vol_ratio_24_168"] = frame["rv_24h"] / frame["rv_168h"].replace(0.0, np.nan)

    ts = pd.to_datetime(frame["timestamp"], utc=True)
    hour = ts.dt.hour.astype(float)
    dow = ts.dt.dayofweek.astype(float)
    frame["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    frame["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    frame["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    frame["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

    frame["target_vol"] = _future_realized_vol(
        frame["log_return"],
        horizon=cfg.forecast_horizon_hours,
        annualization_factor=cfg.annualization_factor,
    )
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return frame
