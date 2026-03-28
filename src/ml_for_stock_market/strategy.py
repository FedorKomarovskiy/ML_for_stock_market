from __future__ import annotations

import math

import pandas as pd

from .config import StrategyConfig


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def annualized_sharpe(returns: pd.Series, periods_per_year: int = 8760) -> float:
    clean = returns.dropna()
    if clean.empty or clean.std(ddof=0) == 0:
        return 0.0
    return float((clean.mean() / clean.std(ddof=0)) * math.sqrt(periods_per_year))


def cagr(returns: pd.Series, periods_per_year: int = 8760) -> float:
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    equity = float((1.0 + clean).prod())
    years = len(clean) / periods_per_year
    if years <= 0:
        return 0.0
    return equity ** (1.0 / years) - 1.0


def simulate_vol_targeting(
    prediction_frame: pd.DataFrame,
    cfg: StrategyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = prediction_frame.copy().sort_values("timestamp").reset_index(drop=True)
    strategies = {
        "buy_hold": pd.Series(1.0, index=frame.index, dtype=float),
        "garch": (cfg.target_annual_vol / frame["garch_pred"]).clip(lower=0.0, upper=cfg.max_leverage),
        "lstm": (cfg.target_annual_vol / frame["lstm_pred"]).clip(lower=0.0, upper=cfg.max_leverage),
    }
    cost_rate = cfg.transaction_cost_bps / 10_000.0
    metric_rows: list[dict[str, float | str]] = []

    for name, exposure in strategies.items():
        shifted = exposure.shift().fillna(exposure.iloc[0])
        turnover = exposure.diff().abs().fillna(0.0)
        strategy_returns = shifted * frame["forward_return_1h"].fillna(0.0) - turnover * cost_rate
        frame[f"{name}_exposure"] = exposure
        frame[f"{name}_return"] = strategy_returns
        metric_rows.append(
            {
                "model": name,
                "total_return_pct": float(((1.0 + strategy_returns).prod() - 1.0) * 100.0),
                "cagr_pct": cagr(strategy_returns) * 100.0,
                "sharpe": annualized_sharpe(strategy_returns),
                "max_drawdown_pct": max_drawdown(strategy_returns) * 100.0,
                "avg_exposure": float(exposure.mean()),
                "turnover": float(turnover.sum()),
            }
        )

    return frame, pd.DataFrame(metric_rows)
