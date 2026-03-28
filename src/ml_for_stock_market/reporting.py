from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .config import OutputConfig


def _ensure_parent(path: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_dataframes(
    feature_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
    outputs: OutputConfig,
) -> None:
    feature_frame.to_csv(_ensure_parent(outputs.feature_frame_path), index=False)
    prediction_frame.to_csv(_ensure_parent(outputs.predictions_path), index=False)
    forecast_metrics.to_csv(_ensure_parent(outputs.forecast_metrics_path), index=False)
    strategy_metrics.to_csv(_ensure_parent(outputs.strategy_metrics_path), index=False)


def plot_forecast_comparison(prediction_frame: pd.DataFrame, outputs: OutputConfig) -> None:
    tail = prediction_frame.tail(1_500).copy()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(tail["timestamp"], tail["target_vol"], label="Realized volatility", linewidth=1.8)
    ax.plot(tail["timestamp"], tail["garch_pred"], label="GARCH forecast", linewidth=1.2)
    ax.plot(tail["timestamp"], tail["lstm_pred"], label="LSTM forecast", linewidth=1.2)
    ax.set_title("Walk-forward volatility forecast comparison")
    ax.set_ylabel("Annualized volatility")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(_ensure_parent(outputs.forecast_plot_path), dpi=180)
    plt.close(fig)


def plot_equity_curve(strategy_frame: pd.DataFrame, outputs: OutputConfig) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    for name in ("buy_hold", "garch", "lstm"):
        equity = (1.0 + strategy_frame[f"{name}_return"].fillna(0.0)).cumprod()
        ax.plot(strategy_frame["timestamp"], equity, label=name)
    ax.set_title("Equity curves of volatility-managed strategies")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(_ensure_parent(outputs.equity_plot_path), dpi=180)
    plt.close(fig)


def plot_monte_carlo(monte_carlo_frame: pd.DataFrame, outputs: OutputConfig) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(monte_carlo_frame["simulated_annual_vol"], bins=40, color="#2b6cb0", alpha=0.8)
    ax.set_title("Monte Carlo distribution of next-period annualized volatility")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(_ensure_parent(outputs.monte_carlo_plot_path), dpi=180)
    plt.close(fig)


def write_summary(summary: dict[str, object], outputs: OutputConfig) -> None:
    _ensure_parent(outputs.summary_path).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
