from __future__ import annotations

from pathlib import Path

from .config import load_config
from .data import load_or_download_data
from .features import build_feature_frame
from .monte_carlo import run_volatility_monte_carlo
from .reporting import (
    plot_equity_curve,
    plot_forecast_comparison,
    plot_monte_carlo,
    save_dataframes,
    write_summary,
)
from .strategy import simulate_vol_targeting
from .walkforward import run_walk_forward


def run_pipeline(config_path: str, force_download: bool = False) -> dict[str, object]:
    cfg = load_config(config_path)
    raw_frame = load_or_download_data(cfg.data, force=force_download)
    feature_frame = build_feature_frame(raw_frame, cfg.features)
    prediction_frame, forecast_metrics = run_walk_forward(
        frame=feature_frame,
        feature_cfg=cfg.features,
        walk_cfg=cfg.walk_forward,
        garch_cfg=cfg.garch,
        model_cfg=cfg.model,
    )
    strategy_frame, strategy_metrics = simulate_vol_targeting(prediction_frame, cfg.strategy)

    best_forecast_model = forecast_metrics.sort_values("rmse").iloc[0]["model"]
    latest_forecast = float(prediction_frame[f"{best_forecast_model}_pred"].iloc[-1])
    monte_carlo_frame, monte_carlo_summary = run_volatility_monte_carlo(
        latest_annual_vol=latest_forecast,
        historical_hourly_returns=prediction_frame["forward_return_1h"],
        cfg=cfg.monte_carlo,
        annualization_factor=cfg.features.annualization_factor,
    )

    save_dataframes(
        feature_frame=feature_frame,
        prediction_frame=prediction_frame,
        forecast_metrics=forecast_metrics,
        strategy_metrics=strategy_metrics,
        outputs=cfg.outputs,
    )
    plot_forecast_comparison(prediction_frame, cfg.outputs)
    plot_equity_curve(strategy_frame, cfg.outputs)
    plot_monte_carlo(monte_carlo_frame, cfg.outputs)

    summary = {
        "symbol": cfg.data.symbol,
        "rows_raw": int(len(raw_frame)),
        "rows_features": int(len(feature_frame)),
        "walk_forward_points": int(len(prediction_frame)),
        "best_forecast_model": best_forecast_model,
        "latest_forecast_annual_vol": latest_forecast,
        "forecast_metrics": forecast_metrics.to_dict(orient="records"),
        "strategy_metrics": strategy_metrics.to_dict(orient="records"),
        "monte_carlo": monte_carlo_summary,
        "artifacts": {
            "feature_frame_path": str(Path(cfg.outputs.feature_frame_path).resolve()),
            "predictions_path": str(Path(cfg.outputs.predictions_path).resolve()),
            "forecast_metrics_path": str(Path(cfg.outputs.forecast_metrics_path).resolve()),
            "strategy_metrics_path": str(Path(cfg.outputs.strategy_metrics_path).resolve()),
            "forecast_plot_path": str(Path(cfg.outputs.forecast_plot_path).resolve()),
            "equity_plot_path": str(Path(cfg.outputs.equity_plot_path).resolve()),
            "monte_carlo_plot_path": str(Path(cfg.outputs.monte_carlo_plot_path).resolve()),
        },
    }
    write_summary(summary, cfg.outputs)
    return summary
