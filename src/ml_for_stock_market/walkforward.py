from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FeatureConfig, GarchConfig, ModelConfig, WalkForwardConfig
from .deep_model import predict_with_lstm, train_lstm_bundle
from .features import FEATURE_COLUMNS
from .garch import fit_garch_forecaster


@dataclass(slots=True)
class Split:
    fold_id: int
    train_end_time: pd.Timestamp
    test_start_time: pd.Timestamp
    test_end_time: pd.Timestamp


def make_walk_forward_splits(
    frame: pd.DataFrame,
    cfg: WalkForwardConfig,
) -> list[Split]:
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    train_delta = pd.Timedelta(days=cfg.train_days)
    test_delta = pd.Timedelta(days=cfg.test_days)
    step_delta = pd.Timedelta(days=cfg.step_days)

    test_start = timestamps.min() + train_delta
    max_time = timestamps.max()
    fold_id = 1
    splits: list[Split] = []

    while test_start + test_delta <= max_time:
        splits.append(
            Split(
                fold_id=fold_id,
                train_end_time=test_start,
                test_start_time=test_start,
                test_end_time=test_start + test_delta,
            )
        )
        test_start = test_start + step_delta
        fold_id += 1
    return splits


def _compute_forecast_metrics(frame: pd.DataFrame, actual_col: str, pred_col: str) -> dict[str, float]:
    actual = frame[actual_col].to_numpy(dtype=float)
    pred = frame[pred_col].to_numpy(dtype=float)
    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mape = float(np.mean(np.abs((actual - pred) / np.clip(actual, 1e-8, None))) * 100.0)
    corr = float(np.corrcoef(actual, pred)[0, 1]) if len(actual) > 1 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "correlation": corr,
    }


def run_walk_forward(
    frame: pd.DataFrame,
    feature_cfg: FeatureConfig,
    walk_cfg: WalkForwardConfig,
    garch_cfg: GarchConfig,
    model_cfg: ModelConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = make_walk_forward_splits(frame, walk_cfg)
    if not splits:
        raise RuntimeError("No walk-forward splits were generated.")

    predictions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | int | str]] = []

    for split in splits:
        train_mask = frame["timestamp"] < split.test_start_time
        test_mask = (frame["timestamp"] >= split.test_start_time) & (frame["timestamp"] < split.test_end_time)
        train_indices = np.flatnonzero(train_mask.to_numpy())
        test_indices = np.flatnonzero(test_mask.to_numpy())
        if train_indices.size == 0 or test_indices.size == 0:
            continue

        garch = fit_garch_forecaster(
            train_returns=frame.loc[train_mask, "log_return"].to_numpy(dtype=float),
            feature_cfg=feature_cfg,
            garch_cfg=garch_cfg,
        )
        garch_preds = []
        for idx in test_indices:
            garch.update(float(frame.loc[idx, "log_return"]))
            garch_preds.append(garch.forecast_annualized_vol())

        bundle = train_lstm_bundle(
            frame=frame,
            feature_columns=FEATURE_COLUMNS,
            train_indices=train_indices,
            cfg=model_cfg,
            sequence_length=feature_cfg.sequence_length,
        )
        lstm_preds = predict_with_lstm(bundle=bundle, frame=frame, indices=test_indices)
        aligned_indices = test_indices[-len(lstm_preds) :]
        garch_preds = np.asarray(garch_preds[-len(lstm_preds) :], dtype=float)
        fold_frame = frame.loc[aligned_indices, ["timestamp", "close", "forward_return_1h", "target_vol"]].copy()
        fold_frame["fold_id"] = split.fold_id
        fold_frame["garch_pred"] = garch_preds
        fold_frame["lstm_pred"] = lstm_preds
        predictions.append(fold_frame)

        for model_name, pred_col in (("garch", "garch_pred"), ("lstm", "lstm_pred")):
            metrics = _compute_forecast_metrics(fold_frame, actual_col="target_vol", pred_col=pred_col)
            metric_rows.append(
                {
                    "fold_id": split.fold_id,
                    "model": model_name,
                    **metrics,
                }
            )

    if not predictions:
        raise RuntimeError("Walk-forward run produced no predictions.")

    prediction_frame = pd.concat(predictions, ignore_index=True)
    metrics_frame = pd.DataFrame(metric_rows)
    summary_metrics = (
        metrics_frame.groupby("model")[["mae", "rmse", "mape_pct", "correlation"]]
        .mean()
        .reset_index()
    )
    return prediction_frame, summary_metrics
