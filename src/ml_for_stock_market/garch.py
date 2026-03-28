from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from arch import arch_model

from .config import FeatureConfig, GarchConfig


@dataclass(slots=True)
class GarchForecaster:
    omega: float
    alpha: float
    beta: float
    next_variance_pct2: float
    forecast_horizon_hours: int
    annualization_factor: int

    def update(self, observed_return: float) -> None:
        observed_pct = observed_return * 100.0
        self.next_variance_pct2 = (
            self.omega
            + self.alpha * observed_pct**2
            + self.beta * self.next_variance_pct2
        )

    def forecast_annualized_vol(self) -> float:
        variances = []
        current = self.next_variance_pct2
        for _ in range(self.forecast_horizon_hours):
            variances.append(current)
            current = self.omega + (self.alpha + self.beta) * current
        mean_variance = max(float(np.mean(variances)), 1e-12)
        return (math.sqrt(mean_variance) / 100.0) * math.sqrt(self.annualization_factor)


def fit_garch_forecaster(
    train_returns: np.ndarray,
    feature_cfg: FeatureConfig,
    garch_cfg: GarchConfig,
) -> GarchForecaster:
    clean_returns = np.asarray(train_returns, dtype=float)
    clean_returns = clean_returns[np.isfinite(clean_returns)]
    if clean_returns.size < 200:
        raise ValueError("At least 200 return observations are required for GARCH fitting.")

    scaled_returns = clean_returns * 100.0
    model = arch_model(
        scaled_returns,
        mean="Zero",
        vol="GARCH",
        p=garch_cfg.p,
        q=garch_cfg.q,
        dist="normal",
        rescale=False,
    )
    result = model.fit(disp="off")
    params = result.params
    omega = float(params["omega"])
    alpha = float(params[[name for name in params.index if name.startswith("alpha[")][0]])
    beta = float(params[[name for name in params.index if name.startswith("beta[")][0]])

    last_conditional_variance = float(result.conditional_volatility[-1] ** 2)
    last_return_pct = float(scaled_returns[-1])
    next_variance_pct2 = omega + alpha * last_return_pct**2 + beta * last_conditional_variance

    return GarchForecaster(
        omega=omega,
        alpha=alpha,
        beta=beta,
        next_variance_pct2=max(next_variance_pct2, 1e-10),
        forecast_horizon_hours=feature_cfg.forecast_horizon_hours,
        annualization_factor=feature_cfg.annualization_factor,
    )
