"""Quantile regression for reserve estimation.

CatBoost's native Quantile loss produces P50 / P75 / P95 reserve estimates
with uncertainty bands — enabling actuarially-sound capital allocation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor

from claims.features import catboost_cat_indices

RESERVE_QUANTILES = [0.50, 0.75, 0.95]


def quantile_model(
    quantile: float = 0.75,
    iterations: int = 400,
    learning_rate: float = 0.05,
    depth: int = 5,
    verbose: int = 0,
) -> CatBoostRegressor:
    """CatBoost quantile regressor for reserve estimation at a given percentile.

    CatBoost's ``Quantile:alpha=q`` loss minimises the pinball loss, which
    gives asymmetric penalties above/below the quantile — directly interpretable
    as reserve adequacy at confidence level q.

    Pass cat_features=CATEGORICAL_FEATURES in the .fit() call.
    """
    return CatBoostRegressor(
        loss_function=f"Quantile:alpha={quantile}",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        random_seed=42,
        verbose=verbose,
    )


def fit_reserve_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    quantiles: list[float] = RESERVE_QUANTILES,
) -> dict[float, CatBoostRegressor]:
    """Fit one quantile model per reserve level."""
    models = {}
    for q in quantiles:
        m = quantile_model(quantile=q)
        m.fit(X_train, y_train, eval_set=(X_eval, y_eval), verbose=False)
        models[q] = m
    return models


def reserve_predictions(
    X: np.ndarray,
    models: dict[float, CatBoostRegressor],
) -> pd.DataFrame:
    """Return a DataFrame with P50 / P75 / P95 reserve estimates per policy."""
    preds = {f"P{int(q * 100)}": m.predict(X) for q, m in models.items()}
    return pd.DataFrame(preds)


def plot_reserve_bands(
    y_true: np.ndarray,
    reserves: pd.DataFrame,
    n_samples: int = 200,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter plot of actual vs predicted with P50/P75/P95 bands."""
    ax = ax or plt.gca()
    idx = np.random.default_rng(0).integers(0, len(y_true), size=min(n_samples, len(y_true)))

    ax.scatter(reserves.iloc[idx]["P50"], y_true[idx], alpha=0.4, s=10, label="Actual vs P50")
    ax.fill_betweenx(
        np.sort(y_true[idx]),
        np.sort(reserves.iloc[idx]["P50"]),
        np.sort(reserves.iloc[idx]["P95"]),
        alpha=0.2, label="P50–P95 band",
    )
    ax.set_xlabel("Predicted Reserve")
    ax.set_ylabel("Actual Claim Amount (€)")
    ax.set_title("Reserve Estimates with Uncertainty Bands")
    ax.legend()
    return ax
