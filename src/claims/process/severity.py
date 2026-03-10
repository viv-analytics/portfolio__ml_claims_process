"""Claim severity modeling: Gamma GLM baseline vs CatBoost regression.

Only policies with at least one claim are used (severity is undefined otherwise).
"""

from __future__ import annotations

import numpy as np
from catboost import CatBoostRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import Pipeline

from claims.features import get_preprocessor, catboost_cat_indices


def gamma_glm_pipeline() -> Pipeline:
    """Gamma GLM (TweedieRegressor power=2, log link) — actuarial baseline."""
    return Pipeline([
        ("preprocessor", get_preprocessor()),
        ("reg", TweedieRegressor(
            power=2,       # Gamma distribution
            alpha=0.5,
            link="log",
            max_iter=1000,
        )),
    ])


def catboost_severity(
    iterations: int = 866,
    learning_rate: float = 0.088,
    depth: int = 7,
    l2_leaf_reg: float = 4.04,
    verbose: int = 0,
) -> CatBoostRegressor:
    """CatBoost RMSE regressor for severity, with native categorical support.

    Defaults are Optuna-tuned values (50 trials, 300s).
    Pass cat_features=CATEGORICAL_FEATURES in the .fit() call.
    """
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=42,
        verbose=verbose,
        early_stopping_rounds=50,
    )


def poisson_glm_pipeline() -> Pipeline:
    """Poisson GLM for claim frequency (ClaimNb / Exposure)."""
    return Pipeline([
        ("preprocessor", get_preprocessor()),
        ("reg", TweedieRegressor(
            power=1,       # Poisson distribution
            alpha=0.5,
            link="log",
            max_iter=1000,
        )),
    ])


def catboost_frequency(
    iterations: int = 866,
    learning_rate: float = 0.088,
    depth: int = 7,
    l2_leaf_reg: float = 4.04,
    verbose: int = 0,
) -> CatBoostRegressor:
    """CatBoost Poisson regressor for claim frequency.

    Defaults are Optuna-tuned values (50 trials, 300s).
    Pass cat_features=CATEGORICAL_FEATURES in the .fit() call.
    """
    return CatBoostRegressor(
        loss_function="Poisson",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=42,
        verbose=verbose,
        early_stopping_rounds=50,
    )
