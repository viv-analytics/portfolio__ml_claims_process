"""Classification models for claim occurrence prediction.

Three estimators are compared:
  1. Logistic Regression   — interpretable GLM baseline
  2. Random Forest         — tree ensemble with class balancing
  3. CatBoost              — gradient boosting with native categoricals
"""

from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from claims.features import get_preprocessor, get_catboost_preprocessor, catboost_cat_indices


def logistic_pipeline(class_weight: str = "balanced", C: float = 1.0) -> Pipeline:
    """Logistic Regression with full preprocessing (scaling + encoding)."""
    return Pipeline([
        ("preprocessor", get_preprocessor()),
        ("clf", LogisticRegression(
            class_weight=class_weight,
            C=C,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )),
    ])


def random_forest_pipeline(
    n_estimators: int = 300,
    max_depth: int | None = 8,
    class_weight: str = "balanced",
) -> Pipeline:
    """Random Forest with balanced class weights."""
    return Pipeline([
        ("preprocessor", get_preprocessor()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=42,
        )),
    ])


def catboost_classifier(
    iterations: int = 500,
    learning_rate: float = 0.05,
    depth: int = 6,
    scale_pos_weight: float | None = None,
    verbose: int = 0,
) -> CatBoostClassifier:
    """CatBoost classifier with native categorical feature support.

    Notes
    -----
    Pass X as a pandas DataFrame with raw string categoricals and specify
    ``cat_features=CATEGORICAL_FEATURES`` in the ``.fit()`` call.
    cat_features is intentionally omitted from the constructor to avoid
    index/name conflicts when X is a DataFrame.

    Parameters
    ----------
    scale_pos_weight : ratio of negative to positive class counts.
        Set to (n_neg / n_pos) when classes are heavily imbalanced.
    """
    params: dict = dict(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        eval_metric="AUC",
        random_seed=42,
        verbose=verbose,
        early_stopping_rounds=50,
    )
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight

    return CatBoostClassifier(**params)


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Compute scale_pos_weight = n_neg / n_pos for imbalanced binary targets."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float(n_neg / max(n_pos, 1))
