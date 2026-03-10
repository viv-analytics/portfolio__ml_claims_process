"""Optuna hyperparameter tuning for CatBoost and Logistic Regression.

Usage
-----
>>> from claims.classification.tuning import tune_catboost
>>> best = tune_catboost(X_train, y_train, X_val, y_val)
>>> model = CatBoostClassifier(**best)
"""

from __future__ import annotations

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

import numpy as np
from sklearn.metrics import roc_auc_score


def tune_catboost(
    X_train,
    y_train,
    X_val,
    y_val,
    cat_features: list[int] | None = None,
    n_trials: int = 50,
    timeout: int = 300,
) -> dict:
    """Tune CatBoost hyperparameters using Optuna TPE.

    Optimises ROC-AUC on the validation set with CatBoost early stopping.

    Parameters
    ----------
    X_train, y_train : training data
    X_val, y_val     : held-out validation data (used for early stopping)
    cat_features     : list of categorical feature column indices
    n_trials         : number of Optuna trials
    timeout          : wall-clock budget in seconds

    Returns
    -------
    dict of best hyperparameters (ready to pass to CatBoostClassifier)
    """
    from catboost import CatBoostClassifier

    def objective(trial: optuna.Trial) -> float:
        params = {
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "iterations": trial.suggest_int("iterations", 200, 1000),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "eval_metric": "AUC",
            "random_seed": 42,
            "verbose": 0,
            "early_stopping_rounds": 50,
        }
        model = CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_features,
        )
        y_prob = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study.best_params


def tune_logistic(
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials: int = 30,
) -> dict:
    """Tune Logistic Regression hyperparameters using Optuna TPE.

    Wraps the full get_preprocessor() pipeline internally.

    Parameters
    ----------
    X_train, y_train : training DataFrames (before preprocessing)
    X_val, y_val     : validation DataFrames

    Returns
    -------
    dict of best hyperparameters: {"C": float, "penalty": str}
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from claims.features import get_preprocessor

    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 1e-4, 100, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])

        pipe = Pipeline([
            ("prep", get_preprocessor()),
            ("clf", LogisticRegression(
                C=C, penalty=penalty, solver="saga",
                max_iter=1000, random_state=42,
            )),
        ])
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
