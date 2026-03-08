"""Supervised fraud detection with cost-sensitive CatBoost.

In insurance, a missed fraud (FN) costs 10-50× more than a false alert (FP).
We encode this asymmetry via CatBoost's class_weights and a custom decision
threshold derived from the cost matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.metrics import precision_recall_curve


# Cost matrix (€ units, illustrative)
FN_COST = 5_000   # average fraudulent claim paid out
FP_COST = 150     # investigation cost per false alert


def cost_sensitive_catboost(
    cat_features: list[str] | None = None,
    fn_cost: float = FN_COST,
    fp_cost: float = FP_COST,
    iterations: int = 500,
    learning_rate: float = 0.05,
    depth: int = 6,
    verbose: int = 0,
) -> CatBoostClassifier:
    """CatBoost classifier with class weights derived from the cost matrix.

    scale_pos_weight = FN_COST / FP_COST encodes the asymmetric cost.
    Pass cat_features=CATEGORICAL_FEATURES in the .fit() call.
    """
    scale = fn_cost / fp_cost
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        scale_pos_weight=scale,
        eval_metric="AUC",
        random_seed=42,
        verbose=verbose,
        early_stopping_rounds=50,
    )


def optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fn_cost: float = FN_COST,
    fp_cost: float = FP_COST,
) -> float:
    """Find the decision threshold that minimises expected total cost.

    Expected cost = FP_cost * FP_rate + FN_cost * FN_rate
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    costs = []
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        tp = r * n_pos
        fn = (1 - r) * n_pos
        fp = (tp / p - tp) if p > 0 else n_neg
        costs.append(fn * fn_cost + fp * fp_cost)

    best_idx = int(np.argmin(costs))
    return float(thresholds[best_idx])


def plot_cost_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fn_cost: float = FN_COST,
    fp_cost: float = FP_COST,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot total cost as a function of decision threshold."""
    ax = ax or plt.gca()
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    costs = []
    for p, r in zip(precision[:-1], recall[:-1]):
        tp = r * n_pos
        fn = (1 - r) * n_pos
        fp = (tp / p - tp) if p > 0 else n_neg
        costs.append(fn * fn_cost + fp * fp_cost)

    opt_t = optimal_threshold(y_true, y_prob, fn_cost, fp_cost)
    ax.plot(thresholds, costs, lw=2)
    ax.axvline(opt_t, color="red", linestyle="--", label=f"Optimal threshold = {opt_t:.3f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Total Expected Cost (€)")
    ax.set_title("Cost Curve — Fraud Detection")
    ax.legend()
    return ax


def business_impact(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    fn_cost: float = FN_COST,
    fp_cost: float = FP_COST,
) -> pd.DataFrame:
    """Compute business impact vs. a no-model baseline (flag nothing)."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    baseline_cost = y_true.sum() * fn_cost   # all frauds go undetected
    model_cost = fn * fn_cost + fp * fp_cost
    savings = baseline_cost - model_cost

    return pd.DataFrame({
        "Metric": ["True Positives (fraud caught)", "False Positives (false alerts)",
                   "False Negatives (missed fraud)", "Baseline Cost (€)",
                   "Model Cost (€)", "Estimated Savings (€)"],
        "Value": [tp, fp, fn, f"{baseline_cost:,.0f}", f"{model_cost:,.0f}", f"{savings:,.0f}"],
    })
