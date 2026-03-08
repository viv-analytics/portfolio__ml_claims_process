"""Shared evaluation metrics used across all three modules.

Actuarial metrics (Lorenz curve, Gini coefficient) sit alongside
standard ML metrics (Brier score, Cohen's Kappa, AUC-PR).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    cohen_kappa_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def classification_report_dict(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return a dict of key classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "gini": gini_coefficient(y_true, y_prob),
    }


# ---------------------------------------------------------------------------
# Actuarial metrics
# ---------------------------------------------------------------------------

def gini_coefficient(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Gini coefficient = 2*AUC − 1.

    The actuarial standard for measuring model lift on insurance portfolios.
    A random model scores 0; a perfect model scores 1.
    """
    return 2.0 * roc_auc_score(y_true, y_score) - 1.0


def lorenz_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    exposure: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cumulative exposure and cumulative claims for Lorenz curve.

    Returns
    -------
    cum_exposure : ndarray — x-axis (0 to 1)
    cum_claims   : ndarray — y-axis (0 to 1)
    """
    df = pd.DataFrame({"target": y_true, "score": y_score})
    df["exposure"] = exposure if exposure is not None else 1.0
    df = df.sort_values("score")

    cum_exp = np.cumsum(df["exposure"].values) / df["exposure"].sum()
    cum_claims = np.cumsum(df["target"].values * df["exposure"].values)
    total_claims = cum_claims[-1]
    cum_claims = cum_claims / total_claims if total_claims > 0 else cum_claims

    return cum_exp, cum_claims


def plot_lorenz_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str = "Model",
    exposure: np.ndarray | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot Lorenz curve with Gini annotation."""
    ax = ax or plt.gca()
    cum_exp, cum_claims = lorenz_curve(y_true, y_score, exposure)
    gini = gini_coefficient(y_true, y_score)

    ax.plot(cum_exp, cum_claims, label=f"{model_name} (Gini={gini:.3f})", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random model")
    ax.set_xlabel("Cumulative Exposure Share")
    ax.set_ylabel("Cumulative Claims Share")
    ax.set_title("Lorenz Curve — Actuarial Lift")
    ax.legend()
    return ax


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def regression_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None)))),
    }


# ---------------------------------------------------------------------------
# STP routing
# ---------------------------------------------------------------------------

def stp_routing(
    y_prob: np.ndarray,
    low_thresh: float = 0.10,
    high_thresh: float = 0.40,
) -> pd.Series:
    """Assign each policy to an STP processing lane.

    Lanes
    -----
    auto-settle  : p < low_thresh   — low risk, instant payout
    review       : low ≤ p < high   — manual adjuster review
    investigate  : p ≥ high_thresh  — detailed investigation
    """
    lanes = np.where(
        y_prob < low_thresh, "auto-settle",
        np.where(y_prob < high_thresh, "review", "investigate"),
    )
    return pd.Series(lanes, name="stp_lane")


def stp_summary(y_prob: np.ndarray, y_true: np.ndarray | None = None) -> pd.DataFrame:
    """Return a summary table of STP routing with optional true-label enrichment."""
    lanes = stp_routing(y_prob)
    summary = lanes.value_counts().rename("n_policies").to_frame()
    summary["pct"] = (summary["n_policies"] / len(lanes) * 100).round(1)
    if y_true is not None:
        y_true_s = pd.Series(y_true, name="HasClaim")
        summary["claim_rate"] = (
            pd.concat([lanes, y_true_s], axis=1)
            .groupby("stp_lane")["HasClaim"]
            .mean()
            .round(4)
        )
    return summary.loc[["auto-settle", "review", "investigate"], :]
