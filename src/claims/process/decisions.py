"""Decision support layer: pure premium, geographic risk, and fairness audit.

Pure premium = E[ClaimFrequency] × E[Severity]  (two-part actuarial model)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Pure Premium
# ---------------------------------------------------------------------------

def pure_premium(
    freq_pred: np.ndarray,
    sev_pred: np.ndarray,
) -> np.ndarray:
    """Compute loss cost = frequency × severity predictions."""
    return freq_pred * np.clip(sev_pred, 0, None)


def pure_premium_summary(
    y_actual: np.ndarray,
    pp_pred: np.ndarray,
    exposure: np.ndarray,
) -> pd.DataFrame:
    """Compare actual vs predicted loss cost at portfolio level."""
    actual_lc = (y_actual * exposure).sum() / exposure.sum()
    pred_lc = (pp_pred * exposure).sum() / exposure.sum()
    return pd.DataFrame({
        "Metric": ["Actual Loss Cost (€/policy)", "Predicted Loss Cost (€/policy)", "Ratio"],
        "Value": [f"{actual_lc:.2f}", f"{pred_lc:.2f}", f"{pred_lc / actual_lc:.4f}"],
    })


# ---------------------------------------------------------------------------
# Geographic risk
# ---------------------------------------------------------------------------

def regional_risk_profile(
    df: pd.DataFrame,
    region_col: str = "Region",
    freq_pred_col: str = "PredFrequency",
    sev_pred_col: str = "PredSeverity",
) -> pd.DataFrame:
    """Aggregate predicted risk metrics by region for choropleth mapping."""
    return (
        df.groupby(region_col)
        .agg(
            n_policies=(region_col, "count"),
            actual_claim_rate=("HasClaim", "mean"),
            pred_frequency=(freq_pred_col, "mean"),
            pred_severity=(sev_pred_col, "mean"),
        )
        .assign(pred_pure_premium=lambda d: d["pred_frequency"] * d["pred_severity"])
        .sort_values("pred_pure_premium", ascending=False)
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Fairness audit (EU Gender Directive context)
# ---------------------------------------------------------------------------

def fairness_audit(
    df: pd.DataFrame,
    pred_col: str,
    group_col: str,
    actual_col: str = "HasClaim",
) -> pd.DataFrame:
    """Compute per-group mean prediction and actual rate.

    Helps detect proxy discrimination (e.g., model uses VehBrand as a
    gender proxy in violation of EU Directive 2004/113/EC).
    """
    summary = (
        df.groupby(group_col)
        .agg(
            n=(group_col, "count"),
            actual_rate=(actual_col, "mean"),
            pred_mean=(pred_col, "mean"),
        )
        .reset_index()
    )
    overall_pred = df[pred_col].mean()
    summary["disparate_impact_ratio"] = summary["pred_mean"] / overall_pred
    return summary


def plot_fairness_bars(
    audit_df: pd.DataFrame,
    group_col: str,
    figsize: tuple[int, int] = (12, 4),
) -> plt.Figure:
    """Side-by-side bar chart of actual vs predicted rates per group."""
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(audit_df))
    width = 0.35
    ax.bar(x - width / 2, audit_df["actual_rate"], width, label="Actual rate")
    ax.bar(x + width / 2, audit_df["pred_mean"], width, label="Predicted rate", alpha=0.8)
    ax.axhline(audit_df["actual_rate"].mean(), ls="--", color="gray", label="Overall mean")
    ax.set_xticks(x)
    ax.set_xticklabels(audit_df[group_col], rotation=45, ha="right")
    ax.set_title(f"Fairness Audit by {group_col}")
    ax.legend()
    plt.tight_layout()
    return fig
