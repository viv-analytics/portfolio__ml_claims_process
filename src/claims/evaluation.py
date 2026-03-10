"""Shared evaluation metrics used across all three modules.

Actuarial metrics (Lorenz curve, Gini coefficient) sit alongside
standard ML metrics (Brier score, Cohen's Kappa, AUC-PR).
"""

from __future__ import annotations

import logging

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

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private validators
# ---------------------------------------------------------------------------

def _check_binary(arr: np.ndarray, name: str = "y_true") -> None:
    """Raise ValueError if *arr* contains values other than 0 and 1."""
    unique = np.unique(arr)
    if not np.isin(unique, [0, 1]).all():
        raise ValueError(
            f"{name} must contain only 0/1 labels; got unique values {unique.tolist()}"
        )


def _check_probs(arr: np.ndarray, name: str = "y_prob") -> None:
    """Raise ValueError if *arr* contains values outside [0, 1]."""
    if np.any(arr < 0) or np.any(arr > 1):
        raise ValueError(
            f"{name} must be in [0, 1]; got min={arr.min():.4f}, max={arr.max():.4f}"
        )


def _check_aligned(*arrays: np.ndarray, names: list[str] | None = None) -> None:
    """Raise ValueError if arrays have different shapes."""
    shapes = [np.asarray(a).shape for a in arrays]
    if len(set(s[0] for s in shapes)) > 1:
        labels = names or [f"array_{i}" for i in range(len(arrays))]
        detail = ", ".join(f"{n}: {s}" for n, s in zip(labels, shapes))
        raise ValueError(f"Arrays must have the same length; got {detail}")


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

    Notes
    -----
    y_score is rank-based — any monotonic scores work, including Poisson
    frequency predictions > 1.  Only binary y_true is validated.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    _check_binary(y_true, "y_true")
    _check_aligned(y_true, y_score, names=["y_true", "y_score"])
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

    Notes
    -----
    y_score is rank-based — Poisson frequency predictions > 1 are valid inputs.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    _check_binary(y_true, "y_true")
    _check_aligned(y_true, y_score, names=["y_true", "y_score"])

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
    y_prob = np.asarray(y_prob)
    _check_probs(y_prob, "y_prob")
    n = len(y_prob)
    log.info("stp_routing: routing %d policies (low=%.2f, high=%.2f)", n, low_thresh, high_thresh)
    lanes = np.where(
        y_prob < low_thresh, "auto-settle",
        np.where(y_prob < high_thresh, "review", "investigate"),
    )
    return pd.Series(lanes, name="stp_lane")


def stp_summary(y_prob: np.ndarray, y_true: np.ndarray | None = None) -> pd.DataFrame:
    """Return a summary table of STP routing with optional true-label enrichment."""
    all_lanes = ["auto-settle", "review", "investigate"]
    lanes = stp_routing(y_prob)
    summary = (
        lanes.value_counts()
        .rename("n_policies")
        .reindex(all_lanes, fill_value=0)
        .to_frame()
    )
    summary["pct"] = (summary["n_policies"] / len(lanes) * 100).round(1)
    if y_true is not None:
        y_true_s = pd.Series(y_true, name="HasClaim")
        rates = (
            pd.concat([lanes, y_true_s], axis=1)
            .groupby("stp_lane")["HasClaim"]
            .mean()
            .round(4)
            .reindex(all_lanes, fill_value=np.nan)
        )
        summary["claim_rate"] = rates
    return summary


# ---------------------------------------------------------------------------
# Gains / Lift charts
# ---------------------------------------------------------------------------

def gains_chart(
    y_true: np.ndarray,
    y_score: np.ndarray,
    exposure: np.ndarray | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute gains chart data sorted by descending score.

    Returns
    -------
    DataFrame with columns:
        decile, n_policies, n_positives, cum_positives,
        cum_exposure_pct, cum_capture_rate
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    _check_binary(y_true, "y_true")
    _check_probs(y_score, "y_score")
    _check_aligned(y_true, y_score, names=["y_true", "y_score"])

    df = pd.DataFrame({"y_true": y_true, "y_score": y_score})
    if exposure is not None:
        df["exposure"] = np.asarray(exposure)
    else:
        df["exposure"] = 1.0

    # Sort descending by score, use positional index to avoid duplicate-edge issue
    df = df.sort_values("y_score", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, n_bins, labels=False) + 1

    grp = df.groupby("decile", sort=True)
    result = pd.DataFrame({
        "decile": grp["y_true"].count().index,
        "n_policies": grp["y_true"].count().values,
        "n_positives": grp["y_true"].sum().values,
        "exposure": grp["exposure"].sum().values,
    })

    total_positives = result["n_positives"].sum()
    total_exposure = result["exposure"].sum()

    result["cum_positives"] = result["n_positives"].cumsum()
    result["cum_exposure_pct"] = (result["exposure"].cumsum() / total_exposure * 100).round(2)
    result["cum_capture_rate"] = (
        result["cum_positives"] / total_positives * 100
        if total_positives > 0 else 0.0
    ).round(2)

    log.info("gains_chart: computed %d-decile gains chart (total positives=%d)", n_bins, int(total_positives))
    return result.drop(columns=["exposure"]).reset_index(drop=True)


def lift_chart(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute lift chart: claim rate per decile vs overall rate.

    Returns
    -------
    DataFrame with columns: decile, n_policies, claim_rate, lift
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    _check_binary(y_true, "y_true")
    _check_probs(y_score, "y_score")
    _check_aligned(y_true, y_score, names=["y_true", "y_score"])

    df = pd.DataFrame({"y_true": y_true, "y_score": y_score})
    df = df.sort_values("y_score", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, n_bins, labels=False) + 1

    overall_rate = y_true.mean()
    grp = df.groupby("decile", sort=True)

    result = pd.DataFrame({
        "decile": grp["y_true"].count().index,
        "n_policies": grp["y_true"].count().values,
        "claim_rate": grp["y_true"].mean().round(4).values,
    })
    result["lift"] = (result["claim_rate"] / overall_rate).round(4) if overall_rate > 0 else 0.0

    log.info("lift_chart: overall claim rate=%.4f, top-decile lift=%.2f", overall_rate, result["lift"].iloc[0])
    return result.reset_index(drop=True)


def decile_analysis(
    y_true: np.ndarray,
    y_score: np.ndarray,
    exposure: np.ndarray | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Full decile analysis: gains + lift joined on decile.

    Returns
    -------
    DataFrame with columns:
        decile, n_policies, n_positives, claim_rate, lift, cum_capture_rate
    """
    gc = gains_chart(y_true, y_score, exposure=exposure, n_bins=n_bins)
    lc = lift_chart(y_true, y_score, n_bins=n_bins)

    merged = gc[["decile", "n_policies", "n_positives", "cum_capture_rate"]].merge(
        lc[["decile", "claim_rate", "lift"]], on="decile"
    )
    return merged


def plot_gains_lift(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Dual-axis figure: gains curve (left) + lift bar chart (right)."""
    gc = gains_chart(y_true, y_score, n_bins=n_bins)
    lc = lift_chart(y_true, y_score, n_bins=n_bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Gains curve
    ax1.plot(gc["cum_exposure_pct"], gc["cum_capture_rate"], marker="o", lw=2, label="Model")
    ax1.plot([0, 100], [0, 100], "k--", lw=1, label="Random")
    ax1.set_xlabel("Cumulative Population %")
    ax1.set_ylabel("Cumulative Capture Rate %")
    ax1.set_title("Gains Chart")
    ax1.legend()

    # Lift bar chart
    ax2.bar(lc["decile"].astype(str), lc["lift"], color="steelblue")
    ax2.axhline(1.0, color="red", linestyle="--", lw=1, label="Baseline lift=1")
    ax2.set_xlabel("Decile (1 = highest score)")
    ax2.set_ylabel("Lift")
    ax2.set_title("Lift Chart")
    ax2.legend()

    fig.tight_layout()
    return fig
