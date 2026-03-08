"""Probability calibration utilities.

Two state-of-the-art post-hoc calibration strategies for imbalanced binary
classification (claim rate ~5%):

  - Beta       : Beta calibration (Kull, Silva Filho & Flach, 2017)
                 Smooth parametric family based on the Beta CDF.
                 Superior to Platt for skewed score distributions.

  - Venn-ABERS : Venn-ABERS predictor (Vovk et al., 2012; Johansson et al., 2021)
                 Part of the Conformal Prediction family. Produces a calibrated
                 probability *interval* [p0, p1] with finite-sample validity
                 guarantees. Uses two isotonic regressions — one per class —
                 but avoids the overfitting of plain isotonic calibration on
                 imbalanced data.

Why not Platt (sigmoid) or plain isotonic?
  - Platt assumes a logistic relationship between raw scores and true probs.
    CatBoost scores are not logistically distributed → poor fit.
  - Plain isotonic regression has high variance on small positive-class
    calibration sets and is not smooth.

References
----------
Kull, M., Silva Filho, T. M., & Flach, P. (2017). Beyond sigmoids:
    How to obtain well-calibrated probabilities from binary classifiers
    with beta calibration. *Electronic Journal of Statistics*.

Vovk, V., Petej, I., & Fedorova, V. (2012). Large-scale probabilistic
    predictors with and without guarantees of validity. *NeurIPS*.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from betacal import BetaCalibration
from sklearn.calibration import calibration_curve
from venn_abers import VennAbersCalibrator


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------

def calibrate_beta(
    y_prob_cal: np.ndarray,
    y_cal: np.ndarray,
    parameters: str = "abm",
) -> BetaCalibration:
    """Fit Beta calibration on held-out probability scores.

    Parameters
    ----------
    y_prob_cal : raw predicted probabilities from base model on cal set
    y_cal      : true binary labels on cal set
    parameters : 'abm' (full 3-param), 'ab', 'am', or 'bm' (constrained)

    Returns
    -------
    Fitted BetaCalibration — call .predict(scores.reshape(-1,1)) to transform.
    """
    bc = BetaCalibration(parameters=parameters)
    bc.fit(y_prob_cal.reshape(-1, 1), y_cal)
    return bc


def calibrate_venn_abers(
    estimator,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    random_state: int = 42,
) -> VennAbersCalibrator:
    """Fit a Venn-ABERS calibrator on a held-out calibration set.

    The base estimator must already be fitted (inductive mode). X_cal must
    be a genuine held-out set not used during model training or early stopping.

    Parameters
    ----------
    estimator : fitted classifier with predict_proba (e.g. CatBoostClassifier)
    X_cal     : calibration features
    y_cal     : calibration labels

    Returns
    -------
    Fitted VennAbersCalibrator — call .predict_proba(X)[:,1] for positive-class
    calibrated probabilities.
    """
    va = VennAbersCalibrator(
        estimator=estimator,
        inductive=True,
        random_state=random_state,
    )
    va.fit(X_cal, y_cal)
    return va


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Both bounded in [0, 1]; lower is better.
    """
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    bin_counts = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    weights = bin_counts / bin_counts.sum()
    errors = np.abs(frac_pos - mean_pred)
    return {
        "ece": float(np.sum(weights[: len(errors)] * errors)),
        "mce": float(errors.max()),
    }


def plot_reliability_diagram(
    probs_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    n_bins: int = 10,
    zoom_max: float = 0.30,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Reliability diagram comparing Beta vs Venn-ABERS calibration.

    Parameters
    ----------
    probs_dict : {label: (y_true, y_prob)}
    zoom_max   : upper limit for x/y axes (zoom into low-probability region)
    """
    ax = ax or plt.gca()
    colors = {
        "Uncalibrated":  "#999999",
        "Beta":          "#4472C4",
        "Venn-ABERS":    "#70AD47",
    }
    for label, (y_true, y_prob) in probs_dict.items():
        frac, mean_p = calibration_curve(y_true, y_prob, n_bins=n_bins)
        ece = calibration_metrics(y_true, y_prob)["ece"]
        ax.plot(
            mean_p, frac, "o-", lw=2, markersize=5,
            label=f"{label} (ECE={ece:.4f})",
            color=colors.get(label, "#ED7D31"),
        )
    ax.plot([0, zoom_max], [0, zoom_max], "k--", lw=1.2, label="Perfect")
    ax.set_xlim(0, zoom_max)
    ax.set_ylim(0, zoom_max)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Reliability Diagram\n(Beta vs Venn-ABERS — zoomed)")
    ax.legend(fontsize=9)
    return ax
