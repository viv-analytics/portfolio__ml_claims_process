"""Unsupervised anomaly detection for fraud ring identification.

Two complementary detectors are combined:
  - Isolation Forest : global anomaly detector, fast on large data
  - Local Outlier Factor : density-based, catches local anomalies

Their normalised scores are averaged into a single AnomalyScore [0, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler


def create_fraud_proxy_labels(
    df: pd.DataFrame,
    severity_col: str = "AvgSeverity",
    percentile: float = 97.5,
) -> pd.DataFrame:
    """Create synthetic fraud labels using actuarial domain rules.

    Three rule-based triggers — a claim is suspicious if ANY rule fires:
      Rule 1 : ClaimAmount exceeds the {percentile}th percentile
      Rule 2 : Penalised driver (BonusMalus > 100) with high severity
      Rule 3 : New vehicle (VehAge < 2) with very high severity

    Note
    ----
    Labels are a *proxy*, not ground truth.  They let us demonstrate a
    supervised pipeline; in production these would be adjuster-verified flags.
    """
    df = df.copy()
    sev = pd.to_numeric(df[severity_col], errors="coerce")
    bm = pd.to_numeric(df.get("BonusMalus", pd.Series(np.nan, index=df.index)), errors="coerce")
    va = pd.to_numeric(df.get("VehAge", pd.Series(np.nan, index=df.index)), errors="coerce")

    p_high = np.nanpercentile(sev, percentile)
    p_90 = np.nanpercentile(sev, 90)
    p_95 = np.nanpercentile(sev, 95)

    rule1 = sev > p_high
    rule2 = (bm > 100) & (sev > p_90)
    rule3 = (va < 2) & (sev > p_95)

    df["FraudProxy"] = (rule1 | rule2 | rule3).astype(int)
    return df


def compute_anomaly_scores(
    X: np.ndarray,
    contamination: float = 0.05,
    n_estimators: int = 200,
    n_neighbors: int = 20,
    random_state: int = 42,
) -> np.ndarray:
    """Two-stage anomaly scoring: Isolation Forest + LOF → combined score.

    Returns
    -------
    anomaly_score : ndarray of shape (n_samples,), range [0, 1]
        Higher values indicate more anomalous (potentially fraudulent) claims.
    """
    # Isolation Forest — negative decision_function = higher anomaly
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso_raw = -iso.fit(X).decision_function(X)

    # Local Outlier Factor (novelty=True for predict on new data later)
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False,
        n_jobs=-1,
    )
    lof_raw = -lof.fit_predict(X).astype(float)  # +1 inlier / -1 outlier → flip
    lof_raw = (-lof.negative_outlier_factor_)     # raw score (higher = more anomalous)

    scaler = MinMaxScaler()
    iso_norm = scaler.fit_transform(iso_raw.reshape(-1, 1)).ravel()
    lof_norm = scaler.fit_transform(lof_raw.reshape(-1, 1)).ravel()

    return 0.5 * iso_norm + 0.5 * lof_norm
