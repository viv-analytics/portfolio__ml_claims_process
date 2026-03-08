"""Smoke tests for src/claims utilities."""

import numpy as np
import pandas as pd
import pytest

from claims.evaluation import gini_coefficient, lorenz_curve, stp_routing, stp_summary
from claims.features import get_feature_names, catboost_cat_indices, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from claims.fraud.anomaly import create_fraud_proxy_labels
from claims.process.decisions import pure_premium


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def test_gini_perfect():
    y = np.array([0, 0, 1, 1])
    assert gini_coefficient(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0, abs=0.01)


def test_gini_random():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    g = gini_coefficient(y, rng.random(1000))
    assert abs(g) < 0.1  # random model ≈ 0


def test_lorenz_curve_bounds():
    y = np.array([0, 1, 0, 1, 1])
    s = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
    cum_exp, cum_claims = lorenz_curve(y, s)
    assert cum_exp[-1] == pytest.approx(1.0)
    assert cum_claims[-1] == pytest.approx(1.0)


def test_stp_routing_lanes():
    probs = np.array([0.05, 0.25, 0.60])
    lanes = stp_routing(probs)
    assert list(lanes) == ["auto-settle", "review", "investigate"]


def test_stp_summary_shape():
    probs = np.random.default_rng(0).random(200)
    y = (probs > 0.5).astype(int)
    summary = stp_summary(probs, y)
    assert set(summary.index) == {"auto-settle", "review", "investigate"}


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def test_feature_names_disjoint():
    names = get_feature_names()
    assert len(names) == len(set(names))


def test_catboost_cat_indices_valid():
    indices = catboost_cat_indices()
    all_feats = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert all(0 <= i < len(all_feats) for i in indices)


# ---------------------------------------------------------------------------
# fraud proxy labels
# ---------------------------------------------------------------------------

def test_fraud_proxy_labels():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "AvgSeverity": rng.exponential(scale=2000, size=500),
        "BonusMalus": rng.integers(50, 200, size=500),
        "VehAge": rng.integers(0, 15, size=500),
    })
    result = create_fraud_proxy_labels(df)
    assert "FraudProxy" in result.columns
    assert result["FraudProxy"].isin([0, 1]).all()
    assert result["FraudProxy"].sum() > 0  # at least some flagged


# ---------------------------------------------------------------------------
# pure premium
# ---------------------------------------------------------------------------

def test_pure_premium_positive():
    freq = np.array([0.1, 0.2, 0.05])
    sev = np.array([3000.0, 1500.0, 5000.0])
    pp = pure_premium(freq, sev)
    assert (pp >= 0).all()
    assert pp[0] == pytest.approx(300.0)
