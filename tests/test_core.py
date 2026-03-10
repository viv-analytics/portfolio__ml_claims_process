"""Smoke tests for src/claims utilities."""

import tempfile
import numpy as np
import pandas as pd
import pytest

from claims.evaluation import (
    gini_coefficient, lorenz_curve, stp_routing, stp_summary,
    gains_chart, lift_chart, decile_analysis,
)
from claims.features import get_feature_names, catboost_cat_indices, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from claims.fraud.anomaly import create_fraud_proxy_labels
from claims.process.decisions import pure_premium
from claims.data import temporal_split
from claims.config import load_config, ClaimsConfig


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


# ---------------------------------------------------------------------------
# evaluation — validators
# ---------------------------------------------------------------------------

def test_gini_invalid_labels():
    with pytest.raises(ValueError, match="0/1"):
        gini_coefficient(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]))


def test_gini_accepts_frequency_scores():
    # gini_coefficient is rank-based — Poisson frequency scores > 1 are valid
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.5, 0.9, 1.8])  # frequency prediction, may exceed 1
    g = gini_coefficient(y, scores)
    assert -1.0 <= g <= 1.0


def test_check_aligned_mismatch():
    with pytest.raises(ValueError, match="length"):
        gini_coefficient(np.array([0, 1]), np.array([0.1, 0.5, 0.9]))


# ---------------------------------------------------------------------------
# gains chart
# ---------------------------------------------------------------------------

def _make_ys(n=200, seed=7):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n)
    y_score = np.clip(y_true * 0.6 + rng.random(n) * 0.4, 0, 1)
    return y_true, y_score


def test_gains_chart_shape():
    y_true, y_score = _make_ys()
    gc = gains_chart(y_true, y_score, n_bins=10)
    assert gc.shape[0] == 10
    for col in ["decile", "n_policies", "n_positives", "cum_positives", "cum_exposure_pct", "cum_capture_rate"]:
        assert col in gc.columns


def test_gains_chart_cum_capture_monotone():
    y_true, y_score = _make_ys()
    gc = gains_chart(y_true, y_score, n_bins=10)
    assert (gc["cum_capture_rate"].diff().dropna() >= -1e-9).all()


def test_gains_chart_final_capture_100():
    y_true, y_score = _make_ys()
    gc = gains_chart(y_true, y_score, n_bins=10)
    assert gc["cum_capture_rate"].iloc[-1] == pytest.approx(100.0, abs=0.01)


# ---------------------------------------------------------------------------
# lift chart
# ---------------------------------------------------------------------------

def test_lift_chart_shape():
    y_true, y_score = _make_ys()
    lc = lift_chart(y_true, y_score, n_bins=10)
    assert lc.shape[0] == 10
    assert "lift" in lc.columns


# ---------------------------------------------------------------------------
# decile analysis
# ---------------------------------------------------------------------------

def test_decile_analysis_columns():
    y_true, y_score = _make_ys()
    da = decile_analysis(y_true, y_score, n_bins=10)
    for col in ["decile", "n_policies", "n_positives", "claim_rate", "lift", "cum_capture_rate"]:
        assert col in da.columns


# ---------------------------------------------------------------------------
# temporal split
# ---------------------------------------------------------------------------

def _make_temporal_df(n=100):
    return pd.DataFrame({
        "IDpol": np.arange(1, n + 1),
        "HasClaim": np.random.default_rng(0).integers(0, 2, n),
    })


def test_temporal_split_returns_correct_n_folds():
    df = _make_temporal_df()
    folds = temporal_split(df, n_folds=3)
    assert len(folds) == 3


def test_temporal_split_no_leakage():
    df = _make_temporal_df()
    for train_idx, test_idx in temporal_split(df, n_folds=3):
        assert len(np.intersect1d(train_idx, test_idx)) == 0


def test_temporal_split_train_grows():
    df = _make_temporal_df()
    folds = temporal_split(df, n_folds=3)
    train_sizes = [len(tr) for tr, _ in folds]
    assert train_sizes == sorted(train_sizes)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_load_config_returns_dataclass():
    cfg = load_config(path="/nonexistent/path.yaml")
    assert isinstance(cfg, ClaimsConfig)


def test_load_config_defaults():
    cfg = load_config(path="/nonexistent/path.yaml")
    assert cfg.fraud.fn_cost == 5000
    assert cfg.fraud.fp_cost == 150
    assert cfg.stp.low_threshold == pytest.approx(0.10)


def test_load_config_override():
    content = "fraud:\n  fn_cost: 9999\n  fp_cost: 200\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        tmp_path = f.name
    cfg = load_config(path=tmp_path)
    assert cfg.fraud.fn_cost == 9999
    assert cfg.fraud.fp_cost == 200
