"""Feature engineering pipelines using scikit-learn ColumnTransformer.

CatBoost receives raw categoricals (OrdinalEncoder with unknown handling).
Logistic Regression receives scaled numerics + one-hot categoricals.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = ["VehAge", "DrivAge", "BonusMalus", "Density", "Exposure"]
CATEGORICAL_FEATURES = ["VehBrand", "VehGas", "Area", "Region", "VehPower"]
LOG_FEATURES = ["Density"]          # heavy-tailed → log1p transform
TARGET_COL = "HasClaim"

# Engineered numeric features (computed by engineer_features() below)
# These are used in place of NUMERIC_FEATURES for all models.
ENG_NUMERIC_FEATURES = [
    "VehAge", "DrivAge", "BonusMalus", "log_Density", "Exposure",
    "DrivAge_sq", "BM_excess", "YoungDriver", "SeniorDriver", "NewVehicle",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute engineered numeric features and add them to the DataFrame.

    Additions
    ---------
    log_Density   : log1p(Density) — corrects heavy right tail
    DrivAge_sq    : DrivAge² — captures U-shaped risk curve (young & senior)
    BM_excess     : max(BonusMalus - 100, 0) — penalised amount above neutral
    YoungDriver   : DrivAge < 25 (high-risk segment)
    SeniorDriver  : DrivAge > 70 (high-risk segment)
    NewVehicle    : VehAge < 2 (higher theft / total-loss probability)
    """
    df = df.copy()
    driv_age = pd.to_numeric(df["DrivAge"], errors="coerce")
    bm       = pd.to_numeric(df["BonusMalus"], errors="coerce")
    density  = pd.to_numeric(df["Density"], errors="coerce")
    veh_age  = pd.to_numeric(df["VehAge"], errors="coerce")

    df["log_Density"]  = np.log1p(density)
    df["DrivAge_sq"]   = driv_age ** 2
    df["BM_excess"]    = (bm - 100).clip(lower=0)
    df["YoungDriver"]  = (driv_age < 25).astype(float)
    df["SeniorDriver"] = (driv_age > 70).astype(float)
    df["NewVehicle"]   = (veh_age < 2).astype(float)

    eng_cols = ["log_Density", "DrivAge_sq", "BM_excess", "YoungDriver", "SeniorDriver", "NewVehicle"]
    nan_count = df[eng_cols].isna().sum().sum()
    if nan_count > 0:
        warnings.warn(
            f"engineer_features: {nan_count} NaN value(s) in engineered columns {eng_cols}. "
            "Check input for missing or non-numeric values.",
            UserWarning,
            stacklevel=2,
        )
    return df


# ---------------------------------------------------------------------------
# Sub-pipelines
# ---------------------------------------------------------------------------

def _log1p_pipe() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", StandardScaler()),
    ])


def _numeric_pipe() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def _ordinal_pipe() -> Pipeline:
    """OrdinalEncoder — compatible with CatBoost's cat_features indices."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])


# ---------------------------------------------------------------------------
# Public transformers
# ---------------------------------------------------------------------------

def get_preprocessor() -> ColumnTransformer:
    """Full preprocessing pipeline for linear models using engineered features.

    Call engineer_features(df) before using this preprocessor.
    ENG_NUMERIC_FEATURES are already transformed (log_Density, etc.) — only
    standard scaling is applied.
    """
    return ColumnTransformer(
        [
            ("num", _numeric_pipe(), ENG_NUMERIC_FEATURES),
            ("cat", _ordinal_pipe(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_catboost_preprocessor() -> ColumnTransformer:
    """Minimal preprocessor for CatBoost — only imputation, no scaling.

    CatBoost handles categoricals natively; we only need to fill nulls.
    """
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("cat", SimpleImputer(strategy="most_frequent"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names() -> list[str]:
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


def catboost_cat_indices() -> list[int]:
    """Return column indices of categorical features after get_catboost_preprocessor."""
    all_feats = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return [all_feats.index(f) for f in CATEGORICAL_FEATURES]


# ---------------------------------------------------------------------------
# Fraud-specific features
# ---------------------------------------------------------------------------

def add_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer interaction features useful for fraud detection."""
    df = df.copy()
    # High-risk driver + high bonus-malus score
    df["HighRiskDriver"] = ((df["DrivAge"] < 25) | (df["DrivAge"] > 75)).astype(int)
    # Density buckets (urban vs rural risk)
    df["UrbanArea"] = (pd.to_numeric(df["Density"], errors="coerce") > 500).astype(int)
    # BonusMalus above base (100 = neutral; >100 = penalised driver)
    df["PenalisedDriver"] = (pd.to_numeric(df["BonusMalus"], errors="coerce") > 100).astype(int)
    # Interaction: penalised + urban
    df["RiskInteraction"] = df["PenalisedDriver"] * df["UrbanArea"]
    return df
