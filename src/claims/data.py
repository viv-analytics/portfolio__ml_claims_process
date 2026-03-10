"""Data loading and preprocessing for the French Motor TPL dataset (freMTPL2).

The dataset covers 678k motor third-party liability policies and is loaded
directly from OpenML via scikit-learn — no manual download required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_fremtpl2(as_frame: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load freMTPL2 frequency and severity tables from OpenML.

    Returns
    -------
    freq : DataFrame  — one row per policy (678k rows, 12 columns)
    sev  : DataFrame  — one row per individual claim (26k rows, 2 columns)
    """
    freq = fetch_openml(data_id=41214, as_frame=as_frame, parser="auto").frame
    sev = fetch_openml(data_id=41215, as_frame=as_frame, parser="auto").frame
    return freq, sev


def build_claims_dataset(freq: pd.DataFrame, sev: pd.DataFrame) -> pd.DataFrame:
    """Join frequency and severity tables and engineer core targets.

    Targets created
    ---------------
    HasClaim        : int   — binary, did any claim occur? (classification)
    ClaimFrequency  : float — ClaimNb / Exposure (Poisson target)
    AvgSeverity     : float — mean claim amount per policy (NaN when no claim)
    PurePremium     : float — TotalClaimAmount / Exposure (loss cost)
    """
    # Aggregate claim amounts per policy
    sev_agg = (
        sev.groupby("IDpol")["ClaimAmount"]
        .agg(TotalClaimAmount="sum", NumClaims="count")
        .reset_index()
    )

    df = freq.merge(sev_agg, on="IDpol", how="left")
    df["TotalClaimAmount"] = df["TotalClaimAmount"].fillna(0.0)
    df["NumClaims"] = df["NumClaims"].fillna(0).astype(int)

    # Cast core columns
    df["ClaimNb"] = pd.to_numeric(df["ClaimNb"], errors="coerce").fillna(0).astype(int)
    df["Exposure"] = pd.to_numeric(df["Exposure"], errors="coerce").clip(lower=1e-6)

    # Targets
    df["HasClaim"] = (df["ClaimNb"] > 0).astype(int)
    df["ClaimFrequency"] = df["ClaimNb"] / df["Exposure"]
    df["AvgSeverity"] = np.where(
        df["ClaimNb"] > 0, df["TotalClaimAmount"] / df["ClaimNb"], np.nan
    )
    df["PurePremium"] = df["TotalClaimAmount"] / df["Exposure"]

    return df


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def split_dataset(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split (stratify on target for classification)."""
    stratify = df[target] if df[target].nunique() <= 10 else None
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def claims_only(df: pd.DataFrame) -> pd.DataFrame:
    """Return subset of policies that generated at least one claim."""
    return df[df["HasClaim"] == 1].reset_index(drop=True)


def temporal_split(
    df: pd.DataFrame,
    n_folds: int = 3,
    id_col: str = "IDpol",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Time-aware cross-validation using IDpol as a temporal proxy.

    Lower policy IDs correspond to earlier policies (earlier enrollment),
    making IDpol a reasonable surrogate for time when no date column exists.

    Parameters
    ----------
    df      : full dataset (must contain *id_col*)
    n_folds : number of walk-forward folds (≥ 2)
    id_col  : column used as time proxy (sorted ascending)

    Returns
    -------
    list of (train_idx, test_idx) tuples — positional indices for df.iloc[]
    Each successive fold extends the training window by one block.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    sorted_df = df.sort_values(id_col).reset_index(drop=True)
    n = len(sorted_df)
    n_blocks = n_folds + 1
    block_size = n // n_blocks

    # Build positional blocks; last block absorbs remainder
    blocks = [list(range(i * block_size, (i + 1) * block_size)) for i in range(n_blocks - 1)]
    blocks.append(list(range((n_blocks - 1) * block_size, n)))

    folds = []
    for k in range(n_folds):
        train_idx = np.concatenate([blocks[i] for i in range(k + 1)])
        test_idx = np.array(blocks[k + 1])
        folds.append((train_idx, test_idx))

    return folds
