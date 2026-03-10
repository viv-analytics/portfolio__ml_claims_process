"""Centralised configuration for the claims pipeline.

All magic numbers live here.  A ``config.yaml`` at the project root can
override any default; if no file is found the dataclass defaults are used.

Usage
-----
>>> from claims.config import load_config
>>> cfg = load_config()
>>> cfg.fraud.fn_cost
5000
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/claims/config.py → project root


@dataclass
class FraudConfig:
    fn_cost: float = 5_000
    fp_cost: float = 150
    contamination: float = 0.05


@dataclass
class STPConfig:
    low_threshold: float = 0.10
    high_threshold: float = 0.40


@dataclass
class CatBoostConfig:
    iterations: int = 866
    learning_rate: float = 0.088
    depth: int = 7
    l2_leaf_reg: float = 4.04


@dataclass
class TuningConfig:
    n_trials: int = 50
    timeout: int = 300


@dataclass
class ClaimsConfig:
    fraud: FraudConfig = field(default_factory=FraudConfig)
    stp: STPConfig = field(default_factory=STPConfig)
    catboost: CatBoostConfig = field(default_factory=CatBoostConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)


def load_config(path: str | Path | None = None) -> ClaimsConfig:
    """Load configuration from a YAML file, falling back to dataclass defaults.

    Parameters
    ----------
    path : path to a YAML file.  If *None*, looks for ``config.yaml`` at the
           project root.  If the file does not exist, defaults are used.
    """
    if path is None:
        path = _PROJECT_ROOT / "config.yaml"

    path = Path(path)
    if not path.exists():
        return ClaimsConfig()

    try:
        import yaml  # defensive local import
    except ImportError:
        return ClaimsConfig()

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    fraud_raw = raw.get("fraud", {})
    stp_raw = raw.get("stp", {})
    cb_raw = raw.get("catboost", {})
    tuning_raw = raw.get("tuning", {})

    return ClaimsConfig(
        fraud=FraudConfig(
            fn_cost=fraud_raw.get("fn_cost", FraudConfig.fn_cost),
            fp_cost=fraud_raw.get("fp_cost", FraudConfig.fp_cost),
            contamination=fraud_raw.get("contamination", FraudConfig.contamination),
        ),
        stp=STPConfig(
            low_threshold=stp_raw.get("low_threshold", STPConfig.low_threshold),
            high_threshold=stp_raw.get("high_threshold", STPConfig.high_threshold),
        ),
        catboost=CatBoostConfig(
            iterations=cb_raw.get("iterations", CatBoostConfig.iterations),
            learning_rate=cb_raw.get("learning_rate", CatBoostConfig.learning_rate),
            depth=cb_raw.get("depth", CatBoostConfig.depth),
            l2_leaf_reg=cb_raw.get("l2_leaf_reg", CatBoostConfig.l2_leaf_reg),
        ),
        tuning=TuningConfig(
            n_trials=tuning_raw.get("n_trials", TuningConfig.n_trials),
            timeout=tuning_raw.get("timeout", TuningConfig.timeout),
        ),
    )
