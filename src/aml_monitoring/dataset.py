"""Assemble the modelling dataset: raw months -> features -> temporal split.

One place that turns "which months" into ``X_train / y_train / X_val / ... / X_test``.
The notebook and the training script both call this so they can never diverge.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aml_monitoring.config import (
    ALL_MONTHS,
    ARTIFACTS_DIR,
    TARGET,
    TEST_MONTHS,
    TRAIN_MONTHS,
    VAL_MONTHS,
)
from aml_monitoring.data.load import load_months
from aml_monitoring.features.entity import ENTITY_FEATURES, add_entity_features
from aml_monitoring.features.transaction import (
    CATEGORICAL_COLS,
    TRANSACTION_FEATURES,
    add_transaction_features,
)

# The exact column set the model sees. Categoricals are kept as pandas
# 'category' dtype (set in load.py) so LightGBM handles them natively.
NUMERIC_FEATURES = TRANSACTION_FEATURES + ENTITY_FEATURES
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_COLS

# Kept aside for evaluation slicing — never fed to the model.
META_COLS = ["timestamp", "month", "Laundering_type"]


@dataclass
class Dataset:
    """A temporal split, plus test-set metadata for evaluation slices."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    meta_test: pd.DataFrame


def build_feature_frame(months: list[str]) -> pd.DataFrame:
    """Load ``months`` and attach every feature.

    Entity features are computed across the whole span passed in, so give a
    contiguous range that starts *before* the first training month: val/test
    rows then carry real account history, exactly as they would at inference.
    """
    df = load_months(months)
    df = add_transaction_features(df)
    df = add_entity_features(df)
    return df


def _xy(df: pd.DataFrame, months: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    part = df[df["month"].isin(months)]
    return part[FEATURE_COLS].copy(), part[TARGET].copy()


def make_dataset(
    df: pd.DataFrame,
    train_months: list[str] = TRAIN_MONTHS,
    val_months: list[str] = VAL_MONTHS,
    test_months: list[str] = TEST_MONTHS,
) -> Dataset:
    """Split a feature frame by calendar month. Nothing is shuffled: every train
    row precedes every val row, which precedes every test row."""
    X_train, y_train = _xy(df, train_months)
    X_val, y_val = _xy(df, val_months)
    X_test, y_test = _xy(df, test_months)
    meta_test = df[df["month"].isin(test_months)][META_COLS].copy()
    return Dataset(X_train, y_train, X_val, y_val, X_test, y_test, meta_test)

FEATURE_CACHE = ARTIFACTS_DIR / "features.parquet"


def load_or_build_features(
    months: list[str] = ALL_MONTHS, use_cache: bool = True
) -> pd.DataFrame:
    """Build the full feature frame once and cache it to parquet.

    Feature construction over all 11 months takes a few minutes; every later
    notebook run or training run reads the cached parquet instead. Delete
    ``artifacts/features.parquet`` (or pass ``use_cache=False``) to rebuild.
    """
    if use_cache and FEATURE_CACHE.exists():
        return pd.read_parquet(FEATURE_CACHE)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = build_feature_frame(months)
    df.to_parquet(FEATURE_CACHE, index=False)
    return df
