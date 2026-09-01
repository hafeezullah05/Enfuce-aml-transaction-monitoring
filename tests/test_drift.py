"""Sanity checks for the drift primitives."""

import numpy as np
import pandas as pd

from aml_monitoring.monitoring.drift import (
    ks_statistic,
    numeric_bins,
    psi_categorical,
    psi_numeric,
    psi_verdict,
)


def test_psi_numeric_zero_when_identical() -> None:
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(size=20_000))
    edges = numeric_bins(ref)
    same = pd.Series(rng.normal(size=20_000))
    assert psi_numeric(ref, same, edges) < 0.05


def test_psi_numeric_flags_a_shift() -> None:
    rng = np.random.default_rng(1)
    ref = pd.Series(rng.normal(size=20_000))
    edges = numeric_bins(ref)
    shifted = pd.Series(rng.normal(loc=1.0, size=20_000))
    assert psi_numeric(ref, shifted, edges) > 0.2
    assert psi_verdict(psi_numeric(ref, shifted, edges)) == "significant"


def test_psi_categorical_detects_new_category() -> None:
    ref = {"A": 0.5, "B": 0.3, "C": 0.2}
    same = pd.Series(["A"] * 50 + ["B"] * 30 + ["C"] * 20)
    assert psi_categorical(ref, same) < 0.05
    with_new = pd.Series(["A"] * 40 + ["B"] * 20 + ["Z"] * 40)
    assert psi_categorical(ref, with_new) > 0.2


def test_ks_statistic_range() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=5_000)
    assert ks_statistic(a, a) == 0.0
    assert ks_statistic(a, rng.normal(loc=2, size=5_000)) > 0.5
