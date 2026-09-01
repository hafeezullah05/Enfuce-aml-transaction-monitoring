"""Distribution-drift primitives: PSI for every feature, KS for the score.

PSI (Population Stability Index) is the retail-banking / credit-risk standard and
the one an AML reviewer will expect:

    PSI = sum_i (a_i - e_i) * ln(a_i / e_i)

over bins i, where e_i / a_i are the expected (reference) and actual (batch)
proportions in bin i. Rule of thumb: < 0.1 stable, 0.1-0.2 moderate shift,
> 0.2 significant shift.

KS (two-sample Kolmogorov-Smirnov) is used only for the model score - a single
continuous distribution where the max CDF gap is the natural summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Floor for empty bins. Too small (1e-6) makes PSI explode when a feature moves
# entirely out of the reference range; 1e-4 is the common convention and keeps
# the number interpretable while still flagging the shift.
_EPS = 1e-4


def numeric_bins(reference: pd.Series, n_bins: int = 10) -> np.ndarray:
    """Quantile bin edges from the reference sample (open at both ends)."""
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.nanquantile(reference.to_numpy(dtype="float64"), qs))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _proportions(x: pd.Series, edges: np.ndarray) -> np.ndarray:
    counts = np.histogram(x.dropna().to_numpy(dtype="float64"), bins=edges)[0]
    total = counts.sum()
    if total == 0:
        return np.full(len(counts), _EPS)
    return np.clip(counts / total, _EPS, None)


def psi_numeric(expected: pd.Series, actual: pd.Series, edges: np.ndarray) -> float:
    """PSI for a numeric feature, using pre-computed reference bin edges."""
    e = _proportions(expected, edges)
    a = _proportions(actual, edges)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(expected_props: dict[str, float], actual: pd.Series) -> float:
    """PSI for a categorical feature.

    Args:
        expected_props: {category: reference proportion} from the Reference snapshot.
        actual: the batch's values for this feature.

    Categories unseen in the reference collapse into one 'other' bucket — their
    appearance is itself a drift signal.
    """
    known = list(expected_props)
    a_counts = actual.astype(str).value_counts(normalize=True)
    a_known = a_counts.reindex(known).fillna(0.0)
    a_other = float(a_counts.drop(index=known, errors="ignore").sum())

    e = np.clip(np.append(np.array(list(expected_props.values())), _EPS), _EPS, None)
    a = np.clip(np.append(a_known.to_numpy(), max(a_other, _EPS)), _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def ks_statistic(expected: np.ndarray, actual: np.ndarray) -> float:
    """Two-sample KS statistic (max gap between the two empirical CDFs)."""
    e = np.sort(np.asarray(expected, dtype="float64"))
    a = np.sort(np.asarray(actual, dtype="float64"))
    grid = np.concatenate([e, a])
    cdf_e = np.searchsorted(e, grid, side="right") / e.size
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    return float(np.max(np.abs(cdf_e - cdf_a)))


def psi_verdict(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.2:
        return "moderate"
    return "significant"
