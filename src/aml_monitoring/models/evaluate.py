"""Evaluation for an AML alerting model.

The headline metric is **PR-AUC** (average precision), not ROC-AUC. At 0.1%
prevalence ROC-AUC is dominated by millions of easy negatives and stays high
even for a weak model. What matters operationally is: within the number of
alerts investigators can review per day, how much laundering do we catch, and
how many of those alerts are wasted?

Workflow:
  1. ``score_frame`` -- attach model scores to labels + metadata.
  2. ``ranking_metrics`` -- PR-AUC / ROC-AUC (threshold-free).
  3. ``operating_points`` -- precision / recall / alerts-per-day across a range
     of alert budgets. Pick the operating threshold on the VALIDATION scores.
  4. ``recall_by_typology`` -- where the recall actually comes from / goes missing.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

DEFAULT_BUDGETS: tuple[float, ...] = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)


def score_frame(model, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> pd.DataFrame:
    """Combine model scores, the true label and metadata into one frame.

    Args:
        model: Anything with ``predict_proba``.
        X: Feature matrix aligned with ``y`` and ``meta`` (same row order).
        y: True ``Is_laundering`` labels.
        meta: Columns ``timestamp``, ``month``, ``Laundering_type``.

    Returns:
        Frame with ``score``, ``label``, ``date`` plus the metadata columns.
    """
    out = meta.reset_index(drop=True).copy()
    out["label"] = pd.Series(y).to_numpy()
    out["score"] = model.predict_proba(X)[:, 1]
    out["date"] = pd.to_datetime(out["timestamp"]).dt.date
    return out


def ranking_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Threshold-free metrics."""
    return {
        "pr_auc": float(average_precision_score(df["label"], df["score"])),
        "roc_auc": float(roc_auc_score(df["label"], df["score"])),
        "prevalence": float(df["label"].mean()),
        "positives": int(df["label"].sum()),
    }


def threshold_for_budget(df: pd.DataFrame, budget: float) -> float:
    """The global score cutoff that flags ``budget`` fraction of transactions."""
    return float(df["score"].quantile(1.0 - budget))


def operating_points(
    df: pd.DataFrame, budgets: tuple[float, ...] = DEFAULT_BUDGETS
) -> pd.DataFrame:
    """Precision / recall / workload at each alert budget.

    A single global threshold is set at the budget percentile. Workload is
    reported per day because investigator capacity is a daily constraint.
    """
    n_days = df["date"].nunique()
    total_pos = int(df["label"].sum())
    rows = []
    for b in budgets:
        thr = threshold_for_budget(df, b)
        alert = df["score"] >= thr
        n_alert = int(alert.sum())
        tp = int((alert & (df["label"] == 1)).sum())
        rows.append(
            {
                "budget": b,
                "threshold": thr,
                "alerts_per_day": round(n_alert / n_days, 1),
                "precision": tp / n_alert if n_alert else 0.0,
                "recall": tp / total_pos if total_pos else 0.0,
                "caught": tp,
                "missed": total_pos - tp,
            }
        )
    return pd.DataFrame(rows)


def recall_by_typology(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Per-typology recall at a chosen operating threshold (laundering rows only)."""
    pos = df[df["label"] == 1].copy()
    pos["caught"] = pos["score"] >= threshold
    return (
        pos.groupby("Laundering_type", observed=True)["caught"]
        .agg(n="count", caught="sum")
        .assign(recall=lambda d: (d["caught"] / d["n"]).round(3))
        .sort_values("recall")
    )


def pr_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Precision / recall / threshold points for plotting the PR curve."""
    precision, recall, thr = precision_recall_curve(df["label"], df["score"])
    return pd.DataFrame(
        {"precision": precision[:-1], "recall": recall[:-1], "threshold": thr}
    )
