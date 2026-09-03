"""Retraining trigger and safe promotion.

Trigger policy (docs/part2-lifecycle.md / ADR): retrain if ANY of
  * scheduled floor    — it has been a full calendar month since training
  * alert-rate drift   — alert rate outside [0.7%, 1.5%] for 3 consecutive days
  * input drift        — psi_max > 0.2, or >= 3 features over 0.1, for 5 days
  * performance drop    — precision down > 10 pts vs. baseline (once labels arrive)

Promotion: a retrain produces a *challenger*. It is registered but not promoted
until it beats the champion on the most recent labelled slice. A human approves
the final promotion. The previous version stays loadable for rollback.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import average_precision_score

from aml_monitoring.dataset import FEATURE_COLS, TARGET, Dataset
from aml_monitoring.models.train import fit_lightgbm

ALERT_RATE_BAND = (0.007, 0.015)
ALERT_RATE_RUN = 3
PSI_RUN = 5
PRECISION_DROP = 0.10


@dataclass
class RetrainDecision:
    retrain: bool
    reasons: list[str]
    trigger_day: str | None


def _consecutive(mask: pd.Series, run: int) -> str | None:
    """First day that completes a run of `run` consecutive True values."""
    streak = 0
    for day, flag in mask.items():
        streak = streak + 1 if flag else 0
        if streak >= run:
            return str(day)
    return None


def should_retrain(
    history: pd.DataFrame,
    baseline_precision: float | None = None,
    days_since_training: int = 0,
) -> RetrainDecision:
    """Evaluate the trigger policy against accumulated monitoring history.

    Args:
        history: one row per day from ``check_batch`` (indexed or with a ``day`` column).
        baseline_precision: reference precision at deployment; enables the
            performance-drop check once batch precision is available.
        days_since_training: calendar days since the current model was trained.
    """
    h = history.set_index("day") if "day" in history.columns else history
    reasons: list[str] = []
    trigger_day: str | None = None

    if days_since_training >= 30:
        reasons.append("scheduled-floor")

    ar = h["alert_rate"]
    ar_bad = (ar < ALERT_RATE_BAND[0]) | (ar > ALERT_RATE_BAND[1])
    d = _consecutive(ar_bad, ALERT_RATE_RUN)
    if d:
        reasons.append("alert-rate-drift")
        trigger_day = min(trigger_day, d) if trigger_day else d

    psi_bad = (h["psi_max"] > 0.2) | (h["psi_features_over_0.1"] >= 3)
    d = _consecutive(psi_bad, PSI_RUN)
    if d:
        reasons.append("input-drift")
        trigger_day = min(trigger_day, d) if trigger_day else d

    if baseline_precision is not None and "precision" in h.columns:
        recent = h["precision"].dropna().tail(7).mean()
        if pd.notna(recent) and (baseline_precision - recent) > PRECISION_DROP:
            reasons.append("performance-drop")

    return RetrainDecision(bool(reasons), reasons, trigger_day)


# ---------------------------------------------------------------- retrain --


def rolling_window(all_months: list[str], through: str, width: int = 8) -> list[str]:
    """The training months for a retrain that includes data 'through' a month."""
    idx = all_months.index(through)
    return all_months[max(0, idx - width + 1) : idx + 1]


def fit_challenger(ds: Dataset, progress: bool = True) -> object:
    """Fit a challenger on a (rolling-window) dataset. Same recipe as the champion."""
    return fit_lightgbm(ds, scale_pos_weight=1.0, progress=progress)


def compare_on_labelled(
    champion, challenger, labelled: pd.DataFrame, budget: float = 0.01
) -> dict:
    """Champion vs. challenger on a recent labelled slice: PR-AUC + recall@budget."""
    y = labelled[TARGET].to_numpy()
    X = labelled[FEATURE_COLS]
    out = {}
    for name, m in (("champion", champion), ("challenger", challenger)):
        s = m.predict_proba(X)[:, 1]
        thr = pd.Series(s).quantile(1 - budget)
        alert = s >= thr
        tp = int((alert & (y == 1)).sum())
        out[f"{name}_pr_auc"] = float(average_precision_score(y, s))
        out[f"{name}_recall_at_budget"] = tp / int(y.sum()) if y.sum() else float("nan")
    out["challenger_wins"] = (
        out["challenger_pr_auc"] > out["champion_pr_auc"]
        and out["challenger_recall_at_budget"] >= out["champion_recall_at_budget"] - 0.01
    )
    return out
