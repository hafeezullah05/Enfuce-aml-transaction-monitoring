"""Check one daily batch against the reference -> one flat metrics record.

Four layers, cheapest first (see docs/part2-lifecycle.md):
  1. data quality   — volume, nulls, unseen categories  (a broken feed looks like
                      model failure; catch it before trusting anything else)
  2. input drift    — PSI per feature vs. the training distribution
  3. prediction drift — score distribution (KS) + alert rate at the fixed threshold
  4. performance    — precision / recall at the budget.  In production this row is
                      empty for weeks (labels lag); here we have labels, so we fill
                      it and treat it as "the signal that arrives late".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aml_monitoring.dataset import CATEGORICAL_COLS, FEATURE_COLS, NUMERIC_FEATURES
from aml_monitoring.monitoring.drift import (
    ks_statistic,
    psi_categorical,
    psi_numeric,
)
from aml_monitoring.monitoring.reference import Reference

PSI_FEATURE_ALARM = 0.2          # significant shift on a single feature
PSI_BROAD_ALARM = (0.1, 3)      # moderate shift on >= 3 features
ALERT_RATE_BAND = (0.007, 0.015)

# Excluded from the PSI alarm (still computed and reported):
#  * *_prior_txn_count — unbounded cumulative counters; PSI vs. a fixed reference
#    rises forever with calendar time regardless of behaviour. Known structural
#    drift. Backlog: cap or window them (a Part 1 fix).
#  * day_of_week — constant within a single daily batch, so PSI is degenerate at
#    this granularity. Monitor it on a weekly rollup instead.
PSI_ALARM_EXCLUDE = {"sender_prior_txn_count", "receiver_prior_txn_count", "day_of_week"}
KNOWN_DRIFT = {"sender_prior_txn_count", "receiver_prior_txn_count"}


def check_batch(
    batch: pd.DataFrame,
    scores: np.ndarray,
    ref: Reference,
    day: str,
) -> dict:
    """Return one metrics record for a single day's transactions."""
    n = len(batch)
    rec: dict[str, object] = {"day": day, "n": n, "model_version": ref.model_version}

    # ---- 1. data quality -------------------------------------------------
    lo = ref.rows_per_day_mean - 3 * ref.rows_per_day_std
    hi = ref.rows_per_day_mean + 3 * ref.rows_per_day_std
    rec["volume_ok"] = bool(lo <= n <= hi)
    rec["max_null_rate"] = float(
        max(batch[c].isna().mean() - ref.null_rate.get(c, 0.0) for c in FEATURE_COLS)
    )
    unseen = 0
    for col in CATEGORICAL_COLS:
        known = set(ref.categorical_ref[col])
        unseen += batch.loc[~batch[col].astype(str).isin(known)].shape[0]
    rec["unseen_category_rows"] = int(unseen)

    # If the feed is broken, the downstream metrics are noise — gate them.
    if not rec["volume_ok"]:
        for k in ("psi_max", "psi_known_drift", "score_mean", "score_p95", "score_ks",
                  "alert_rate", "precision", "recall"):
            rec[k] = np.nan
        rec["psi_max_feature"] = ""
        rec["psi_features_over_0.1"] = 0
        rec["psi_top3"] = ""
        rec["alerts"] = 0
        rec["alarm"] = "data-quality: incomplete batch"
        return rec

    # ---- 2. input drift (PSI) -----------------------------------------
    psi: dict[str, float] = {}
    for col in NUMERIC_FEATURES:
        edges = np.asarray(ref.numeric_edges[col], dtype="float64")
        psi[col] = psi_numeric(pd.Series(ref.numeric_ref[col]), batch[col], edges)
    for col in CATEGORICAL_COLS:
        psi[col] = psi_categorical(ref.categorical_ref[col], batch[col])
    alarm_psi = {k: v for k, v in psi.items() if k not in PSI_ALARM_EXCLUDE}
    rec["psi_max"] = float(max(alarm_psi.values()))
    rec["psi_max_feature"] = max(alarm_psi, key=alarm_psi.get)
    rec["psi_features_over_0.1"] = int(sum(v > 0.1 for v in alarm_psi.values()))
    rec["psi_known_drift"] = float(max(psi[k] for k in KNOWN_DRIFT if k in psi))
    ranked = [k for k in sorted(psi, key=psi.get, reverse=True) if k != "day_of_week"]
    rec["psi_top3"] = "; ".join(f"{k}={psi[k]:.2f}" for k in ranked[:3])

    # ---- 3. prediction drift ----------------------------------------
    rec["score_mean"] = float(np.mean(scores))
    rec["score_p95"] = float(np.quantile(scores, 0.95))
    rec["score_ks"] = ks_statistic(np.asarray(ref.score_ref), scores)
    alert = scores >= ref.threshold
    rec["alert_rate"] = float(alert.mean())
    rec["alerts"] = int(alert.sum())

    # ---- 4. performance (labels lag in production) -----------------
    if "Is_laundering" in batch.columns:
        y = batch["Is_laundering"].to_numpy()
        tp = int((alert & (y == 1)).sum())
        pos = int(y.sum())
        rec["precision"] = tp / alert.sum() if alert.sum() else np.nan
        rec["recall"] = tp / pos if pos else np.nan
        rec["positives"] = pos

    # ---- alarms ------------------------------------------------------
    reasons = []
    if not rec["volume_ok"]:
        reasons.append("volume")
    if rec["unseen_category_rows"] > 0:
        reasons.append("unseen_category")
    if rec["psi_max"] > PSI_FEATURE_ALARM:
        reasons.append(f"psi:{rec['psi_max_feature']}")
    if rec["psi_features_over_0.1"] >= PSI_BROAD_ALARM[1]:
        reasons.append("psi_broad")
    if not (ALERT_RATE_BAND[0] <= rec["alert_rate"] <= ALERT_RATE_BAND[1]):
        reasons.append("alert_rate")
    rec["alarm"] = ", ".join(reasons)

    return rec


def score_batch(model, batch: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(batch[FEATURE_COLS])[:, 1]
