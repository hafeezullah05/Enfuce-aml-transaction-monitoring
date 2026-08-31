"""Regenerate notebooks/main.ipynb with the Part 1 structure.

One-off helper so the notebook layout lives in version control as plain text.
Run: uv run python scripts/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NB = Path("notebooks/main.ipynb")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


cells = [
    md("""
# Part 1 — Model Development & Evaluation

**Goal:** score each transaction for money-laundering risk, and turn the score into
alerts within a fixed daily investigator budget.

Pipeline code lives in `src/aml_monitoring/`; this notebook is the narrative.
"""),
    # 1
    md("## 1. Setup"),
    code("""
%load_ext autoreload
%autoreload 2

import time
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from aml_monitoring.config import TRAIN_MONTHS, VAL_MONTHS, TEST_MONTHS, ALERT_BUDGET_PCT
from aml_monitoring.dataset import load_or_build_features, make_dataset, FEATURE_COLS
from aml_monitoring.features.transaction import TRANSACTION_FEATURES
from aml_monitoring.features.entity import ENTITY_FEATURES
from aml_monitoring.models.train import train_baseline, train_lightgbm

pd.set_option("display.max_columns", None)
"""),
    # 2
    md("""
## 2. Load + build features

`load_or_build_features()` builds all 11 months of features once and caches them to
`artifacts/features.parquet` (a few minutes), then reads the cache on every later run.

Entity features are computed over the **whole** span, so val/test rows carry real
account history — exactly as they would at inference time.
"""),
    code("""
t = time.time()
frame = load_or_build_features()
print(f"{frame.shape} in {time.time() - t:.0f}s")
frame[["timestamp", "month", "Amount", "Is_laundering"]].head()
"""),
    # 3
    md("## 3. Feature signal check"),
    code("""
frame.groupby("Is_laundering")[TRANSACTION_FEATURES + ENTITY_FEATURES].mean().T.round(2)
"""),
    # 4
    md("""
## 4. Temporal split

Split by calendar month — no shuffling. Every training row precedes every validation
row, which precedes every test row. This mirrors production (train on the past,
predict the future) and prevents an account's future behaviour leaking backwards.

- **train**: 2022-10 … 2023-05 (8 months)
- **val**: 2023-06 — threshold selection only
- **test**: 2023-07 … 2023-08 — untouched until the end
"""),
    code("""
ds = make_dataset(frame)
for name, y in [("train", ds.y_train), ("val", ds.y_val), ("test", ds.y_test)]:
    print(f"{name:5s} n={len(y):>9,}  positives={int(y.sum()):>5}  rate={y.mean():.5f}")
print(f"\\nfeatures fed to the model: {len(FEATURE_COLS)}")
"""),
    # 5
    md("""
## 5. Class imbalance

Prevalence is 0.10%. We handle it with **cost-weighting** (`scale_pos_weight`), not
resampling:

- SMOTE interpolates in a categorical + count feature space → fabricated, impossible rows
- The minority is 20+ typologies, not one cluster
- Synthetic rows have no timestamp → break the temporal split
- Rebalancing distorts calibration; we want honest ranking scores

Val/test stay at natural prevalence. The detection-vs-workload trade-off is a
**threshold** decision (Section 8), not a data decision. See ADR-0005.

Section 7 tests `scale_pos_weight` empirically.
"""),
    # 6
    md("""
## 6. Baseline — Logistic Regression

A transparent reference. If a linear model on these features already separates the
classes, the features carry real signal and any LightGBM gain is incremental.
"""),
    code("""
baseline = train_baseline(ds)
"""),
    # 7
    md("""
## 7. LightGBM + the imbalance sweep

We train LightGBM at four `scale_pos_weight` values. The naive choice is
negatives/positives (~1000). The sweep shows what that actually does to PR-AUC.
Each call logs its own MLflow run (`mlflow ui` to compare).
"""),
    code("""
results, models = [], {}
for spw in [1.0, 5.0, 100.0, 1003.0]:
    m = train_lightgbm(ds, scale_pos_weight=spw)
    models[spw] = m
    p = m.predict_proba(ds.X_val)[:, 1]
    results.append({
        "scale_pos_weight": spw,
        "val_pr_auc": average_precision_score(ds.y_val, p),
        "val_roc_auc": roc_auc_score(ds.y_val, p),
    })

comparison = pd.DataFrame(results)
comparison
"""),
    code("""
# The model we take forward.
model = models[1.0]
"""),
    # 8
    md("""
## 8. Evaluation

**Headline metric: PR-AUC (average precision), not ROC-AUC.** At 0.1% prevalence
ROC-AUC is dominated by millions of trivially-classified negatives and stays high
even for a weak model. PR-AUC only rewards precision *on the positives*, which is
what an alerting system lives or dies by.

But no single number captures the real question: **within the alerts investigators
can review per day, how much laundering do we catch?** So the core artefact is the
operating-point table — precision / recall / alerts-per-day across alert budgets.
"""),
    code("""
from aml_monitoring.models.evaluate import (
    score_frame, ranking_metrics, operating_points,
    threshold_for_budget, recall_by_typology, pr_curve,
)

val_scored = score_frame(model, ds.X_val, ds.y_val, ds.meta_val)
test_scored = score_frame(model, ds.X_test, ds.y_test, ds.meta_test)

pd.DataFrame({"val": ranking_metrics(val_scored), "test": ranking_metrics(test_scored)})
"""),
    md("""
### 8a. Operating points on the test set

Each row: a global score cutoff set so that `budget` of transactions are alerted.
`alerts_per_day` is the investigator workload; `recall` is the share of laundering
caught; `precision` is the share of alerts that are real.
"""),
    code("""
operating_points(test_scored).style.format({
    "budget": "{:.3f}", "threshold": "{:.4f}", "precision": "{:.3f}", "recall": "{:.3f}"
})
"""),
    md("""
### 8b. Choosing the operating threshold

The threshold is chosen on the **validation** month at the 1% budget, then applied
unchanged to the held-out test months — so the reported numbers are honest.
"""),
    code("""
thr = threshold_for_budget(val_scored, ALERT_BUDGET_PCT)

alert = test_scored["score"] >= thr
n_alert = int(alert.sum())
n_days = test_scored["date"].nunique()
pos = int(test_scored["label"].sum())
tp = int((alert & (test_scored["label"] == 1)).sum())

print(f"threshold (val, {ALERT_BUDGET_PCT:.0%} budget) : {thr:.4f}")
print(f"test alerts / day                  : {n_alert / n_days:,.0f}")
print(f"test precision                     : {tp / n_alert:.3f}")
print(f"test recall                        : {tp / pos:.3f}")
print(f"laundering caught / missed         : {tp} / {pos - tp}")
"""),
    md("""
### 8c. Where does recall come from — and go missing?

Recall per laundering typology at the chosen threshold. This tells the investigation
team which patterns the model reliably catches (structuring, fan-out) and which slip
through (e.g. single large transfers that look like ordinary large payments).
"""),
    code("""
recall_by_typology(test_scored, thr)
"""),
    md("### 8d. Feature importance"),
    code("""
pd.Series(model.feature_importances_, index=ds.X_test.columns).sort_values(ascending=False).head(15)
"""),
    md("### 8e. Plots — PR curve and recall vs workload"),
    code("""
import numpy as np
import matplotlib.pyplot as plt

op = operating_points(test_scored, budgets=tuple(np.round(np.linspace(0.0005, 0.05, 30), 5)))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(op["recall"], op["precision"], marker=".")
ax[0].set(xlabel="recall", ylabel="precision", title="Precision–Recall (test)")
ax[1].plot(op["alerts_per_day"], op["recall"], marker=".")
ax[1].axvline(n_alert / n_days, ls="--", c="k", lw=1, label="chosen budget")
ax[1].set(xlabel="alerts / day", ylabel="recall", title="Recall vs investigator workload")
ax[1].legend()
for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout()
"""),
    md("""
## 9. Test-set summary

*(fill after running: PR-AUC, the operating point, and one sentence on the
detection-vs-workload trade-off for the deck.)*
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "aml-monitoring (3.11.x)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB.write_text(json.dumps(nb, indent=1))
print(f"wrote {NB} ({len(cells)} cells)")
