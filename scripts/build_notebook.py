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

*(next session)*

- PR-AUC vs ROC-AUC — why ROC-AUC misleads at 0.1% prevalence
- precision / recall / alerts-per-day at the 1% alert budget
- threshold selection with an explicit cost argument
- recall sliced by `Laundering_type`
- feature importance
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
