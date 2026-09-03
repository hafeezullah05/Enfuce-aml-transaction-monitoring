"""Regenerate notebooks/part1.ipynb with the Part 1 structure.

One-off helper so the notebook layout lives in version control as plain text.
Run: uv run python scripts/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NB = Path("notebooks/part1.ipynb")


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

import mlflow

from aml_monitoring.config import (
    ALERT_BUDGET_PCT, ARTIFACTS_DIR, MLFLOW_URI,
    TRAIN_MONTHS, VAL_MONTHS, TEST_MONTHS,
)
from aml_monitoring.dataset import load_or_build_features, make_dataset, FEATURE_COLS
from aml_monitoring.features.transaction import TRANSACTION_FEATURES
from aml_monitoring.features.entity import ENTITY_FEATURES

mlflow.set_tracking_uri(MLFLOW_URI)
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
    # 6 + 7
    md("""
## 6-7. Models, imbalance sweep, and the registered model

**Training is an offline batch job**, not notebook code: `scripts/run_part1.py`
fits the Logistic Regression baseline and LightGBM at four `scale_pos_weight`
values, logs every run to MLflow, and registers the chosen model. This notebook
**loads** the registered model and evaluates it — the same separation you want in
production (experiment vs. analysis).

The sweep below is the evidence behind ADR-0005: the "obvious" fix for imbalance —
set `scale_pos_weight` to the negatives/positives ratio (~1000) — is *catastrophic*
here (PR-AUC 0.008, worse than the linear baseline). Mild weighting (1–5) is best.
"""),
    code("""
pd.read_csv(ARTIFACTS_DIR / "sweep_results.csv")
"""),
    md("""
LogReg baseline PR-AUC is **0.012** — ~15x below LightGBM. The features carry
signal, but the value is in the non-linear interactions and the entity-history
features. We take `scale_pos_weight=1` forward.
"""),
    code("""
model = mlflow.lightgbm.load_model("models:/aml-transaction-monitoring@champion")
model
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
    md("""
### 8d. Explainability — SHAP

Split-count importance tells you *how often* a feature is used, not *how it moves a
decision*. For an AML control we need the second, per alert: every alert an
investigator opens needs a reason, and every SAR filed needs a documented basis.

`aml_monitoring.models.explain` wraps a tree-path-dependent `TreeExplainer` (exact
for GBDTs). `global_importance` is the honest ranking; `explain_alert` is the
reason code for a single transaction.
"""),
    code("""
from aml_monitoring.models.explain import (
    build_explainer, shap_frame, global_importance,
    explain_alert, pick_reason_code_alert,
)

explainer = build_explainer(model)
sv = shap_frame(explainer, ds.X_test.sample(20_000, random_state=0))
global_importance(sv).head(12)
"""),
    code("""
# reason codes for one caught laundering alert (same worked example as the deck)
tp_rows = ds.X_test[ds.y_test.to_numpy() == 1]
tp_row_scores = model.predict_proba(tp_rows)[:, 1]
alert_i = pick_reason_code_alert(model, explainer, tp_rows, tp_row_scores)
explain_alert(model, explainer, tp_rows.iloc[[alert_i]], top_n=8)
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

| | value |
|---|---|
| Test PR-AUC | **0.54** (val 0.68 — see Part 2: the model decays over ~2 months) |
| Test ROC-AUC | 0.98 |
| Operating point (threshold fixed on val at 1% budget, applied to test) | **405 alerts/day · precision 6.4% · recall 77%** (realised alert rate 1.4%) |
| By-budget curve (threshold re-set on test) | 1% budget → 289/day · 8.6% · 74%;  0.1% → 29/day · 57% · 49% |
| Strong typologies | Smurfing 100%, Cash_Withdrawal 97%, Structuring 79% |
| Weak typologies | Fan_Out 60%, Layered_Fan_Out 63% — pure graph-structure patterns, no graph features yet |

**The trade-off, in one sentence:** with the threshold fixed on validation and
applied blind to test, ≈405 alerts/day (≈26 of them real) catches roughly
three-quarters of laundering; tightening the budget to 0.1% (29 alerts/day) raises
precision to 57% but drops recall to 49%. The right point is a capacity +
risk-appetite decision for the investigations team, and it is a threshold move —
the model does not change.

**Known limitations (feed Part 4):**
- No graph features -> weaker on fan-out / bipartite typologies.
- Val -> test PR-AUC drop of 0.14 -> retraining cadence matters (Part 2).
- Labels are synthetic and instantaneous; real SAR outcomes lag by months.
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
