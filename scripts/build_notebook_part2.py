"""Regenerate notebooks/part2.ipynb (Part 2 narrative + plots).

Reads artifacts/monitoring_history.csv (produced by scripts/run_part2.py), so it
is fast — it does not replay the batches.

Run: uv run python scripts/build_notebook_part2.py
"""

from __future__ import annotations

import json
from pathlib import Path

NB = Path("notebooks/part2.ipynb")


def md(t: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)}


def code(t: str) -> dict:
    return {
        "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
        "source": t.strip("\n").splitlines(keepends=True),
    }


cells = [
    md("""
# Part 2 — ML Lifecycle & MLOps

The Part 1 model (`aml-transaction-monitoring@champion`, trained on Oct 2022 – May
2023) is **deployed**. We replay **Jul–Aug 2023 as daily production batches** and
show two lifecycle capabilities running:

1. **Monitoring** every batch — data quality, input drift (PSI), prediction drift, and
   the delayed performance signal.
2. A **retraining trigger** + challenger evaluation + gated promotion.

Design for the rest (registry, champion/challenger, shadow, label lag, governance):
`docs/part2-lifecycle.md`.
"""),
    md("## 1. Setup"),
    code("""
%load_ext autoreload
%autoreload 2

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from aml_monitoring.config import ARTIFACTS_DIR, MLFLOW_URI
from aml_monitoring.monitoring.batch import ALERT_RATE_BAND
from aml_monitoring.lifecycle import should_retrain

mlflow.set_tracking_uri(MLFLOW_URI)
pd.set_option("display.max_columns", None)

history = pd.read_csv(ARTIFACTS_DIR / "monitoring_history.csv", parse_dates=["day"])
ref = json.loads((ARTIFACTS_DIR / "reference.json").read_text())
print(f"{len(history)} daily batches   threshold={ref['threshold']:.4f}   "
      f"expected {ref['rows_per_day_mean']:.0f} rows/day")
history.head()
"""),
    md("""
## 2. The reference snapshot

Monitoring compares each batch to a snapshot of the world **as it looked when the
model went live** — the training feature distributions and the validation-month
score distribution — versioned with the model. "Drift against what" is then
unambiguous. Built by `aml_monitoring.monitoring.reference.build_reference`.
"""),
    md("## 3. Monitoring dashboard"),
    code("""
fig, ax = plt.subplots(2, 2, figsize=(13, 7))
d = history["day"]

ax[0, 0].plot(d, history["alert_rate"], marker=".", color="#00A896")
ax[0, 0].axhspan(*ALERT_RATE_BAND, color="#00A896", alpha=0.08)
ax[0, 0].axhline(0.01, ls="--", c="grey", lw=1)
ax[0, 0].set_title("Alert rate vs. the 0.7–1.5% band")

ax[0, 1].plot(d, history["score_mean"], marker=".", label="mean")
ax[0, 1].plot(d, history["score_p95"], marker=".", label="p95")
ax[0, 1].set_title("Score distribution drift"); ax[0, 1].legend()

ax[1, 0].plot(d, history["psi_max"], marker=".", label="alarm set (max)")
ax[1, 0].plot(d, history["psi_known_drift"], marker=".", label="prior_txn_count (known)")
ax[1, 0].axhline(0.2, ls="--", c="red", lw=1)
ax[1, 0].set_title("Input drift — PSI"); ax[1, 0].legend()

ax[1, 1].plot(d, history["precision"], marker=".", label="precision")
ax[1, 1].plot(d, history["recall"], marker=".", label="recall")
ax[1, 1].set_title("Delayed performance (arrives weeks later)"); ax[1, 1].legend()

for a in ax.flat:
    a.grid(alpha=0.3); a.tick_params(axis="x", rotation=45)
plt.tight_layout()
"""),
    code("""
# monthly summary
history["month"] = history["day"].dt.strftime("%Y-%m")
history.groupby("month")[["alert_rate", "score_mean", "psi_max", "psi_known_drift",
                          "precision", "recall"]].mean().round(4)
"""),
    md("""
### What the monitor shows

- **July**: alert rate ~1.2%, inside the band. Model healthy.
- **August**: alert rate drifts to ~1.7%, breaching the band — the fixed threshold
  no longer delivers the 1% budget because the score distribution crept up.
- **Input drift (alarm set)**: essentially flat — no genuine feature drift.
- **Known structural drift**: `*_prior_txn_count` PSI climbs past 1.0. These are
  unbounded cumulative counters — they drift with calendar time by construction.
  Reported, excluded from the alarm, **backlogged as a Part 1 feature fix** (cap or
  window them). This is monitoring doing its job — it surfaced a design issue.
"""),
    md("## 4. The retraining trigger"),
    code("""
baseline_precision = history["precision"].dropna().iloc[:5].mean()
decision = should_retrain(history, baseline_precision=baseline_precision, days_since_training=60)
print("retrain:", decision.retrain)
print("reasons:", decision.reasons)
print("first triggered:", decision.trigger_day)
"""),
    md("""
Policy (`aml_monitoring.lifecycle.should_retrain`): retrain if **any** of —
scheduled monthly floor · alert-rate outside band for 3 consecutive days · input
drift for 5 days · delayed-precision drop > 10 pts. Here: the scheduled floor and
the August alert-rate drift both fire.
"""),
    md("## 5. Challenger — retrain on a rolling window"),
    code("""
from aml_monitoring.config import ALL_MONTHS
from aml_monitoring.dataset import load_or_build_features, make_dataset
from aml_monitoring.lifecycle import rolling_window, fit_challenger, compare_on_labelled

frame = load_or_build_features()
champion = mlflow.lightgbm.load_model("models:/aml-transaction-monitoring@champion")

months = rolling_window(ALL_MONTHS, through="2023-07", width=8)
print("challenger window:", months[0], "..", months[-1])
ch_ds = make_dataset(frame, train_months=months, val_months=["2023-08"], test_months=["2023-08"])
challenger = fit_challenger(ch_ds)
"""),
    code("""
labelled_aug = frame[frame["month"] == "2023-08"]
cmp = compare_on_labelled(champion, challenger, labelled_aug)
pd.DataFrame({
    "champion":   [cmp["champion_pr_auc"], cmp["champion_recall_at_budget"]],
    "challenger": [cmp["challenger_pr_auc"], cmp["challenger_recall_at_budget"]],
}, index=["PR-AUC (Aug)", "recall @ 1% budget (Aug)"]).round(3)
"""),
    md("""
## 6. Promotion

The challenger **is not promoted automatically**. It beats the champion on the
most recent labelled slice, so `scripts/run_part2.py` registers the new version
and sets the `@challenger` alias — awaiting a human (model owner + compliance)
sign-off before it takes the `@champion` alias. The current champion version stays
registered and loadable, so rollback is one alias move.

```
mlflow models  →  aml-transaction-monitoring
   v1   @champion    (in production, kept for rollback)
   v6   @challenger  (retrained through Jul-2023, awaiting approval)
```

Scoring code only ever asks for `@champion` — it never pins a version — so a
promotion or a rollback is a registry operation, not a deploy.
"""),
    md("""
## 7. What this demonstrates

| Capability | Shown |
|---|---|
| Data-quality / drift / performance monitoring on every batch | §3 |
| Input drift surfacing a real feature-design issue | §3 |
| A codified retraining trigger | §4 |
| Rolling-window retrain + champion/challenger comparison | §5 |
| Gated, human-approved promotion with rollback | §6 |

**Conceptual** (design in `docs/part2-lifecycle.md`): registry mechanics,
shadow deployment, the delayed/biased-label problem and reservoir sampling,
feature-store parity, governance & audit, incident response.
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "aml-monitoring (3.11.x)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
NB.write_text(json.dumps(nb, indent=1))
print(f"wrote {NB} ({len(cells)} cells)")
