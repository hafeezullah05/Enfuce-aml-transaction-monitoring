# Part 2 — ML Lifecycle & MLOps

The model from Part 1 is deployed. New transaction batches arrive daily. This is
how I would manage the model over time.

Per the brief, I **demonstrate two** capabilities (built, run on the Jul–Aug 2023
batches) and treat the rest conceptually.

| Capability | Status | Code |
|---|---|---|
| Batch monitoring (data quality · input drift · prediction drift · delayed performance) | **built** | `src/aml_monitoring/monitoring/` |
| Retraining trigger + challenger promotion | **built** | `src/aml_monitoring/lifecycle.py` |
| Registry / versioning · champion-challenger · shadow · label lag · feature store · governance · rollback | conceptual | this doc |

End-to-end demo: `scripts/run_part2.py` · narrative + plots: `notebooks/part2.ipynb`

## The problem, concretely

Part 1 already showed the model is not static:

- PR-AUC **0.68 → 0.54** from the validation month to two months later.
- Alert rate drifted **1.0% → 1.4%** at a fixed threshold, because the score
  distribution crept upward.

And the thing we most want to measure — did an alert lead to a confirmed SAR — is
**not available for months**. So the lifecycle has to run on proxies, with a
slower true-performance signal arriving later.

```
 daily batch ─► validate ─► features ─► score ─► alerts ─► investigators
                    │           │          │        │            │
                    └───────────┴────► MONITORING ◄─┘            │
                                          │                      │
                                    trigger fires                │  dispositions
                                          ▼                      │  (weeks)
                                  retrain (rolling window)        │  + SARs (months)
                                          ▼                       │
                                 challenger ─► shadow ─► approve ─► promote
                                          │                        registry
                                          └──────── rollback ◄──────┘
```

## What the demo shows (Jul–Aug 2023, replayed as 54 daily batches)

Run: `uv run python scripts/run_part2.py` → `notebooks/part2.ipynb`

| | July | August |
|---|---|---|
| alert rate (fixed threshold) | 1.21% — in band | **1.67% — breaches the 1.5% band from Aug 1** |
| input-drift PSI (alarm set, max) | 0.14 | 0.13 — genuinely stable |
| precision / recall (the delayed signal) | 7.4% / 80% | 5.5% / 72% |

- **First sustained alarm**: 2023-07-27, then daily through August (`alert_rate`).
- **Retraining trigger fires 2023-08-03** — reasons: `scheduled-floor` + `alert-rate-drift`.
- **Monitoring surfaced a feature-design issue**: `sender/receiver_prior_txn_count`
  are unbounded cumulative counters; their PSI climbs to **1.36** purely because
  calendar time passes. Tagged as known structural drift, excluded from the alarm,
  **backlogged as a Part 1 fix** (cap or window). This is monitoring doing its job.
- **Challenger** (retrained on a rolling Dec 2022 – Jul 2023 window), judged on
  August: PR-AUC **0.69 vs the champion's 0.45**, recall@budget **0.82 vs 0.66**.
  Registered, `@challenger` alias set — awaiting a human promotion to `@champion`.

---

## Demonstrated capability 1 — Monitoring on every daily batch

Four layers, cheapest signal first:

| Layer | Metric | Why it matters | Alarm |
|---|---|---|---|
| **Data quality** | row count vs. 30-day norm, null rates, new categorical values, schema | a broken feed looks like model failure | count outside ±40%, any schema break → **block batch** |
| **Input drift** | PSI / KS per feature vs. the training distribution | features moving = model operating off-distribution | PSI > 0.2 on any feature; > 0.1 on ≥ 3 features |
| **Prediction drift** | score distribution (mean, p95, KS vs. reference), alert rate at the fixed threshold | the model's behaviour changing even if inputs look stable | alert rate outside [0.7%, 1.5%] |
| **Delayed performance** | once dispositions arrive: precision on alerted cases; once SARs arrive: recall on a sampled reservoir | the only true-quality signal | precision drop > 10 pts vs. baseline |

The reference distribution is the **training window**, versioned alongside the
model in the registry so "drift against what" is unambiguous.

Output: one metrics record per batch → CloudWatch / a dashboard, and any alarm →
the retraining trigger.

## Demonstrated capability 2 — Retraining trigger + safe promotion

**Trigger** (retrain if *any* of):

- scheduled: monthly (a floor, so the model never goes stale silently)
- alert rate outside [0.7%, 1.5%] for 3 consecutive days
- input PSI > 0.2 on any feature, or > 0.1 on ≥ 3 features, for 5 days
- delayed-precision drop > 10 points once labels land

**Retrain** on a **rolling window** (last 6–8 months), not all history — the model
should track current behaviour, and old laundering patterns may no longer be
representative.

**Promote safely** — a new model is a *challenger*, never promoted directly:

1. **Shadow**: score the last N daily batches alongside the champion. Compare
   score-distribution stability and, on the most recent labelled data, PR-AUC and
   recall-at-budget.
2. **Gate**: challenger must beat the champion on the labelled slice by a margin,
   and not increase alert-rate volatility.
3. **Approve**: a human (model owner + compliance) signs off — this is a
   regulated control, not a metric race.
4. **Promote**: registry stage transition `Staging → Production`; previous model
   moves to `Archived` but stays loadable for **rollback** (revert stage, re-score).

Every promotion writes an immutable record: data window, code SHA, metrics,
approver, timestamp.

---

## Conceptual — the rest

**Model registry & versioning.** MLflow (already wired in Part 1). Stages
None → Staging → Production → Archived. Each version carries: training data
window reference, git SHA, params, validation + test metrics, and the reference
distribution for drift. The registry is the single control point — scoring always
loads "the Production model", never a path.

**Delayed / biased labels — the hard problem.** Two label sources:
- *Investigator dispositions* — fast (days–weeks) but only for transactions we
  alerted on, so a biased sample. Usable for precision monitoring; risky for
  training (it teaches the model to agree with itself).
- *Confirmed SARs / law-enforcement feedback* — unbiased ground truth, but months
  late.
Mitigation: sample a small **reservoir of non-alerted transactions** for manual
review each period, to get unbiased recall estimates and de-bias training labels.

**Champion/challenger & A/B.** Covered above for promotion. For a larger change
(new feature family) I'd run a true A/B — split accounts, not transactions — for a
period before committing.

**Feature store.** Not needed here: batch-only, no online serving, so an
offline feature table (Parquet / Iceberg on S3) with the training/serving code
*shared as one module* gives train-serve parity without the infrastructure. Add a
real feature store only if a real-time path appears.

**Data / concept drift distinction.** Input drift (PSI) tells us the world
changed; performance drift (delayed) tells us the *relationship* changed. Only the
second strictly requires a retrain; the first is a warning to watch closely.

**Governance & audit.** Model card per version; approval records; immutable run
history; SHAP explanation stored per alert (investigator needs "why", regulator
needs reproducibility). Annual model validation.

**Rollback & incident response.** Registry stage revert + re-score the affected
batches. Runbook for: bad feed, drift alarm, challenger regression,
disposition-pipeline outage.
