# ADR-0006: Retraining trigger and gated promotion

## Status
Accepted

## Context
The model decays: PR-AUC 0.68 → 0.54 and alert rate 1.0% → 1.4% over the two
months after the training window (Part 1). New batches arrive daily. The true
quality signal — confirmed SARs — lags by months, so retraining cannot wait for
it and cannot be driven only by it.

## Decision

**Trigger** — retrain if *any* of:
- scheduled monthly floor (the model never goes stale silently);
- alert rate outside [0.7%, 1.5%] for 3 consecutive days;
- input PSI > 0.2 on any feature, or > 0.1 on ≥ 3 features, for 5 days;
- delayed precision drop > 10 points vs. the deployment baseline, once labels land.

**Retrain window** — a rolling 6–8 months, not all history. The model should
track current behaviour; very old patterns may no longer be representative.

**Promotion** — a retrain produces a *challenger*, never promoted automatically:
1. shadow-score recent batches; compare on the most recent labelled slice;
2. gate: challenger must beat the champion on PR-AUC and recall-at-budget and not
   increase alert-rate volatility;
3. a model owner + compliance approve — this is a regulated control;
4. the challenger takes the `@champion` alias; the previous version stays
   registered and loadable.

**Registry mechanics** — MLflow aliases, not version pins. Scoring always loads
`models:/aml-transaction-monitoring@champion`. Promotion and rollback are a single
alias move; no redeploy.

## Rejected alternatives
- **Retrain on a fixed schedule only** — misses fast drift between runs.
- **Retrain only on measured performance drop** — the signal is months late; by
  the time it fires, months of degraded alerting have already happened.
- **Auto-promote the challenger if metrics pass** — removes the human checkpoint a
  regulated AML control requires, and a metric win on a recent slice is not proof
  of no regression elsewhere.
- **Retrain on all history each time** — slower, and dilutes recent behaviour.

## Consequences
- Retraining is proactive (drift) and bounded (schedule), not purely reactive.
- Every promotion leaves an immutable record: data window, code SHA, metrics,
  approver, timestamp.
- The known unbounded-counter drift in `*_prior_txn_count` (see
  `docs/part2-lifecycle.md`) is excluded from the trigger and tracked as a Part 1
  feature-fix backlog item, so it does not cause perpetual retraining.
