# ADR-0004: Split train / validation / test by calendar month

## Status
Accepted

## Context
Transactions arrive over time (Oct 2022 – Aug 2023, 11 monthly files). The
production task is to score *future* transactions given a model trained on the
*past*. Accounts recur across months, and entity-history features summarise an
account's recent behaviour.

## Decision
Split strictly by time, no shuffling:
- **train**: 2022-10 … 2023-05 (8 months)
- **validation**: 2023-06 (threshold selection only)
- **test**: 2023-07 … 2023-08 (held out until the end; reused as "production
  batches" in Part 2)

Entity features are computed over the full timeline before splitting, so
validation and test rows carry real account history — as they would at inference.

## Rejected: random / stratified k-fold split
- Leaks future behaviour of an account into its own past rows.
- Produces an over-optimistic score and hides exactly the temporal decay we
  care about operating (val PR-AUC 0.68 vs. test 0.54).

## Consequences
- Metrics reflect real forward performance.
- Fewer positives per split; the model config is deliberately conservative.
- The val→test drop is a feature, not a bug — it quantifies retraining need.
