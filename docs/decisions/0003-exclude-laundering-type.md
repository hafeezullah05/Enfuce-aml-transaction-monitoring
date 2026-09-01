# ADR-0003: Exclude `Laundering_type` from model features

## Status
Accepted

## Context
The dataset has a `Laundering_type` column. EDA shows its values are disjoint by
class: laundering rows carry typology names (Structuring, Cycle, Fan-Out, …),
non-laundering rows carry only `Normal_*` values. There is zero overlap.

## Decision
`Laundering_type` is never a model input. It is retained only for evaluation —
per-typology recall analysis.

## Consequences
- No target leakage from an annotation that would not exist at scoring time.
- We gain a diagnostic: which laundering patterns the model catches vs. misses
  (Section 8c of the notebook), which informs feature work and Part 4.
