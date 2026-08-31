# ADR-0005: Handle class imbalance with cost-weighting, not resampling

## Status
Accepted

## Context
Prevalence of `Is_laundering` is ~0.10%, stable across the 11 months. The feature
space is mostly categorical (payment type, currencies, locations) and count-based
(trailing-window transaction counts). The minority class spans 20+ typologies.
The split is temporal.

## Decision
Train LightGBM with `scale_pos_weight = 1` (loss left ~unweighted). Keep
validation and test at natural prevalence. Handle the detection-vs-workload
trade-off at threshold selection (the alert budget), not in the training data.

A `scale_pos_weight` sweep on the validation month is the evidence:

| scale_pos_weight | val PR-AUC | val ROC-AUC |
|---|---|---|
| 1 | 0.682 | 0.990 |
| 5 | 0.685 | 0.991 |
| 100 | 0.585 | 0.989 |
| 1003 (≈ negatives/positives, the "obvious" default) | **0.008** | 0.861 |

Heavy positive-weighting collapses PR-AUC below the linear baseline (0.012): it
floods the top of the ranking with borderline negatives and the loss surface
degenerates (early stopping fires at iteration 2). ROC-AUC barely moves, which is
why ROC-AUC is the wrong headline metric here.

## Rejected: SMOTE / synthetic oversampling
- Interpolates in a categorical + count space → fabricated, impossible rows
  (fractional transaction counts, blended payment types).
- The minority is many typologies, not one cluster; interpolating between them
  invents hybrids that overlap the normal class.
- Synthetic points have no timestamp → violate the temporal split.
- Rebalancing distorts probability calibration; alert ranking needs honest scores.

## Consequences
- Every metric reflects production prevalence directly.
- Imbalance becomes a threshold decision, revisited per the alert budget.
- If a specific typology is genuinely under-represented in future, address it
  with targeted data collection, not synthetic interpolation.
