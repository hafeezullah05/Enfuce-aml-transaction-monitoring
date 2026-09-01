# Part 1 — Model Development & Evaluation

## Problem

Score each transaction for the probability it is associated with money
laundering, and turn scores into alerts for human investigators who have finite
daily review capacity. The objective is **not** accuracy — it is: *within the
alerts investigators can review per day, how much laundering do we catch, and how
many alerts are wasted?*

## Data

SAML-D synthetic dataset — 9.5M transactions, Oct 2022 – Aug 2023, 11 monthly
files. Prevalence 0.10% (1 in ~960), stable with a slight upward drift
(0.10% → 0.13%). No missing values. Labels are per-transaction: accounts that
launder do so in only ~1.3% of their transactions.

## Preprocessing & feature engineering

**Transaction-level** (single row, no leakage risk): `log1p(amount)`, amount
band flags, `cross_border` (sender vs. receiver bank location), `currency_mismatch`
(payment vs. received currency), hour, day-of-week.

**Entity-level, causal** (per sender and per receiver account) — the signal for
the behavioural typologies: prior transaction count, seconds since previous
transaction, trailing 1d/7d/30d transaction counts, trailing 7d amount
sum/mean, and current amount vs. the account's recent mean (spike detection).

Leakage prevention: the frame is globally time-sorted; rolling windows use
`closed="left"` (window `[t-w, t)`, current row excluded); expanding counts use
`cumcount`; a unit test asserts an account's first transaction sees an empty
history. Entity features are computed over the full timeline *before* the split
so validation/test rows carry real history. `Laundering_type` is excluded
(ADR-0003).

Feature importance is dominated by the entity-history features
(`receiver_secs_since_last`, `sender_prior_txn_count`, `amount_log`,
`*_amount_vs_mean_7d`).

## Class imbalance

Cost-weighting (`scale_pos_weight = 1`), not resampling. A validation sweep shows
the naive "weight by the imbalance ratio (~1000×)" choice collapses PR-AUC to
0.008 — below the linear baseline. See ADR-0005. Validation and test stay at
natural prevalence.

## Split

Temporal, by calendar month: train 2022-10…2023-05, validation 2023-06,
test 2023-07…2023-08. No shuffling. See ADR-0004.

## Models

- **Baseline**: Logistic Regression (scaled numerics + one-hot categoricals,
  `class_weight="balanced"`). Val PR-AUC 0.012.
- **Shipped**: LightGBM, native categoricals, deliberately conservative
  (`num_leaves=15`, `max_depth=4`, `min_child_samples=200`, `lr=0.02`,
  L1/L2 regularisation) because there are only ~5k positives in training — a
  larger tree memorises them in one boosting round.

Training is an offline job (`scripts/run_part1_models.py`): every run logged to
MLflow, the shipped model registered as `aml-transaction-monitoring` v1. The
notebook loads the registered model and never retrains.

## Evaluation

**Headline metric: PR-AUC (average precision).** At 0.10% prevalence ROC-AUC is
dominated by trivially-classified negatives and stays ~0.98 even for weak models;
PR-AUC only rewards precision on the positives, which is what an alerting system
depends on.

| | PR-AUC | ROC-AUC |
|---|---|---|
| Validation (Jun) | 0.68 | 0.99 |
| Test (Jul–Aug) | **0.54** | 0.98 |

**Operating points (test), global threshold at the budget percentile:**

| Alert budget | Alerts/day | Precision | Recall |
|---|---|---|---|
| 0.1% | 29 | 57% | 49% |
| 0.5% | 144 | 16% | 67% |
| **1%** | **289** | **8.6%** | **74%** |
| 2% | 577 | 4.7% | 80% |

**Chosen operating point** — threshold fixed on the validation month at the 1%
budget (score ≥ 0.0079), then applied unchanged to the test months:

| | test |
|---|---|
| alerts / day | 405 |
| precision | 6.4% |
| recall | 76.7% (1390 caught / 423 missed) |

The alert rate lands at ~1.4%, not 1% — the score distribution drifts upward
over the two months, so a fixed threshold slowly widens the net. That gap
(0.68 → 0.54 PR-AUC, 1.0% → 1.4% alert rate in ~2 months) is the concrete case
for the monitoring and retraining work in Part 2.

**The trade-off in one sentence:** reviewing ~400 alerts/day (~22 of them real)
catches ~three-quarters of laundering; tightening the budget to 0.1% (29
alerts/day) raises precision to 57% but drops recall to 49%. The right point is a
capacity + risk-appetite decision for the investigations team — and it is a
threshold move, the model does not change.

**Recall by typology (@ 1% budget)** — strong on volume/velocity patterns
(Smurfing 100%, Cash_Withdrawal 97%, Structuring 79%), weak on pure
graph-structure patterns (Fan_Out 60%, Layered_Fan_Out 63%) because there are no
graph features yet.

## Known limitations (feed Parts 2 & 4)

- No graph features → weaker on fan-out / bipartite typologies.
- Validation → test PR-AUC drop of 0.14, and alert rate creep 1.0% → 1.4%, over
  ~2 months → drift monitoring + retraining cadence matter (Part 2).
- Labels are synthetic and instantaneous; real SAR outcomes lag by months, which
  changes the retraining and evaluation design.
