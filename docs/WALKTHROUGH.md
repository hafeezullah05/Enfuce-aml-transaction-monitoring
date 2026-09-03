# AML Transaction Monitoring — the full walkthrough

A plain-English narrative of the whole solution: what I built, every choice I
made, the alternative I considered, why I went the way I did, and the trade-off.
Written to be read out loud in preparation, not shown to anyone.

Contents:
1. The framing (read this first — it explains everything else)
2. The data
3. Part 1 — preprocessing
4. Part 1 — feature engineering
5. Part 1 — class imbalance
6. Part 1 — the split
7. Part 1 — the model
8. Part 1 — evaluation and the metric
9. Part 1 — the detection-vs-workload trade-off
10. Part 1 — error analysis and honest limits
11. Part 2 — the lifecycle loop
12. Part 2 — monitoring
13. Part 2 — the retraining trigger and promotion
14. Part 3 — production architecture
15. Part 4 — delivery
16. The code, as a flow

---

## 1. The framing

**The one idea everything else follows from:** a transaction-monitoring model
does not make an accept-or-decline decision. It produces a *ranked queue of work*
for human investigators, and those investigators have a fixed amount of time per
day. So the question is never "how accurate is the model" — it is *"within the
number of alerts a team can actually review in a day, how much laundering do we
catch, and how many of those alerts are wasted effort?"*

**Why this matters for the decisions:** once you accept that framing, a lot
falls out of it automatically.

- I optimise **ranking quality**, and I choose the **operating threshold
  separately**. I never use a 0.5 cut-off — at one-in-a-thousand prevalence a 0.5
  cut-off is meaningless.
- The headline metric has to be one that rewards precision *on the positives*,
  not overall accuracy — that points to PR-AUC, not ROC-AUC.
- Class imbalance stops being a "fix the training data" problem and becomes a
  "where do I put the threshold" problem.
- "Balance detection against unnecessary alerts" — which the brief calls out
  specifically — is literally a curve, and picking the point on it is the
  investigations team's decision, not mine.

**The alternative framing** would be to treat it as a straightforward binary
classifier and optimise F1 or accuracy. Why that's worse: it optimises for a
decision boundary the system never uses, it's dominated by the 99.9% easy
negatives, and it gives the investigation team nothing to reason about when they
decide how many analysts to staff.

**The other thing I want them to see:** I built it the way I'd build it for
production, not as a notebook. The pipeline is an installable package with unit
tests, every significant decision is a one-page ADR in the repo, and every model
is tracked in MLflow and promoted through a registry. That's not polish for the
write-up — it's the substrate.

---

## 2. The data

SAML-D — a synthetic AML dataset. **9.5 million transactions, 11 months** (Oct
2022 to Aug 2023), arriving as one file per month.

Key facts and what each one changed:

- **Prevalence is 0.10%** — about 1 in 960 — and roughly stable, with a slight
  upward drift to 0.13% over the period. *That drift is itself a monitoring
  signal; I come back to it in Part 2.*
- **The label is per-transaction, not per-account.** An account that launders
  looks completely ordinary in about 99% of its transactions — I measured it:
  among accounts that ever launder, only 1.3% of their transactions are flagged.
  *So I cannot build an account classifier. I have to score each transaction
  using what's known about the account at that moment.*
- **No missing values.** Clean synthetic data. *So there's genuinely no
  imputation problem — which I'll say explicitly rather than pretend I did
  clever imputation.*
- **293k sender accounts, 90% of them recurring.** *That's what makes
  account-history features viable — if accounts didn't repeat, there'd be no
  history to summarise.*
- **The files are already monthly.** *That's my natural unit for the temporal
  split, and later for the "daily batches" in Part 2.*
- **28 named laundering typologies** (Structuring, Fan-Out, Cycle,
  Gather-Scatter, Smurfing, …). *I use these for error analysis — to see which
  patterns the model catches — but never as a feature, because the typology
  labels are disjoint between the classes, so they're derived from the target.*

---

## 3. Part 1 — preprocessing

**What I did:** almost nothing, on purpose.

On load: enforce an explicit typed schema (half the memory, and the schema
intent is visible in the code), build one `timestamp` from the separate date and
time columns, and sort the whole dataset by time once, globally.

There are no missing values, so no imputation — except in the entity features,
where an account's very first transaction has no history. That NaN is
*meaningful* — it means "no history", not "missing value". LightGBM handles it
natively; for the linear baseline I fill it with the median.

**The choice that's worth explaining:** preprocessing **differs by model**.

- For **LightGBM** I keep the categorical columns as a native `category` dtype —
  **no one-hot encoding** — because the tree splits on categories directly. And
  **no scaling**, because trees are scale-invariant. The only transform I keep is
  `log1p(amount)`, purely so the feature-importance numbers are readable.
- For the **Logistic Regression baseline** I do the full standard treatment:
  median-impute, standardise, one-hot. Linear models require it.

**Alternatives I rejected:**

- **Target/mean encoding of categoricals** — it's a common trick, but it adds a
  leakage surface (you're injecting label information into a feature) and it
  makes the model harder to reason about, for no measurable gain here since the
  categoricals are low-cardinality (7 to 18 values).
- **PCA / dimensionality reduction** — 26 features is not a dimensionality
  problem, and PCA would destroy the interpretability that an investigator and a
  regulator both need.

**The trade-off:** minimal preprocessing means I'm leaning on the model to do
the work. That's fine for a tree; it's exactly why the linear baseline is so
much weaker (I'll get to the numbers). If I had to ship the linear model, I'd
invest much more here.

---

## 4. Part 1 — feature engineering

**Two groups.**

**Transaction-level** — derived from a single row, so zero leakage risk:
`log1p(amount)`, small/large amount flags, `cross_border` (sender and receiver
bank locations differ), `currency_mismatch` (payment vs received currency),
`hour`, `day_of_week`.

These carry real but modest signal. From the EDA: cross-border transactions are
**25% of laundering vs 10% of normal**; currency mismatch is **34% vs 11%**;
large amounts (>£100k) are **3.2% vs 0.3%**.

**Entity-level, causal** — per sender account *and* per receiver account:

- `prior_txn_count` — how many transactions this account has had before
- `secs_since_last` — seconds since its previous transaction (velocity)
- `cnt_1d`, `cnt_7d`, `cnt_30d` — transaction counts over trailing windows
- `sum_7d`, `mean_7d` — trailing 7-day amount total and average
- `amount_vs_mean_7d` — this transaction's amount over the account's recent
  average. **This is the single strongest feature.** A transaction that's 30x an
  account's normal size is a strong flag.

**Why these windows:** I chose them for *meaning*, not by tuning. One day catches
a burst; seven days captures the account's rhythm; thirty days is its baseline.

**Why entity features are the point:** the EDA showed the laundering typologies
are behavioural — structuring, smurfing, fan-out are all about *how an account
transacts*, not about any single transaction looking weird. And feature
importance confirms it: the top features are all entity-history features
(`sender_prior_txn_count`, `receiver_prior_txn_count`, the `secs_since_last`
pair, then `amount_log`).

**Leakage prevention — the part I'd most want them to probe.** Every windowed
feature is strictly backward-looking:

1. The frame is sorted globally by timestamp before any feature is computed.
2. Rolling windows use a **left-closed interval** `[t - w, t)` — the current
   transaction and everything after it are excluded.
3. Expanding counts use `cumcount`, which counts only prior rows.
4. There is a **unit test** (`tests/test_entity_features.py`) that builds a tiny
   3-transaction dataset and asserts that an account's first transaction sees an
   empty history — count 0, windows 0, `secs_since_last` NaN. If someone breaks
   the causality later, that test fails.

**What I excluded, and why:**

- **Account IDs as features** — they're identifiers, not behaviour. Letting the
  model memorise specific accounts is both a subtle leak (train/test share
  accounts) and useless on genuinely new accounts.
- **`Laundering_type`** — target-derived (disjoint values by class). Using it
  would be textbook leakage.

**The honest gap:** I have **no graph features** — sender fan-out degree,
receiver fan-in degree, connected-component size. That's why the model is weak on
the pure graph-structure typologies (fan-out, bipartite — I'll show the numbers).
It's the top item on my backlog. I didn't do it because behavioural aggregates
were enough to demonstrate the approach, and graph features need a bit more
plumbing (you're building a transaction graph and computing windowed degree).

---

## 5. Part 1 — class imbalance

**This is the slide I most want to land, because the obvious answer is wrong.**

Prevalence is 0.10%. The instinct is to reweight the loss by the imbalance ratio
— set LightGBM's `scale_pos_weight` to roughly 1000 (negatives ÷ positives).

**I tested it.** A sweep on the validation month:

| scale_pos_weight | validation PR-AUC |
|---|---|
| 1 (what I shipped) | **0.68** |
| 5 | 0.68 |
| 100 | 0.59 |
| ~1000 (the "obvious" default) | **0.008** |
| (Logistic Regression baseline, class_weight="balanced") | 0.012 |

The naive choice **collapses PR-AUC by a factor of about 80** — to *below* the
linear baseline. Meanwhile ROC-AUC barely moves (stays ~0.86–0.99 across the
whole sweep).

**Why:** the model already ranks well. Heavy positive-weighting doesn't improve
the ranking — it just shoves borderline negatives to the top of the alert queue,
and it makes the loss surface degenerate (early stopping fires at iteration 2).
And the fact that ROC-AUC doesn't notice is exactly why ROC-AUC is the wrong
metric to steer by.

**The conclusion:** *imbalance is a threshold problem, not a training-data
problem.* I leave the loss essentially unweighted and handle the
detection-vs-workload trade-off at the operating threshold.

**Alternatives I considered:**

- **Random undersampling of the majority** — legitimate, and I'd keep it as a
  faster-training option. It throws away data but on 7M rows that's fine. I
  didn't need it.
- **SMOTE / synthetic oversampling** — rejected on principle, not just
  empirically:
  - It interpolates between minority points. My feature space is mostly
    categorical and count-based. "Halfway between Cash Deposit and Cross-border"
    or "3.5 transactions in a 7-day window" are rows that cannot occur.
  - The minority isn't one blob — it's 20+ typologies. Interpolating between a
    structuring transaction and a fan-out transaction invents a hybrid that
    corresponds to no real laundering and can land on top of the normal class.
  - Synthetic points have no timestamp, so they have no place in a time-ordered
    split.
  - It distorts calibration — after SMOTE the scores reflect a ~balanced
    prevalence, not the real 0.1%, and I want scores that rank honestly on the
    true distribution.

**The trade-off:** by keeping the natural distribution everywhere, every metric I
report is on real prevalence — nothing is inflated by resampling. The cost is
that the raw predicted probabilities are all small; but I never threshold at
0.5, I rank and take the top 1%, so absolute calibration doesn't matter, only
the ordering.

---

## 6. Part 1 — the split

**What I did:** split strictly by calendar month, no shuffling.

- **Train**: Oct 2022 – May 2023 (8 months)
- **Validation**: June 2023 — used *only* to choose the operating threshold
- **Test**: July – August 2023 — touched exactly once, at the very end (and
  reused as the "daily batches" in Part 2)

Encoders and the entity-feature reference are fit on the training window only.
The entity features themselves are computed over the *full* timeline before
splitting, so a July test row carries the account's real history back to
October — exactly what the model would have at inference time.

**Why not a random split:** accounts recur across months. A random split would
put an account's August transactions in the training set and its October
transactions in the test set — the model would "predict" October using
information from August. That's leakage. It inflates the score, and worse, **it
hides the one thing I most need to see: how the model decays over time.** With
the temporal split I can measure that decay (0.68 → 0.54 PR-AUC), and that
measurement is what motivates all of Part 2.

**The alternative I considered properly — account-grouped k-fold:** keep all of
an account's rows in the same fold. That removes the account-identity leakage.
But it still mixes time periods — training folds contain data from after the
test fold — so it still gives an over-optimistic picture of *forward*
performance. For a system that scores tomorrow's transactions, time has to be
the split axis.

**The trade-off:** a temporal split gives me fewer positives per split (about 7k
in training), and a single fixed test period rather than a cross-validated
estimate. That's why the model config is deliberately conservative (next
section), and why I'd want more test periods before trusting a small difference
between two models in production.

---

## 7. Part 1 — the model

**Baseline: Logistic Regression** (scaled numerics + one-hot categoricals,
`class_weight="balanced"`). Validation PR-AUC **0.012**.

**Why a baseline at all:** it's a transparent floor. If a linear model already
separated the classes, the fancy features and the tree wouldn't be load-bearing.
The fact that LightGBM is ~15x better tells me the value is in the non-linear
interactions and the entity-history features — it's not just "the features are
obvious".

**Shipped: LightGBM.**

- **Why gradient-boosted trees:** the feature set is a mix of low-cardinality
  categoricals and skewed counts. Trees handle that natively — no scaling, no
  encoding — they're strong on tabular data, and they're fast to retrain, which
  matters for Part 2.
- **Why not a neural net:** 26 tabular features and ~7k positives. A net is
  overkill, harder to make reproducible, and harder to explain to a regulator.
- **Why not XGBoost:** equivalent results; LightGBM is just faster on native
  categoricals.

**The hyperparameters are deliberately conservative** — and there's a story
here. My first attempt used a large tree (`num_leaves=63`, `learning_rate=0.05`)
and it **early-stopped at iteration 1** — it memorised the ~7k positives in a
single boosting round. So I shrank it hard: `num_leaves=15`, `max_depth=4`,
`min_child_samples=200`, `learning_rate=0.02`, up to 2000 trees with early
stopping, plus L1 and L2 regularisation. Now it learns steadily over ~2000
rounds, and PR-AUC went from 0.18 to 0.68.

**The point to make:** I didn't tune for score — the brief says score isn't the
objective. I tuned for the model to *learn properly instead of memorising*, which
is a different thing, and I can show the before/after.

**Training is an offline job, not notebook code.** `scripts/run_part1.py`
runs the sweep, logs all 5 runs to MLflow, and registers the winner as
`aml-transaction-monitoring@champion`. The notebook *loads* the registered model
and evaluates it. That experiment-vs-analysis separation is exactly what you want
in production, and it's why Part 2 has something real to plug into.

---

## 8. Part 1 — evaluation and the metric

**Headline: PR-AUC (average precision). Not ROC-AUC. Not accuracy or F1.**

- **Accuracy / F1** need a fixed classification threshold. I alert on the top 1%,
  not on a 0.5 cut — so a 0.5-threshold metric measures something the system
  never does.
- **ROC-AUC** — with a thousand negatives per positive, it's dominated by how
  well you rank the easy negatives against each other. It sits at ~0.98 here even
  for a weak model. It can't tell a good alert list from a bad one.
- **PR-AUC** is threshold-free (so it summarises ranking quality) and only
  rewards precision *on the positive class* — which is what an alert list lives
  on.

**The numbers:**

| | PR-AUC | ROC-AUC |
|---|---|---|
| Validation (June) | 0.68 | 0.99 |
| **Test (July–August)** | **0.54** | 0.98 |

PR-AUC 0.54 at 0.1% prevalence is about **540x better than random** (random =
prevalence). For SAML-D at the transaction level, that's a good result — and
score isn't the grade anyway.

**The val→test drop of 0.14 is real model decay** over two months. I want them to
notice it, because it's the bridge to Part 2.

**Alongside PR-AUC I report threshold-specific numbers** — precision, recall,
alerts-per-day at a chosen budget. Those are what an operations lead actually
needs to plan headcount.

---

## 9. Part 1 — the detection-vs-workload trade-off

**This is what the brief specifically asked about.**

I set the threshold on the **validation** month to hit a 1% alert budget, then
apply it unchanged to the test months. Results on test:

| Alert budget | Alerts/day | Precision | Recall |
|---|---|---|---|
| 0.1% | 29 | 57% | 49% |
| 0.5% | 144 | 16% | 67% |
| **1% (chosen)** | **~405** | **6.4%** | **77%** |
| 2% | 577 | 4.7% | 80% |

**Reading it out:** reviewing about 400 alerts a day — of which roughly 25 are
real — catches about **three-quarters of all laundering**. Tighten the budget to
0.1% and precision jumps to 57% but recall falls to 49% — you miss half.

**The key point:** where you sit on this curve is **not a modelling decision**.
It's a capacity-and-risk-appetite decision for the investigations function. My
job is to produce the curve and let them choose the point — and re-plot it as
the model and the data change.

**If pushed on cost:** you can't put a clean number on a missed SAR — it depends
on the case, and there's regulatory and reputational cost beyond the direct one.
The framing I'd use: the regulator's expectation sets a *floor* on acceptable
recall, investigator headcount sets the alert budget, and the model's quality
determines whether you can satisfy both at once. If you can't, that's a concrete
business case for either more analysts or a better model.

**One more detail on this slide:** when I fixed the threshold on June and applied
it to July–August, the alert rate actually drifted from 1.0% up to 1.4% —
because the score distribution crept up. The threshold is a moving target. Yet
another Part 2 hook.

---

## 10. Part 1 — error analysis and honest limits

**Recall by laundering typology, at the 1% budget:**

- **Strong** — Smurfing 100%, Cash Withdrawal 97%, Single-large 97%, Structuring
  79%. These are volume/velocity patterns, which my entity features capture well.
- **Weak** — Fan-Out 60%, Layered Fan-Out 63%, Bipartite 65%. These are defined
  by *graph structure*, and I have no graph features.

**This slice is the diagnostic that tells the investigation team what to trust.**

**The honest limitations — I raise these before they do:**

1. **No graph features** → weak on fan-out / bipartite. Top of the backlog.
2. **The model decays measurably** over two months (0.68 → 0.54). Part 2 handles
   it.
3. **Synthetic, instantaneous labels.** In reality a SAR outcome comes back
   months later, which changes how retraining and evaluation have to work
   (Part 2).
4. **A single global threshold.** A per-corridor threshold — a different cut for
   domestic card vs cross-border — would very likely lift recall.
5. **One test period.** I'd want several before trusting a small model-vs-model
   difference in production.

---

## 11. Part 2 — the lifecycle loop

**The setup:** the Part 1 model is deployed as `@champion`. New batches arrive
daily. I replay **July and August as 54 daily production batches**.

**Why Part 2 exists — the concrete case:** Part 1 already showed PR-AUC 0.68 →
0.54 and the alert rate drifting 1.0% → 1.4% in two months. The model is not
static.

**The loop I operate:** every daily batch is validated → features built → scored
→ alerts out → **and every batch is monitored**. When monitoring fires a
trigger, I retrain on a rolling window; the new model is a challenger; it's
promoted only after a metrics gate and a human sign-off.

**The brief says demonstrate one or two capabilities.** I built **three that hang
together** — monitoring, the retraining trigger, and challenger promotion — and
ran them for real. The rest is conceptual: registry mechanics, shadow
deployment, the delayed-label problem, feature-store parity, governance,
incident response.

**The framing point that drives the whole design:** the thing we truly care
about — did an alert become a confirmed SAR — isn't known for *months*. So the
lifecycle has to run on **proxy signals**, with the true-performance signal
arriving late. Everything in Part 2 is shaped by that constraint.

---

## 12. Part 2 — monitoring

**Four layers, cheapest signal first**, each compared to a **reference snapshot**
that's built at deployment from the training distributions and versioned with the
model — so "drift against what" is never ambiguous.

**Layer 1 — data quality.** Volume vs the 30-day norm, null rates, unseen
categorical values, schema. *On the replay it caught the final batch: 8,000 rows
against an expected 30,000, because the data ends mid-day on the 23rd.* It's an
incomplete feed — the monitor gates the downstream metrics rather than trusting
them. **A broken feed is the number-one cause of what looks like model failure.**

**Layer 2 — input drift (PSI).** Population Stability Index per feature — the
retail-banking standard, with well-known thresholds (0.1 moderate, 0.2
significant). *On the replay: essentially flat, max around 0.13 — except two
features.* `sender_` and `receiver_prior_txn_count` are **unbounded cumulative
counters**. Their PSI climbs to 1.4 over two months — not because behaviour
changed, but because they grow with calendar time by construction. So the
monitor did two jobs: confirmed the real inputs are stable, **and surfaced a
feature-design bug.** I tag those two as known structural drift, exclude them
from the alarm, and put "cap or window them" on the Part 1 backlog. *This is
exactly what you deploy monitoring to catch.*

**Layer 3 — prediction drift.** The score distribution (KS test) and the alert
rate at the fixed threshold. *This is the real signal: the alert rate drifts from
1.2% in July to 1.7% in August and breaches the band from August 1st.* The fixed
threshold no longer delivers the 1% budget because the scores crept up.

**Layer 4 — delayed performance.** Once dispositions arrive (weeks), precision on
the alerted cases; once SARs arrive (months), recall on a **sampled reservoir of
non-alerted transactions** — so the recall estimate isn't biased by what we chose
to alert on. *On the replay, with labels available: precision 7.4% → 5.5%, recall
80% → 72%.*

**Why PSI and not just "use Evidently":** Evidently is a fine library and I'd
likely use it in production for the reporting layer. But PSI is three lines of
maths, it's the metric an AML reviewer expects, and I wanted the drift logic to
be transparent and testable rather than a black box. `tests/test_drift.py` checks
PSI is ~0 for identical distributions and >0.2 for a shifted one.

---

## 13. Part 2 — the retraining trigger and promotion

**The trigger fires if *any* of:**

- **scheduled monthly floor** — so the model never goes stale silently
- **alert rate outside [0.7%, 1.5%] for 3 consecutive days**
- **input PSI > 0.2 on any feature, or > 0.1 on ≥ 3 features, for 5 days**
- **delayed precision drop > 10 points** vs the deployment baseline, once labels
  land

*On the replay, the trigger fired on August 3rd — reasons: the scheduled floor
and the alert-rate drift.*

**Why a blend of triggers, not just one:**

- **Schedule only** — misses fast drift between runs.
- **Measured performance only** — the signal is months late; by the time it
  fires, months of degraded alerting have already happened.
- So: proactive (drift) *and* bounded (schedule).

**Retrain window: a rolling 6–8 months, not all history.** The model should track
current behaviour; very old laundering patterns may not be representative. On the
replay I retrained on December–July.

**Promotion — a challenger is never promoted automatically:**

1. Shadow-score recent batches; compare on the most recent labelled slice
   (August).
2. Gate: the challenger must beat the champion on PR-AUC *and* recall-at-budget,
   and not increase alert-rate volatility.
3. A model owner and compliance approve — **this is a regulated control, not a
   metric race.**
4. The challenger takes the `@champion` alias; the previous version stays
   registered and loadable.

*On the replay: challenger PR-AUC 0.69 vs champion 0.45 on August, recall 0.82 vs
0.66. The retrain recovers the performance the drift cost us.*

**Why aliases, not version numbers:** the scoring code always asks for
`models:/aml-transaction-monitoring@champion` — it never pins `v3`. So promotion
is one alias move, and rollback is the same move in reverse. **A model change is
a registry operation, not a redeploy.** That's the whole value of having a
registry.

**Why not auto-promote if the gate passes:** a metric win on one recent slice is
not proof of no regression elsewhere, and a regulated AML control needs a human
checkpoint. The gate does the filtering; the human does the accountability.

**The delayed-label problem, stated plainly** (this is the deepest question they
can ask): I have two label sources. *Investigator dispositions* — fast, but only
for transactions we alerted on, so a biased sample; usable for precision
monitoring, risky for training because it teaches the model to agree with
itself. *Confirmed SARs* — unbiased ground truth, but months late. The
mitigation is the **reservoir**: sample a small number of non-alerted
transactions every period for manual review, to get unbiased recall estimates
and de-bias the training labels.

---

## 14. Part 3 — production architecture

**The guiding decision: batch, not streaming.** Transaction *monitoring* is
post-event — the label and the workflow both come after the transaction settles,
and there's no accept-or-decline decision at transaction time. Batch is simpler,
cheaper, reproducible, and the entity-history features are naturally a batch
computation. Streaming only earns its complexity if the business needs faster
interdiction on a specific high-risk corridor — and then I'd add streaming *just
for that*, not rebuild everything.

**The flow, left to right:**

```
S3 raw (immutable,        →  Validate            →  Features           →  Score
 date-partitioned,           Great Expectations      Glue / EMR            SageMaker Batch Transform
 KMS-encrypted)              / Deequ                  (PySpark)            loads @champion from MLflow
                             fail → quarantine + page                     │
                                                                         ▼
                                    Monitor            ←   Rank + alert   →   case-management tool
                                    Evidently /             DynamoDB / SQS     (SHAP reason attached)
                                    Model Monitor           top 1% by score
                                    │
                                    ▼
                             CloudWatch metrics + alarms  →  SNS  →  retrain Step Function
```

- **Orchestration:** Step Functions — one state machine for the daily pipeline,
  one for retrain → shadow → promote. Native retries, alarms, cheap to run.
- **Registry + tracking:** MLflow on Fargate + RDS + S3.
- **IaC:** Terraform. (There's a ~40-line sketch in `docs/architecture.md`.)
- **Security:** KMS everywhere, VPC-isolated compute, IAM least privilege,
  account numbers tokenised in the curated zone, CloudTrail for audit.
- **Reproducibility:** the raw zone is immutable and date-partitioned, so any
  past batch can be re-scored with any model version.

**The trade-offs — know these cold, one alternative each:**

| Decision | Choice | Alternative | Why this way |
|---|---|---|---|
| Serving | Daily batch | Streaming (Kinesis + online scoring) | Post-event problem; no real-time decision; batch avoids an online feature store and train/serve skew |
| Compute | SageMaker | EKS + Kubeflow | Managed, minimal ops for a small team. EKS if the org already runs k8s and wants portability + cost control at scale |
| Registry | MLflow self-hosted | SageMaker Model Registry | One tool for tracking *and* registry, portable, team standard. Cost: we run the server |
| Feature compute | Glue / EMR Serverless | pandas on a big instance | pandas is fine at today's 9.5M rows, not at 10x. Spark scales, serverless-managed |
| Feature storage | S3 table + shared feature code | SageMaker Feature Store | Batch-only, no online serving — an offline table + shared code gives train/serve parity without the infrastructure |
| Case management | Buy | Build a UI | Not the differentiation; investigators want it in their existing workflow |
| Orchestration | Step Functions | Airflow / MWAA | Serverless, native retries/alarms, cheaper. Airflow if the team wants a richer DAG ecosystem and already runs it |

**The thread through all of these:** bias toward less operational burden for a
small team, and always name the condition that would flip the answer.

---

## 15. Part 4 — delivery

**Guiding principle: reliability → automation → sophistication.** This is a
regulated control; the biggest risks are operational and regulatory, not
modelling. So: make it trustworthy and observable, *then* self-sustaining, *then*
better. Never the reverse. The model **never hard-replaces** the rules engine —
it goes shadow, then parallel, then staged retirement.

**Months 0–3 — make it trustworthy.**
- Reliable daily pipeline with data validation and failure alerting.
- One model, registered, fully monitored.
- Runs in **shadow mode** — scores live batches, actions nothing.
- Baselines (minimum recall, alert budget, model-risk doc format) agreed with
  compliance.
- *Why first: you can't operate, or get regulatory sign-off for, a control you
  can't see.*
- **Exit criteria:** pipeline SLA met for 4 weeks; shadow recall ≥ the rules
  system at the agreed budget; compliance approves go-live in parallel.

**Months 3–6 — make it self-sustaining.**
- Automated retraining with challenger evaluation and human-approved promotion.
- The **investigator feedback loop** — dispositions become labels — plus
  reservoir sampling for unbiased recall.
- Shared train/serve feature module hardened; offline feature table.
- A documented, auditable process to change the operating threshold.
- Now it runs **in parallel** with the rules engine.
- *Why here: label lag is months, so the feedback loop has to start early; and
  manual retraining doesn't scale once it's live.*
- **Exit criteria:** ≥ 2 successful automated retrains; a challenger promotion
  exercised end to end including a rollback test; the label pipeline delivering.

**Months 6–9 — make it better and auditable.**
- Graph features; a second-stage or typology-aware model for fan-out / bipartite;
  per-corridor thresholds.
- Full model cards, an annual validation schedule, an explainability package for
  regulators, DR + on-call.
- Staged retirement of the rules the model reliably subsumes — one at a time,
  measured, with compliance.
- *Why last: it's the lowest-risk, highest-optionality work. Doing it before the
  foundations hold just gives you a better model you still can't safely run.*

---

## 16. The code, as a flow

You don't need to memorise the source. You need to be able to say why each box
exists. ~1,300 lines across 16 small files.

**Part 1:**

```
Dataset/data/*.csv.gz
  → data/load.py            load_months()            read + global time-sort
  → features/transaction.py add_transaction_features() per-row features
  → features/entity.py      add_entity_features()     causal account-history (leakage-critical)
  → dataset.py              build_feature_frame()     all features, cached to parquet
                            make_dataset()            temporal split → Dataset object
  → models/train.py         fit_baseline / fit_lightgbm   pure fit functions
  → scripts/run_part1.py  offline job: sweep → MLflow → register @champion
  → models/evaluate.py      PR-AUC, operating-point table, per-typology recall
  → notebooks/part1.ipynb    the narrative: load @champion, evaluate
```

**Part 2:**

```
artifacts/features.parquet + @champion
  → monitoring/reference.py  build_reference()   deployment snapshot ("drift against what")
  → monitoring/drift.py      psi / ks            the drift maths (tested)
  → monitoring/batch.py      check_batch()       one daily batch → one metrics row (4 layers)
  → lifecycle.py             should_retrain()    the trigger policy
                             fit_challenger / compare_on_labelled   rolling retrain + comparison
  → scripts/run_part2.py     the demo: 54 batches, monitor, trigger, retrain, register @challenger
  → notebooks/part2.ipynb    the 4-panel dashboard + trigger + challenger table
```

**Supporting:**

- `config.py` — every constant in one place (paths, the month lists, the alert budget)
- `tests/` — `test_entity_features.py` (leakage), `test_drift.py` (PSI sanity)
- `docs/decisions/0001–0006` — the ADRs, one page each
- `docs/part1-model.md`, `part2-lifecycle.md`, `architecture.md`, `delivery-plan.md` — the write-ups
- `presentation/` — the deck, the generator, and the speaker script
