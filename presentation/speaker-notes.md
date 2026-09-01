# Speaker notes — AML Transaction Monitoring

Read from tablet. **Do not hand these over.** Deck: `AML-Transaction-Monitoring.pptx` (18 slides).
Target: **20 min presenting**, then ~20 min Q&A.
Budget — Part 1 (slides 1–11) ~8 min · Part 2 (12–14) ~5 · Part 3 (15–16) ~4 · Part 4 (17) ~2 · summary (18) ~1.
Pace markers: **slide 7 by minute 4**, **slide 12 by minute 8**, **slide 17 by minute 17**.
If behind: on Part 1, drop the "rejected alternative" asides and speak only the core sentence of each slide.

---

## Slide 1 — Title  *(~20 sec)*

"Thanks for making the time. I'll walk through a transaction-monitoring solution:
the model and how I evaluated it, how I'd operate it in production, the
architecture, and a delivery plan. I built Parts 1 and 2 end-to-end; 3 and 4 are
design. Throughout, I prioritised production readiness and the reasoning behind
each choice — which the brief explicitly asked for over model score."

---

## Slide 2 — Approach & principle  *(~1 min)*

"The framing first, because it drives every decision after it.

A transaction-monitoring model doesn't make an accept-or-decline decision. It
produces a **ranked queue of work** for human investigators, and those
investigators have finite daily capacity. So 'is this transaction accuracy high?'
is the wrong question. The right one is: *within the number of alerts a team can
actually review in a day, how much laundering do we catch, and how many of those
alerts are wasted effort?*

That has a concrete consequence for how I built the model. I optimise **ranking
quality** — can the model sort transactions by risk — and I choose the **operating
threshold separately**, against the alert budget. I never use a fixed 0.5
cut-off, because at one-in-a-thousand prevalence a 0.5 cut-off is meaningless.

And I built it with production in mind from the start: the pipeline is an
installable package with unit tests, every significant decision is a one-page
ADR in the repo, and every training run is tracked in MLflow with the chosen
model promoted to a registry. Those aren't afterthoughts bolted on for the
write-up — they're how the thing was assembled."

---

## Slide 3 — The data  *(~1 min)*

"The dataset is SAML-D — synthetic, 9.5 million transactions across 11 months.

Laundering is about **one in a thousand**, and it's roughly stable, with a slight
upward drift over the period — I'll come back to that drift, because it's itself
a monitoring signal.

Two properties shaped the whole design. **First**: the label is per-transaction,
not per-account. An account that launders looks completely ordinary in about 99%
of its transactions. So I can't build an account-level classifier — I have to
score each transaction, using what's known about the account *at that moment*.
**Second**: the data already arrives in monthly files. That's my natural unit —
for the temporal split you'll see in a moment, and later for the daily batches in
Part 2.

There are also 28 named laundering typologies in the data — structuring, fan-out,
cycles, and so on. I use those for error analysis, to see which patterns the
model catches. I never feed them to the model — I'll explain why on the features
slide."

---

## Slide 4 — Answering the brief  *(~1 min)*

"This slide maps the brief's five asks to what I did, and the next six slides go
through them one at a time.

Preprocessing and features — minimal preprocessing, and causal account-behaviour
features where the real signal is. Class imbalance — I effectively leave the loss
alone and handle it at the threshold, and I'll show you the experiment that
justifies that. The split — strictly temporal, by month. The evaluation metric —
PR-AUC plus alerts-per-day at a fixed budget, and I'll give the reasoning. And
the detection-versus-alert-load trade-off — I present the whole curve, because
where you sit on it is the investigations team's call, not the model's.

Each of these is a one-page ADR in the repo, with the rejected alternatives
written down."

---

## Slide 5 — Preprocessing  *(~1.5 min)*

"Preprocessing is deliberately minimal, and I want to be explicit about why.

On load, three things happen: I enforce an explicit schema with typed columns —
that's half the memory and it makes the schema intent visible in code; I build a
single timestamp from the separate date and time fields; and I sort the entire
dataset by time, once, globally — which the causal features depend on.

There are **no missing values** in the raw data, so there's no imputation to do.
The only NaNs anywhere are in the entity features, on an account's very first
transaction — and that NaN is *meaningful*. It means 'this account has no
history', not 'a value is missing'. LightGBM handles that natively; for the
linear baseline I fill it with the median.

The rest of the preprocessing **differs by model**, on purpose. For LightGBM I
keep the categorical columns as a native category dtype — no one-hot encoding —
because the tree splits on them directly, and I do no scaling, because trees are
scale-invariant. The only transform I keep is a log on the amount, and that's
purely so the feature-importance numbers are readable. For the Logistic
Regression baseline I do the full standard treatment — impute, standardise,
one-hot — because linear models require it.

I deliberately did **not** use target encoding or PCA. Both add leakage surface
and make the model harder to reason about, for no measurable gain on this
problem."

---

## Slide 6 — Feature engineering  *(~2 min)*

"Two groups of features.

The **transaction-level** ones are derived from a single row, so there's no
leakage risk at all. Log-amount, some amount bands, a cross-border flag — sender
and receiver bank locations differ — a currency-mismatch flag, hour, weekday.
These carry real but modest signal: cross-border transactions are about two and a
half times more likely to be laundering.

The **entity-level** features are where the signal actually is, and they're the
interesting engineering. For both the sender and the receiver account, I compute:
how many prior transactions the account has had, how long since its last one, its
transaction counts over trailing 1-day, 7-day and 30-day windows, its trailing
7-day amount sum and mean, and — the strongest single feature — this
transaction's amount versus the account's own recent average. A transaction
that's thirty times an account's normal size is a strong flag.

I chose those windows for meaning, not by tuning: **one day catches a burst**,
**seven days captures the account's rhythm**, **thirty days is its baseline**.

Now the critical part — **leakage prevention**. Every one of those windowed
features is strictly backward-looking. I sort globally by time; the rolling
windows use a left-closed interval, which excludes the current transaction and
everything after it; and there's a unit test that constructs a tiny dataset and
asserts that an account's first-ever transaction sees an empty history. If
someone breaks that later, the test fails.

Two things I excluded on purpose. The raw **account IDs** — they're identifiers,
not behaviour, and letting the model memorise specific accounts is both leakage
and useless on new accounts. And **Laundering_type** — its values are disjoint
between the classes, so it's derived from the label. Using it would be leakage;
I keep it only for the error analysis you'll see later.

Feature importance confirms the design: the top features are all entity-history
features."

---

## Slide 7 — Class imbalance  *(~2 min)*  ← **the one to land well**

"This is the slide I'd most like to talk through, because the obvious answer is
wrong.

Prevalence is one in a thousand. There are three broad options: **resample** the
data — SMOTE or undersampling; **cost-weight** the loss so the model pays more
for missing a positive; or **leave the loss alone** and handle the imbalance
where the decision is actually made, at the threshold.

The instinct is to cost-weight by the imbalance ratio — set `scale_pos_weight` to
around a thousand. I ran that as an experiment, sweeping the weight from 1 up to
1003. Here's the result. At weight 1 — essentially off — PR-AUC on validation is
0.68. Crank it to a thousand and PR-AUC **collapses to 0.008** — worse than the
Logistic Regression baseline.

Why? Because the model already ranks well. All the heavy weighting does is inflate
the score of every borderline negative, so the top of the alert queue fills with
near-misses instead of real cases. And notice ROC-AUC barely moves through the
whole sweep — it stays around 0.86 to 0.99. That's a preview of why ROC-AUC is
the wrong metric to steer by.

So I ship weight 1, and I handle the real trade-off at the threshold.

**SMOTE I rejected outright**, and not just on results. SMOTE interpolates between
minority points. My feature space is mostly categorical and count-based — you
can't interpolate halfway between 'Cash Deposit' and 'Cross-border', or have 3.5
transactions in a window. It fabricates rows that can't exist. And the synthetic
points have no timestamp, so they have no place in a time-ordered split. It's the
wrong tool for this data."

---

## Slide 8 — Splitting the data  *(~1.5 min)*

"The split is strictly temporal. Train on the first eight months, October to May.
Validate on June — and I use June *only* to pick the operating threshold. Test on
July and August, and I touched that test set exactly once, at the very end.

Nothing is shuffled. The reason: accounts recur across months, and a random split
would put an account's August transactions in the training set and its October
transactions in the test set. The model would then 'predict' October using
information from August. That's leakage — it inflates the score, and worse, it
**hides the thing I most need to see**, which is how the model degrades over
time.

I considered one alternative properly: **account-grouped k-fold** — keep all of
an account's rows in the same fold. That removes the account-identity leakage,
but it still mixes time periods, so it still tells you an over-optimistic story
about forward performance. For a system that scores tomorrow's transactions,
time has to be the split axis.

One implementation detail: I compute the entity features over the *full* timeline
before splitting. So a test-set transaction in July carries the account's real
history going back to October — exactly what the model would have at inference
time. The split happens after the features are built, not before."

---

## Slide 9 — Choosing the metric  *(~1.5 min)*

"What I optimise and what I report are two different things, and I want to be
precise about both.

**Accuracy and F1 I rejected** because they need a fixed classification
threshold, usually 0.5, and I never use one — I alert on the top 1% of scores.
A 0.5-threshold metric is measuring something the system never does.

**ROC-AUC I rejected as the headline.** With a thousand negatives for every
positive, ROC-AUC is dominated by how well you rank the easy negatives against
each other. It sits around 0.98 here even for a weak model. It's not *wrong*, it's
just uninformative — it can't tell a good alert list from a bad one.

**PR-AUC — average precision — is the headline.** It's threshold-free, so it
summarises ranking quality, and it only rewards precision on the positive class,
which is exactly what an alert list lives on.

Alongside it I report the **threshold-specific** numbers: precision, recall, and
alerts-per-day at a chosen budget. Those are what an operations lead actually
needs.

And the result to notice: PR-AUC is 0.68 on validation but **0.54 on the
held-out test**. That drop over two months is real model decay — and it's the
reason Part 2 exists."

---

## Slide 10 — Detection vs. unnecessary alerts  *(~1.5 min)*

"This is the trade-off the brief specifically asked about, quantified.

The curve is recall against alerts per day. At the operating point I chose — top
1% of daily scores, with the threshold fixed on validation — that's about **400
alerts a day**, of which roughly 22 are real, and it catches about **77% of all
laundering**. Move left on the curve and precision improves but you miss more;
move right and you catch more but you swamp the team.

The key point: **where to sit on this curve is not a modelling decision.** It's a
capacity-and-risk-appetite decision for the investigations function. My job is to
produce the curve and let them choose the point — and to re-plot it as the model
and the data change.

If pushed on cost: you can't put a clean number on a missed SAR. The way I'd
frame it is — the regulator's expectation sets a floor on acceptable recall,
investigator headcount sets the alert budget, and the model's quality determines
whether you can satisfy both at once. If you can't, that's the business case for
either more analysts or a better model.

One more thing on this slide: the alert rate actually drifted from 1% up to 1.4%
over the two test months, because the scores crept up and the threshold was
fixed. The threshold is a moving target — another Part 2 hook."

---

## Slide 11 — Error analysis & limits  *(~1 min)*

"Slicing recall by laundering typology is the diagnostic that tells the
investigation team what to trust.

The model is strong where behaviour is the signal — smurfing at 100%, cash
withdrawal at 97%, structuring at 79%. It's weak on fan-out and bipartite
patterns, in the low 60s. Those are defined by **graph structure** — one account
paying many, many accounts paying one — and I have no graph features yet. That's
the top of my backlog.

Three other honest limitations. The model decays measurably over two months.
These labels are synthetic and instantaneous — in reality a SAR outcome comes
back months later, which changes how retraining and evaluation have to work. And
I use a single global threshold; a per-corridor threshold — different cut-offs
for, say, domestic card versus cross-border — would very likely lift recall."

---

## Slide 12 — Part 2: the lifecycle loop  *(~1.5 min)*

"Part 1 already made the case for Part 2. The model isn't static — PR-AUC fell
from 0.68 to 0.54, and the alert rate drifted from 1% to 1.4%, in two months.

This is the loop I'd operate: every daily batch is validated, features built,
scored, alerts out — and every batch is **monitored**. When monitoring fires, we
retrain on a rolling window; the new model is a challenger; it's promoted only
after a metrics gate and a human sign-off.

The brief asks for one or two capabilities demonstrated. I built **three that
hang together** — monitoring, the retraining trigger, and challenger promotion —
and ran them on the July–August batches. Conceptual: the registry mechanics,
shadow deployment, the delayed-label problem, feature-store parity, governance,
incident response.

The framing point that drives everything: the thing we truly care about — did an
alert become a confirmed SAR — isn't known for months. So the lifecycle runs on
**proxy signals**, with the true-performance signal arriving late."

---

## Slide 13 — Part 2: monitoring, and what it caught  *(~2 min)*

"Four layers, cheapest signal first, each compared to a reference snapshot that's
versioned with the model — so 'drift against what' is never ambiguous.

**Data quality** — volume, nulls, unseen categories, schema. On the replay it
caught the final batch: 8,000 rows against an expected 30,000, because the data
ends mid-day on the 23rd. It's an incomplete feed — the monitor gates the
downstream metrics rather than trusting them. A broken feed is the number-one
cause of what looks like model failure.

**Input drift** — PSI per feature. Essentially flat, max around 0.13 — *except*
two features. `sender_` and `receiver_prior_txn_count` are unbounded cumulative
counters. Their PSI climbs to 1.4 over the two months — not because behaviour
changed, but because they grow with calendar time by construction. So the monitor
did two jobs here: it confirmed the real inputs are stable, and it surfaced a
feature-design bug. I tag those two as known structural drift, exclude them from
the alarm, and put 'cap or window them' on the Part 1 backlog.

**Prediction drift** — the score distribution and the alert rate. This is the
real signal: the alert rate drifts from 1.2% in July to 1.7% in August and
breaches the band from August 1st. The fixed threshold no longer delivers the 1%
budget because the scores crept up.

**Delayed performance** — we have labels here so I show it, but framed as what
arrives weeks later: precision falls from 7.4% to 5.5%, recall from 80% to 72%."

---

## Slide 14 — Part 2: trigger, retrain, promote  *(~1.5 min)*

"The retraining trigger fired on August 3rd — two reasons: the scheduled monthly
floor, and the alert rate outside the band for three consecutive days.

We retrain on a **rolling eight-month window** — December through July — not all
history, because the model should track current behaviour and old patterns may
not be representative.

The new model is a **challenger**. It's judged on the most recent labelled data,
August. And it's clearly better — PR-AUC 0.69 against the champion's 0.45, recall
at the budget 0.82 against 0.66. The retrain recovers the performance the drift
cost us.

But it is **not promoted automatically**. It's registered, the `@challenger`
alias is set, and it waits for a model-owner and compliance sign-off before it
takes the `@champion` alias. The scoring code only ever asks for `@champion` — it
never pins a version — so promotion is one alias move, and rollback is the same
move in reverse. The previous version stays registered and loadable. That's the
whole point of the registry: model changes are a registry operation, not a
deploy."

---

## Slide 15 — Part 3: the architecture  *(~2 min)*

"Production on AWS as a **daily batch pipeline** — and I want to defend the 'batch'
choice because it's the first question. The label and the workflow are both
post-event; there's no accept-or-decline decision at transaction time. Batch is
simpler, it's cheaper, there's no online/offline feature skew, and the
entity-history features are naturally a batch computation. Streaming only earns
its complexity if the business needs faster interdiction on a specific corridor —
and then I'd add streaming just for that, not rebuild everything.

Left to right: raw transactions land in an **immutable, date-partitioned** S3
zone. A validation step — Great Expectations or Deequ — checks the feed and
quarantines a bad batch before it reaches scoring. A Glue or EMR job builds the
features. SageMaker scores the batch, loading 'the Production model' from the
**MLflow registry** — never a file path, so rollback is just reverting the
registry stage. Alerts are ranked, thresholded, and pushed to a queue and into
the case-management tool with a SHAP reason attached. And a monitoring job emits
the metrics from the previous slide and can trigger the retrain state machine.

Orchestration is Step Functions — one for the daily pipeline, one for retrain,
shadow and promote. The registry runs on Fargate. Everything is Terraform.

Because the raw zone is immutable, any past batch can be re-scored with any model
version — that's the reproducibility story a regulator wants."

---

## Slide 16 — Part 3: the trade-offs  *(~2 min)*

"The technology choices, and — more importantly — the alternative and when I'd
switch.

**Batch vs. streaming** — covered. **SageMaker vs. EKS with Kubeflow** — I chose
SageMaker for a small team, less to operate; I'd choose EKS if the org already
runs Kubernetes and wants portability and cost control at scale.

**MLflow vs. SageMaker's own Model Registry** — MLflow gives one tool for
tracking and registry, it's portable across clouds, and it's the standard the
team already uses. The cost is that we run the server ourselves. SageMaker's
registry is less to operate but ties us in.

**Glue or EMR Serverless vs. pandas on a big instance** — pandas is genuinely
fine at today's 9.5 million rows, but not at ten times that; Spark scales and is
serverless-managed.

**No feature store** — because it's batch-only with no online serving. An offline
feature table plus sharing the exact feature code between training and scoring
gives parity without the infrastructure. I'd add a real feature store the day a
real-time path appears.

**Buy the case-management layer** — it's not where the value is, and investigators
want alerts in their existing workflow.

The thread through all of these: bias toward less operational burden for a small
team, and name the condition that would change the answer."

---

## Slide 17 — Part 4: delivery  *(~2 min)*

"From proof of concept to production, phased over nine months. The principle is
**reliability, then automation, then sophistication** — the opposite of the
instinct — because this is a regulated control and the risks are operational, not
modelling. The model never hard-replaces the rules engine; it goes shadow, then
parallel, then staged retirement.

**Months zero to three — make it trustworthy.** A reliable pipeline with data
validation, one model, full monitoring, and it runs in **shadow mode**, actioning
nothing. Baselines — minimum recall, the alert budget — agreed with compliance.
Why first: you can't operate, or get sign-off for, a control you can't see.

**Months three to six — make it self-sustaining.** Automated retraining with
challenger evaluation, and the investigator feedback loop that turns dispositions
into labels, plus reservoir sampling for unbiased recall. Now it runs in parallel
with the rules engine. Why here: label lag is months, so the feedback loop has to
start early, and manual retraining doesn't scale once it's live.

**Months six to nine — make it better and auditable.** Graph features,
per-corridor thresholds, full model cards and an annual validation schedule, and
staged retirement of the rules the model reliably subsumes. Why last: it's the
lowest-risk, highest-optionality work — doing it before the foundations hold just
gives you a better model you still can't safely run.

Each stage has an exit criterion — pipeline SLA met for four weeks; two
successful automated retrains; and so on."

---

## Slide 18 — Summary  *(~1 min)*

"To wrap up.

I treated this as an **alerting** problem, not a classification problem — recall
at a fixed investigator workload. I made every methodological choice by testing
it against a concrete alternative — temporal versus random split, cost-weighting
versus threshold, PR-AUC versus ROC-AUC — rather than asserting it. The model
catches about **77% of laundering at 400 alerts a day**, and it **visibly decays
over two months**, which is exactly what Part 2 is built to handle. And it's
assembled with the production seams already in place — a tested package, ADRs,
experiment tracking, a model registry.

Happy to go deeper on any part of it."

---

# Anticipated questions

**"PR-AUC 0.68 on validation seems high for this problem — is there leakage?"**
Three safeguards: the split is temporal, every windowed feature is strictly
backward-looking, and there's a unit test that locks the no-leakage behaviour.
The features encode the same behaviour the SAML-D typologies are built from, so
it's genuine signal — and it drops to 0.54 on the true held-out test, which is
what you'd expect if it's honest. If it were leaking, test would match validation.

**"Why LightGBM and not a neural net / logistic regression / XGBoost?"**
The feature set is a mix of low-cardinality categoricals and skewed counts —
gradient-boosted trees handle that natively, with no scaling or encoding, and
they're fast to retrain, which matters for Part 2. Logistic Regression is the
baseline and scores 0.012, so the non-linearity and feature interactions are
doing real work. A neural net is overkill for ~30 tabular features and 10k
positives. XGBoost would be equivalent — LightGBM is just faster on categoricals.

**"Why not just lower the threshold to catch more laundering?"**
Because investigators are the bottleneck, not the model. Lower the threshold and
you generate more alerts per day than the team can review; the backlog grows,
alerts age past usefulness, and the control fails in practice. The curve on
slide 10 is exactly this constraint.

**"How would you get real labels in production?"**
Two sources, both lagged. Investigator dispositions on the alerts we raise —
available in days to weeks, but only for transactions we alerted on, so it's a
biased sample. And confirmed SARs / law-enforcement feedback — the ground truth,
but months later. Retraining has to be designed around that lag; you can't assume
fresh labels.

**"scale_pos_weight = 1 with 0.1% positives — doesn't the model just predict everything negative?"**
Its raw probabilities are low, yes — but we never threshold at 0.5. We rank by
score and alert on the top 1%. The absolute calibration doesn't matter for that;
only the ordering does. If we needed calibrated probabilities for a downstream
cost calculation, I'd add a calibration step — but for ranking it's unnecessary.

**"What's the actual cost of a false negative versus a false positive?"**
You can't price a missed SAR precisely — it depends on the case. The framing I'd
use: the regulator sets a minimum acceptable recall, investigator capacity sets
the alert budget, and the model determines whether both can be met at once. If
they can't, that's a concrete business case for more analysts or a better model.

**"The model decays in two months — how often would you retrain?"**
The honest answer is: monitor and let the data decide, rather than fix a cadence.
The drift monitor and the alert-rate tracker are the trigger. As a starting
default, monthly retraining with a challenger evaluation, tightened if the
monitors fire more often. And retrain on a rolling window, not all history, so
the model tracks current behaviour.

**"What would you do first with another week?"**
Graph features — sender fan-out degree, receiver fan-in degree over a window,
connected-component size. That directly targets the fan-out and bipartite
typologies where recall is weakest, and the data supports it.

**"How do you know 1% is the right alert budget?"**
I don't — it's illustrative. The real number comes from the investigations team's
capacity: analysts times cases-per-analyst-per-day. The deliverable is the curve;
they pick the point, and we revisit it as headcount and model quality change.
