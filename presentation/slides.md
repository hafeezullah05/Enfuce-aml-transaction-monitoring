# Enfuce — AML Transaction Monitoring
### Presentation script + slide content (20 min)

Timing target: Part 1 ~9 min · Part 2 ~5 min · Part 3 ~3 min · Part 4 ~2 min · buffer 1 min.
Build these as Google Slides. "🎤" = what you say. Keep slides sparse; talk to them.

---

## Slide 1 — Title

- **AML Transaction Monitoring — an ML engineering approach**
- Hafeez Ullah · Enfuce technical case · Sep 2026

🎤 "I'll walk through a transaction-monitoring solution — the model, how I'd
operate it, the production architecture, and a delivery plan. I've built Parts 1
and 2 end-to-end; Parts 3 and 4 are design. My focus throughout was production
readiness and the reasoning behind each choice, not the model score — which the
brief also asked for."

---

## Slide 2 — Approach & guiding principle

- Treat it as an **alerting** system, not a classifier: rank transactions, alert
  the top slice, hand them to investigators with finite capacity
- Every decision judged against: *within the daily alert budget, how much
  laundering do we catch vs. how many alerts are wasted?*
- Reproducible pipeline · decisions captured as ADRs · experiment tracking and a
  model registry from day one

🎤 "The core idea: a transaction-monitoring model doesn't make a yes/no decision,
it produces a queue of work for humans. So the right question isn't accuracy —
it's recall at a fixed workload. That framing drives the features, the imbalance
handling, the metrics, everything. And I built it the way I'd build it for
production: pipeline in a package, decisions written down, MLflow tracking every
run."

---

## Slide 3 — The data

- SAML-D synthetic dataset — **9.5M transactions**, Oct 2022 – Aug 2023, 11
  monthly files
- **Prevalence 0.10%** — 1 in ~960 — stable, slight upward drift
- Clean (no missing values); 293k sender accounts, 90% recurring
- Labels are **per-transaction** — accounts that launder do so in only ~1.3% of
  their transactions
- Monthly files map directly onto "batches arriving over time"

🎤 "Nine and a half million transactions over eleven months. Laundering is one in
a thousand. Two things shaped the design: first, the label is per-transaction,
not per-account — a laundering account looks normal 99% of the time, so I can't
just classify accounts. Second, the data is already monthly — that's my natural
unit for the time split, and later for the daily batches in Part 2."

---

## Slide 4 — Key decisions (ADR-backed)

| Decision | Choice | Why |
|---|---|---|
| Serving pattern | Daily **batch** scoring | Monitoring is post-event; no decision at transaction time (ADR-0002) |
| `Laundering_type` column | **Excluded** from features | Target-derived — zero overlap between classes (ADR-0003) |
| Split | **Temporal**, by month | Mirror production; random split leaks future behaviour (ADR-0004) |
| Class imbalance | **Cost-weighting**, not resampling | SMOTE fabricates impossible rows; sweep shows heavy weighting hurts (ADR-0005) |
| Headline metric | **PR-AUC** + alert-budget curve | ROC-AUC misleads at 0.1% prevalence |

🎤 "Five decisions I'd defend. Each is a one-page ADR in the repo. I'll expand
the ones that matter most — the split, imbalance, and metrics — over the next few
slides."

---

## Slide 5 — Feature engineering

- **Transaction-level** (no history, no leakage risk): log-amount, amount bands,
  cross-border, currency-mismatch, hour, weekday
- **Entity-level, causal** (per sender & receiver): prior transaction count, time
  since last transaction, trailing 1d/7d/30d counts, trailing 7d amount
  sum/mean, **amount vs. the account's recent norm**
- Leakage prevention: global time-sort → rolling windows `closed="left"` (window
  excludes the current row) → unit test asserts an account's first transaction
  sees an empty history
- Most important features: the entity-history ones

🎤 "Two groups. The cheap per-transaction features carry some signal — cross-
border transactions are 2.5x more likely to be laundering. But the real signal is
behavioural: how is this account transacting compared to its own history? A
transaction that's 30x the account's 7-day average is a strong flag. The critical
engineering point is leakage: every windowed feature is strictly backward-looking,
enforced by `closed='left'` and locked by a unit test. Feature importance
confirms the entity-history features do the work."

---

## Slide 6 — Class imbalance: it's a threshold problem

- Prevalence 0.10% → the "obvious" fix is `scale_pos_weight ≈ 1000` (neg/pos ratio)
- **I tested it:**

| scale_pos_weight | val PR-AUC | val ROC-AUC |
|---|---|---|
| 1 (shipped) | **0.68** | 0.99 |
| 5 | 0.68 | 0.99 |
| 100 | 0.59 | 0.99 |
| 1003 (naive) | **0.008** | 0.86 |

- Heavy weighting floods the top of the ranking with borderline negatives; ROC-AUC
  barely moves → wrong metric
- SMOTE rejected: interpolating categorical + count features → impossible rows

🎤 "This is the slide I'd most want to talk about. The instinct with 0.1%
prevalence is to reweight the loss by the imbalance ratio. I ran the sweep. It
collapses PR-AUC by a factor of 80 — below the linear baseline — because the model
already ranks well; all the weighting does is push borderline negatives to the top
of the alert queue. Notice ROC-AUC stays at 0.86 the whole time, which is exactly
why ROC-AUC is the wrong headline metric here. The conclusion: imbalance is
handled at the decision threshold, against the alert budget — not in the training
data. SMOTE I rejected on principle — you can't interpolate between 'Cash Deposit'
and 'Cross-border' and get a real transaction."

---

## Slide 7 — Splitting strategy

- 11 monthly blocks: **train** 2022-10…2023-05 · **val** 2023-06 · **test**
  2023-07…08
- No shuffling — every train row precedes every val row precedes every test row
- Encoders + entity features fit on the past only
- Validation = threshold selection · Test = touched once, at the end

🎤 "Strictly temporal. A random split would let an account's August behaviour
inform its prediction in October — that's leakage, and it inflates the score
while hiding the thing I actually need to see, which is how the model decays over
time. Validation is June, used only to pick the operating threshold. Test is July
and August, held out completely."

---

## Slide 8 — Evaluation: why PR-AUC, not ROC-AUC

- At 0.10% prevalence, ROC-AUC is dominated by millions of easy negatives — it
  stays ~0.98 even for a weak model
- PR-AUC (average precision) only rewards precision **on the positives** — what an
  alerting system lives on
- Report both; lead with PR-AUC

| | PR-AUC | ROC-AUC |
|---|---|---|
| Validation (Jun) | 0.68 | 0.99 |
| **Test (Jul–Aug)** | **0.54** | 0.98 |

🎤 "ROC-AUC on this problem is 0.98 and basically meaningless — with a thousand
negatives per positive, you can rank almost all of them correctly and still have
a useless alert list. PR-AUC is the honest summary. And notice the drop from
validation to test — 0.68 to 0.54 in two months. That's real model decay, and
it's the bridge to Part 2."

---

## Slide 9 — Evaluation: the detection vs. workload trade-off

- Operating point: threshold fixed on validation at the **1% alert budget**,
  applied to test

| | test |
|---|---|
| Alerts / day | ~405 |
| Precision | 6.4% |
| **Recall** | **77%** |

- Full curve: 0.1% budget → 29 alerts/day, 57% precision, 49% recall · 2% budget
  → 577/day, 80% recall
- Alert rate drifts 1.0% → 1.4% over two months (fixed threshold, drifting scores)

🎤 "Here's the trade-off quantified. Reviewing about 400 alerts a day — roughly 22
of which are real — catches about three-quarters of all laundering. Tighten the
budget and precision goes up but you miss more. That choice — where to sit on this
curve — isn't a modelling decision, it's a capacity and risk-appetite decision for
the investigations team. My job is to give them the curve. One more thing: the
alert rate crept from 1% to 1.4% because the scores drifted up — the fixed
threshold slowly widened the net. Another Part 2 hook."

---

## Slide 10 — Where recall comes from, and honest limitations

- **Strong**: Smurfing 100%, Cash_Withdrawal 97%, Structuring 79% — volume /
  velocity patterns our features capture
- **Weak**: Fan-Out 60%, Layered-Fan-Out 63% — pure graph-structure patterns, and
  we have **no graph features yet**
- Limitations feeding Parts 2 & 4:
  - No graph features
  - Model decay over ~2 months
  - Synthetic, instantaneous labels — real SAR outcomes lag by months

🎤 "Slicing recall by laundering typology tells the investigation team what to
trust. We're strong where behaviour is the signal — structuring, smurfing. We're
weak on fan-out and bipartite patterns, which are about graph structure, and I
haven't built graph features — that's the top of my future-work list. And a
caveat I'd raise early: these labels are synthetic and instant. In reality a SAR
outcome comes back months later, which changes how you retrain and evaluate."

---

## Slide 11 — Part 2: ML lifecycle & MLOps  *(in progress — fill after building)*

- What I built: _______________
- What's conceptual: model registry + champion/challenger, shadow deployment,
  label-lag handling, governance/audit
- Diagram: _______________

🎤 *(to write)*

---

## Slide 12 — Part 3: Production architecture  *(design)*

- AWS batch: S3 → feature job (Glue/EMR) → training (SageMaker) → MLflow registry
  → daily batch scoring → alert queue → case management → monitoring
  (Model Monitor / Evidently + CloudWatch)
- Orchestration: Step Functions / MWAA
- IaC: Terraform
- Key trade-offs: batch vs. real-time · SageMaker vs. EKS · build vs. buy

🎤 *(to write — talk to the diagram, hit 2–3 trade-offs)*

---

## Slide 13 — Part 4: Delivery, 3 / 6 / 9 months  *(design)*

- **0–3 mo**: reliable daily batch pipeline, one model, monitoring, human-in-the-
  loop; shadow mode first
- **3–6 mo**: automated retraining + challenger models, feature store,
  investigator feedback loop closing the label gap
- **6–9 mo**: graph features / typology-aware models, governance hardening, SLAs,
  threshold automation

🎤 *(to write)*

---

## Slide 14 — Summary

- Framed as alerting, not classification — recall at a fixed workload
- Causal features, temporal split, imbalance as a threshold problem — each choice
  tested and documented
- Test: PR-AUC 0.54; **77% of laundering caught at ~400 alerts/day**
- Model decays measurably in 2 months → Part 2 makes that operable
- Built with the production seams already in place: package, ADRs, MLflow, registry

🎤 "To summarise: I treated this as an alerting problem, made every methodological
choice with the investigator workload in mind, and tested the choices rather than
asserting them. The model catches about 77% of laundering at 400 alerts a day —
and it visibly decays over two months, which is exactly what Part 2 is for. Happy
to go deeper on any of it."

---

## Anticipated questions (prep, not slides)

- **"PR-AUC 0.68 on validation seems high — leakage?"** → temporal split + causal
  features + leakage unit test; features encode the behaviour the typologies are
  built from; and it drops to 0.54 on true held-out test, which is what you'd
  expect if it's honest.
- **"Why LightGBM?"** → categorical + count feature mix handled natively, strong
  on tabular, fast to retrain. Baseline LogReg (0.012) shows the gain is real.
- **"Why not just lower the threshold to catch more?"** → recall/workload curve;
  investigators are the bottleneck, not the model.
- **"How would you get real labels?"** → investigator dispositions on alerts +
  confirmed SARs, with months of lag; design retraining around that lag.
- **"scale_pos_weight=1 with 0.1% positives — the model predicts ~all-negative?"**
  → we don't threshold at 0.5; we rank and take the top 1%. Calibration of the
  absolute probability doesn't matter, ranking does.
- **"Cost of a false negative vs false positive?"** → can't price a missed SAR
  precisely; frame as: regulator expectation sets a minimum recall, capacity sets
  the budget, model quality determines whether both can be met.
