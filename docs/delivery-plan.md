# Part 4 — Delivery: PoC to Production over 3 / 6 / 9 months

## Guiding principle

**Reliability → automation → sophistication.** This is a regulated control. The
biggest risks are operational and regulatory, not modelling. So the order is:
make it trustworthy and observable, then make it self-sustaining, then make it
better. Never the reverse.

The model does **not** replace the existing rules-based system on day one. It runs
in shadow, then in parallel, and only displaces rules once it has a track record
and compliance sign-off.

---

## Month 0–3 — "Make it trustworthy"

**Goal:** the pipeline is reliable and the model's behaviour is fully observable,
running in **shadow mode** — scoring live batches, actioning nothing.

| Workstream | What |
|---|---|
| Pipeline | Step Functions daily job, data validation with quarantine, failure alerting, runbooks |
| Model | One model (Part 1), registered, loaded from the registry — no notebook in the loop |
| Monitoring | Data-quality + input-drift + prediction-drift metrics on every batch, dashboard, alarms |
| Investigator UX | Shadow alerts land in the existing case tool with a SHAP reason, so investigators can eyeball quality |
| Compliance | Agree the baseline: minimum acceptable recall, the alert budget, the model-risk documentation format |

**Why first:** you cannot operate a control you cannot see, and you cannot get
regulatory comfort for a black box. Monitoring and governance scaffolding have to
exist before the model influences a single SAR decision.

**Exit criteria:** pipeline meets its SLA for 4 consecutive weeks; shadow recall
≥ the rules system at the agreed budget; compliance approves go-live in parallel.

---

## Month 3–6 — "Make it self-sustaining"

**Goal:** the model runs in **parallel** with rules, and it maintains itself.

| Workstream | What |
|---|---|
| Retraining | Automated trigger (schedule + drift), rolling-window retrain, challenger evaluation, human-approved promotion |
| Feedback loop | Investigator dispositions flow back as labels; a sampled reservoir of non-alerted transactions is reviewed for unbiased recall |
| Feature parity | Shared train/serve feature module hardened; offline feature table |
| Threshold governance | A documented, auditable process to change the operating threshold |

**Why here and not later:** label lag is the defining constraint of AML ML. The
feedback loop takes months to produce useful volume, so it has to start early.
And once the model is live in parallel, manual retraining does not scale — it has
to be automated before month 6.

**Exit criteria:** ≥ 2 successful automated retrains; a challenger promotion
exercised end to end including rollback test; the label pipeline delivering
dispositions within the target lag.

---

## Month 6–9 — "Make it better and auditable"

**Goal:** improve recall on the weak typologies, harden governance, begin
retiring redundant rules.

| Workstream | What |
|---|---|
| Model | Graph features (fan-in/out degree, component size); a second-stage model or typology-aware model for fan-out / bipartite; per-corridor thresholds |
| Governance | Full model cards, annual validation schedule, explainability package for regulators, DR + on-call |
| Rules retirement | Identify rules the model reliably subsumes; retire them with compliance, one at a time, measured |
| SLAs | Formal SLAs on batch completion, alert latency, monitoring |

**Why last:** model sophistication is the lowest-risk, highest-optionality work.
Doing it before the operational and governance foundations are solid just creates
a better model you still can't safely operate.

---

## Risks and how the plan addresses them

| Risk | Mitigation | When |
|---|---|---|
| Regulator rejects an unmonitored ML control | Monitoring + model docs before go-live; shadow then parallel, never a hard cutover | Month 0–3 |
| Label lag makes the model impossible to evaluate | Feedback loop + non-alerted reservoir sampling started early | Month 3–6 |
| Model decay unnoticed (we saw 0.68 → 0.54 in 2 months) | Drift + alert-rate monitoring with alarms wired to the retraining trigger | Month 0–3 |
| Automated retraining ships a worse model | Challenger + shadow + human approval gate + tested rollback | Month 3–6 |
| Feed breakage misread as model failure | Data validation with quarantine, upstream of scoring | Month 0–3 |
| Over-reliance before the model has a track record | Parallel run with rules; staged, measured rules retirement only from month 6 | Throughout |
