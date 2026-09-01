/* Build the AML Transaction Monitoring deck.
 *
 *   cd presentation
 *   node build_deck.js            # -> AML-Transaction-Monitoring.pptx
 *
 * Speaker script lives in presentation/speaker-notes.md (read from tablet).
 * This file carries only short cue notes in each slide's notes pane.
 */
const pptxgen = require("pptxgenjs");

const NAVY = "1E2761";
const INK = "141C33";
const ICE = "CADCFC";
const TEAL = "00A896";
const CORAL = "F96167";
const WHITE = "FFFFFF";
const MUTE = "5A6B8C";
const LINE = "D8DEEA";

const HEAD = "Cambria";
const BODY = "Calibri";

const p = new pptxgen();
p.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
p.theme = { headFontFace: HEAD, bodyFontFace: BODY };

const MX = 0.6;
const CW = 13.3 - MX * 2;

function darkSlide() {
  const s = p.addSlide();
  s.background = { color: NAVY };
  return s;
}
function lightSlide(part, section) {
  const s = p.addSlide();
  s.background = { color: WHITE };
  if (part) {
    s.addShape(p.ShapeType.ellipse, { x: MX, y: 0.5, w: 0.42, h: 0.42, fill: { color: NAVY } });
    s.addText(String(part), {
      x: MX, y: 0.5, w: 0.42, h: 0.42, align: "center", valign: "middle",
      fontFace: BODY, fontSize: 13, bold: true, color: WHITE, isTextBox: true, margin: 0,
    });
    s.addText(section.toUpperCase(), {
      x: MX + 0.55, y: 0.5, w: CW - 0.55, h: 0.42, valign: "middle",
      fontFace: BODY, fontSize: 12, bold: true, color: MUTE, charSpacing: 2, isTextBox: true, margin: 0,
    });
  }
  return s;
}
function title(s, text, y = 1.05) {
  s.addText(text, {
    x: MX, y, w: CW, h: 0.9, fontFace: HEAD, fontSize: 29, bold: true, color: NAVY,
    isTextBox: true, margin: 0,
  });
}
function bullets(s, items, opt = {}) {
  s.addText(
    items.map((t, i) => ({
      text: t,
      options: { bullet: { code: "2022", indent: 12 }, breakLine: i < items.length - 1, paraSpaceAfter: 8 },
    })),
    {
      x: opt.x ?? MX, y: opt.y ?? 2.1, w: opt.w ?? CW, h: opt.h ?? 4.6,
      fontFace: BODY, fontSize: opt.fontSize ?? 15, color: opt.color ?? INK, valign: "top",
      isTextBox: true, margin: 0, lineSpacingMultiple: 1.05,
    }
  );
}
function card(s, x, y, w, h, fill) {
  s.addShape(p.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.06, fill: { color: fill }, line: { color: LINE, width: 1 } });
}
function cardHead(s, x, y, w, t, color = NAVY) {
  s.addText(t, { x: x + 0.3, y: y + 0.16, w: w - 0.6, h: 0.35, fontFace: BODY, bold: true, fontSize: 13.5, color, isTextBox: true, margin: 0 });
}
function cardBody(s, x, y, w, h, t) {
  s.addText(t, { x: x + 0.3, y: y + 0.6, w: w - 0.6, h, fontFace: BODY, fontSize: 12, color: INK, isTextBox: true, margin: 0, lineSpacingMultiple: 1.05 });
}
function stat(s, x, y, w, big, label, color = NAVY) {
  s.addText(big, { x, y, w, h: 0.95, align: "center", fontFace: HEAD, fontSize: 40, bold: true, color, isTextBox: true, margin: 0 });
  s.addText(label, { x, y: y + 0.95, w, h: 0.6, align: "center", fontFace: BODY, fontSize: 12, color: MUTE, isTextBox: true, margin: 0 });
}
function note(s, t) { s.addNotes(t); }

/* ------------------------------------------------------------------ 1. Title */
{
  const s = darkSlide();
  s.addShape(p.ShapeType.ellipse, { x: 9.7, y: -1.8, w: 6, h: 6, fill: { color: INK } });
  s.addShape(p.ShapeType.ellipse, { x: 11.2, y: 4.6, w: 3.4, h: 3.4, fill: { color: "22305C" } });
  s.addText("AML Transaction Monitoring", {
    x: MX, y: 2.5, w: 9.4, h: 1.1, fontFace: HEAD, fontSize: 44, bold: true, color: WHITE, isTextBox: true, margin: 0,
  });
  s.addText("An ML-engineering approach to detecting suspicious activity", {
    x: MX, y: 3.7, w: 9.4, h: 0.6, fontFace: BODY, fontSize: 17, color: ICE, isTextBox: true, margin: 0,
  });
  s.addText("Hafeez Ullah   ·   Enfuce technical case", {
    x: MX, y: 5.9, w: 9, h: 0.4, fontFace: BODY, fontSize: 13, color: MUTE, isTextBox: true, margin: 0,
  });
  note(s, "Parts 1 & 2 built end-to-end; 3 & 4 are design. Focus: production readiness and the reasoning behind each choice, not model score.");
}

/* ------------------------------------------------------ 2. Approach & principle */
{
  const s = lightSlide(1, "Approach");
  title(s, "An alerting system, not a classifier");
  bullets(s, [
    "The model does not decide yes/no — it produces a ranked queue of work for investigators, who have finite daily capacity",
    "Every choice judged against one question: within the daily alert budget, how much laundering do we catch vs. how many alerts are wasted?",
    "Consequence: I optimise ranking quality and choose the operating point separately — not a fixed 0.5 cut-off",
    "Built with production seams in place: pipeline as a package, decisions as ADRs, MLflow tracking + model registry from the first run",
  ], { w: 7.1, y: 2.2 });
  card(s, 8.1, 2.2, 4.6, 3.4, "F4F7FC");
  s.addText("“Recall at a fixed\ninvestigator workload”", {
    x: 8.4, y: 2.55, w: 4.0, h: 1.5, fontFace: HEAD, fontSize: 21, bold: true, italic: true, color: NAVY, isTextBox: true, margin: 0,
  });
  s.addText("— the objective. Not accuracy,\nnot F1, not a single number.", {
    x: 8.4, y: 4.15, w: 4.0, h: 0.9, fontFace: BODY, fontSize: 12, color: MUTE, isTextBox: true, margin: 0,
  });
  note(s, "A monitoring model creates human work; the right metric is recall at a workload. That framing drives features, imbalance handling, and metrics.");
}

/* --------------------------------------------------------------- 3. The data */
{
  const s = lightSlide(1, "The data");
  title(s, "SAML-D — synthetic transaction data over time");
  stat(s, MX, 2.2, 3.9, "9.5M", "transactions · Oct 2022 – Aug 2023");
  stat(s, MX + 4.15, 2.2, 3.9, "0.10%", "laundering · 1 in ~960", CORAL);
  stat(s, MX + 8.3, 2.2, 3.8, "11", "monthly files = ready-made batches");
  bullets(s, [
    "Prevalence stable, with a slight upward drift (0.10% → 0.13%) — itself a monitoring signal",
    "Labels are per-transaction: an account that launders looks normal in ~99% of its transactions — so we score transactions, not accounts",
    "Clean data, 293k sender accounts, 90% recurring — enough history for behavioural features",
    "28 named laundering typologies (Structuring, Fan-Out, Cycle, …) — used for error analysis, never as a feature",
  ], { y: 4.0 });
  note(s, "9.5M transactions, laundering is 1 in 1000. Two design drivers: per-transaction labels, and the natural monthly batching.");
}

/* --------------------------------------------------------- 4. Answering the brief */
{
  const s = lightSlide(1, "Key decisions");
  title(s, "The brief's five asks — how each is addressed");
  const hopt = { bold: true, color: WHITE, fill: { color: NAVY } };
  const rows = [
    [{ text: "Requirement", options: hopt }, { text: "Choice", options: hopt }, { text: "Core reason", options: hopt }],
    ["Preprocessing & features", "Minimal preprocessing + causal entity features", "Trees need little prep; behaviour is where the signal is"],
    ["Class imbalance", "Cost-weighting ≈ off; handle at the threshold", "A sweep shows heavy re-weighting collapses PR-AUC"],
    ["Train / val / test split", "Temporal, by calendar month", "Random split leaks an account's future behaviour"],
    ["Evaluation metric", "PR-AUC + alerts-per-day at a fixed budget", "ROC-AUC is ~0.98 even for weak models at 0.1%"],
    ["Detection vs. alert load", "Present the whole recall / workload curve", "The operating point is the team's decision, not the model's"],
  ];
  s.addTable(rows, {
    x: MX, y: 2.15, w: CW, colW: [2.8, 3.7, 5.6],
    fontFace: BODY, fontSize: 12, color: INK, valign: "middle",
    border: { type: "solid", color: LINE, pt: 1 }, fill: { color: WHITE }, rowH: 0.62,
  });
  s.addText("Each row is a one-page ADR in the repo (Context / Decision / Consequences / Rejected alternatives).", {
    x: MX, y: 6.35, w: CW, h: 0.4, fontFace: BODY, fontSize: 11.5, italic: true, color: MUTE, isTextBox: true, margin: 0,
  });
  note(s, "This slide maps the brief to the next six. Walk it fast, then go deep on each.");
}

/* ------------------------------------------------------- 5. Preprocessing */
{
  const s = lightSlide(1, "Preprocessing");
  title(s, "Preprocessing — deliberately minimal");
  const cw3 = (CW - 0.5) / 2;
  card(s, MX, 2.15, cw3, 2.95, "F4F7FC");
  cardHead(s, MX, 2.15, cw3, "Done on load");
  cardBody(s, MX, 2.15, cw3, 2.3,
    "· explicit dtypes — schema intent in code, half the memory\n" +
    "· one timestamp from Date + Time; global sort by time\n" +
    "· no missing values in the raw data → no imputation\n" +
    "· the only NaN is an entity feature on an account's first-ever transaction — it means “no history”, not “missing”");
  card(s, MX + cw3 + 0.5, 2.15, cw3, 2.95, WHITE);
  cardHead(s, MX + cw3 + 0.5, 2.15, cw3, "Differs by model", MUTE);
  cardBody(s, MX + cw3 + 0.5, 2.15, cw3, 2.3,
    "LightGBM: categoricals kept as native category dtype — no one-hot; trees are scale-invariant, so no scaling\n\n" +
    "Baseline (LogReg): median-impute → standardise → one-hot — linear models require it");
  bullets(s, [
    "log1p(amount) is the one transform kept for the tree — purely for readable feature importance",
    "No target encoding, no PCA — they add leakage surface and opacity for no measurable gain here",
  ], { y: 5.4, fontSize: 13 });
  note(s, "Preprocessing is minimal for the tree by design. The heavy prep (scaling, one-hot) is only for the linear baseline. NaN in entity features is meaningful, not missing.");
}

/* -------------------------------------------------- 6. Feature engineering */
{
  const s = lightSlide(1, "Feature engineering");
  title(s, "Feature engineering — behaviour is the signal");
  card(s, MX, 2.1, 5.9, 1.7, "F4F7FC");
  cardHead(s, MX, 2.1, 5.9, "Transaction-level  (single row, no leakage risk)");
  cardBody(s, MX, 2.1, 5.9, 1.1, "log-amount · amount bands · cross-border · currency-mismatch · hour · weekday");
  card(s, MX + 6.2, 2.1, 6.1, 1.7, "F4F7FC");
  cardHead(s, MX + 6.2, 2.1, 6.1, "Entity-level  (causal, per sender & receiver)");
  cardBody(s, MX + 6.2, 2.1, 6.1, 1.1, "prior txn count · time since last txn · trailing 1d / 7d / 30d counts · trailing 7d sum & mean · amount vs. the account's recent norm");
  bullets(s, [
    "Windows chosen for meaning: 1d catches bursts · 7d the account's rhythm · 30d its baseline",
    "Leakage prevention: global time-sort → rolling windows closed=\"left\" (current row excluded) → a unit test asserts an account's first transaction sees an empty history",
    "Excluded on purpose: account IDs (identifiers, not behaviour) and Laundering_type (target-derived)",
    "Feature importance is dominated by the entity-history features",
  ], { y: 4.15, fontSize: 13.5 });
  note(s, "Cheap features carry some signal (cross-border 2.5x). Real signal: a transaction 30x the account's 7-day average. Leakage enforced by closed='left' + a unit test.");
}

/* --------------------------------------------------------- 7. Class imbalance */
{
  const s = lightSlide(1, "Class imbalance");
  title(s, "Handling class imbalance — a threshold problem");
  s.addChart(
    p.ChartType.bar,
    [{ name: "val PR-AUC", labels: ["spw = 1\n(shipped)", "spw = 5", "spw = 100", "spw = 1003\n(naive neg/pos)"], values: [0.68, 0.68, 0.59, 0.008] }],
    {
      x: MX, y: 2.15, w: 6.5, h: 4.5, barDir: "col",
      chartColors: [NAVY], showLegend: false, showTitle: false,
      showValue: true, dataLabelPosition: "outEnd", dataLabelColor: INK, dataLabelFontSize: 11, dataLabelFormatCode: "0.00",
      valAxisMaxVal: 0.8, valAxisMinVal: 0, valAxisHidden: true, valGridLine: { style: "none" },
      catAxisLabelColor: INK, catAxisLabelFontSize: 10, catGridLine: { style: "none" },
    }
  );
  bullets(s, [
    "Three options weighed: resample (SMOTE / undersample) · cost-weight the loss · leave the loss alone and move the threshold",
    "The “obvious” cost-weight — the ~1000x imbalance ratio — collapses PR-AUC below the linear baseline (0.012): it floods the alert queue with borderline negatives",
    "ROC-AUC barely moves (0.86 → 0.99) → wrong metric to steer by",
    "SMOTE rejected: interpolating categorical + count features invents impossible rows and has no place in a time-ordered split",
  ], { x: 7.3, y: 2.15, w: CW - 6.7, fontSize: 12.5 });
  note(s, "The slide I most want to discuss. Ran the sweep. Weighting by the imbalance ratio makes it 80x worse. Model already ranks well; weighting just reorders the queue badly.");
}

/* --------------------------------------------------------- 8. Split strategy */
{
  const s = lightSlide(1, "Splitting the data");
  title(s, "Splitting train / validation / test — strictly by time");
  const months = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"];
  const bx = MX, bw = CW, cw = bw / 11, by = 2.35, ch = 0.95;
  months.forEach((m, i) => {
    const kind = i < 8 ? NAVY : i === 8 ? TEAL : CORAL;
    s.addShape(p.ShapeType.rect, { x: bx + i * cw, y: by, w: cw - 0.06, h: ch, fill: { color: kind } });
    s.addText(m, { x: bx + i * cw, y: by + ch + 0.06, w: cw - 0.06, h: 0.28, align: "center", fontFace: BODY, fontSize: 10, color: MUTE, isTextBox: true, margin: 0 });
  });
  s.addText([
    { text: "TRAIN  8 months", options: { color: NAVY, bold: true, breakLine: false } },
    { text: "     VALIDATION  Jun — threshold selection only", options: { color: TEAL, bold: true, breakLine: false } },
    { text: "     TEST  Jul–Aug — touched once", options: { color: CORAL, bold: true } },
  ], { x: MX, y: 3.85, w: CW, h: 0.35, fontFace: BODY, fontSize: 11.5, isTextBox: true, margin: 0 });
  bullets(s, [
    "No shuffling — a random split lets August behaviour inform an October prediction; that inflates the score and hides model decay",
    "Rejected alternative: account-grouped k-fold — removes account leakage but still mixes time periods",
    "Encoders and entity features are fit on the training window only",
    "Entity features are computed over the full timeline before splitting, so val / test rows carry real history — exactly as at inference",
  ], { y: 4.5, fontSize: 13 });
  note(s, "Random split = leakage + hides decay. Account-grouped k-fold still mixes time. Validation = June, threshold only. Test = Jul/Aug, touched once.");
}

/* -------------------------------------------------------- 9. Metric choice */
{
  const s = lightSlide(1, "Evaluation metric");
  title(s, "Choosing the metric — and the reasoning");
  bullets(s, [
    "Accuracy / F1 rejected: a fixed 0.5 threshold is meaningless when we alert on the top 1%",
    "ROC-AUC rejected as the headline: dominated by millions of easy negatives — ~0.98 even for a weak model",
    "PR-AUC (average precision): threshold-free, and only rewards precision on the positives — what an alert list lives on",
    "Alongside it, threshold-specific numbers: precision, recall, and alerts-per-day at a chosen budget",
  ], { w: 7.0, y: 2.15, fontSize: 13.5 });
  s.addTable(
    [
      [{ text: "", options: { fill: { color: WHITE } } }, { text: "PR-AUC", options: { bold: true, color: NAVY } }, { text: "ROC-AUC", options: { bold: true, color: MUTE } }],
      ["Validation (Jun)", "0.68", "0.99"],
      [{ text: "Test (Jul–Aug)", options: { bold: true } }, { text: "0.54", options: { bold: true, color: NAVY } }, "0.98"],
    ],
    { x: 8.0, y: 2.3, w: 4.7, colW: [2.1, 1.3, 1.3], rowH: 0.6, fontFace: BODY, fontSize: 13, valign: "middle", border: { type: "solid", color: LINE, pt: 1 } }
  );
  s.addText("0.68 → 0.54 in two months = real model decay → Part 2", {
    x: 8.0, y: 4.35, w: 4.7, h: 0.8, fontFace: BODY, italic: true, fontSize: 12.5, color: INK, isTextBox: true, margin: 0,
  });
  note(s, "ROC-AUC 0.98 and meaningless with 1000:1. F1/accuracy need a 0.5 cut we never use. PR-AUC is the honest summary. The val->test drop bridges to Part 2.");
}

/* ---------------------------------------------------- 10. The trade-off */
{
  const s = lightSlide(1, "Detection vs. alerts");
  title(s, "Detecting suspicious activity vs. unnecessary alerts");
  s.addChart(
    p.ChartType.line,
    [{ name: "recall", labels: ["29", "58", "144", "289", "405", "577"], values: [0.49, 0.59, 0.67, 0.74, 0.77, 0.80] }],
    {
      x: MX, y: 2.15, w: 6.8, h: 4.4,
      chartColors: [TEAL], lineDataSymbol: "circle", lineSize: 3, showLegend: false, showTitle: false,
      valAxisMinVal: 0, valAxisMaxVal: 1, valAxisTitle: "recall", showValAxisTitle: true, valAxisLabelColor: MUTE, valAxisLabelFontSize: 10, valGridLine: { color: LINE, size: 1 },
      catAxisTitle: "alerts / day", showCatAxisTitle: true, catAxisLabelColor: MUTE, catAxisLabelFontSize: 10, catGridLine: { style: "none" },
    }
  );
  card(s, 7.6, 2.25, CW - 7.0, 2.35, "F4F7FC");
  cardHead(s, 7.6, 2.25, CW - 7.0, "Chosen operating point");
  s.addText("~405 alerts/day  ·  precision 6.4%  ·  recall 77%", { x: 7.9, y: 2.85, w: 5.0, h: 0.5, fontFace: BODY, fontSize: 12.5, color: INK, isTextBox: true, margin: 0 });
  s.addText("threshold fixed on validation at the 1% budget, applied to test", { x: 7.9, y: 3.3, w: 5.0, h: 0.9, fontFace: BODY, fontSize: 11, italic: true, color: MUTE, isTextBox: true, margin: 0 });
  bullets(s, [
    "The point on this curve is a capacity + risk-appetite decision for the investigations team — not a modelling choice",
    "Cost framing: regulator sets a minimum recall; capacity sets the budget; model quality decides if both can be met",
    "Alert rate crept 1.0% → 1.4% as scores drifted — fixed threshold, moving target",
  ], { x: 7.6, y: 4.8, w: CW - 7.0, fontSize: 11.5 });
  note(s, "~400 alerts/day, ~22 real, catches 3/4 of laundering. Tighten -> precision up, recall down. My job is to hand them the curve, not pick the point.");
}

/* ------------------------------------------------ 11. Typology + limits */
{
  const s = lightSlide(1, "Error analysis");
  title(s, "Where recall comes from — and honest limits");
  s.addChart(
    p.ChartType.bar,
    [{ name: "recall @ 1% budget", labels: ["Smurfing", "Cash Withdrawal", "Structuring", "Cycle", "Deposit-Send", "Bipartite", "Layered Fan-Out", "Fan-Out"], values: [1.0, 0.97, 0.79, 0.71, 0.70, 0.65, 0.63, 0.60] }],
    {
      x: MX, y: 2.1, w: 6.6, h: 4.6, barDir: "bar",
      chartColors: [NAVY], showLegend: false, showTitle: false,
      showValue: true, dataLabelPosition: "outEnd", dataLabelColor: INK, dataLabelFontSize: 10, dataLabelFormatCode: "0%",
      valAxisHidden: true, valAxisMaxVal: 1.15, valGridLine: { style: "none" },
      catAxisLabelColor: INK, catAxisLabelFontSize: 10, catGridLine: { style: "none" },
    }
  );
  s.addText("Strong where behaviour is the signal.  Weak on graph-structure patterns — no graph features yet.", {
    x: 7.5, y: 2.1, w: CW - 6.9, h: 1.2, fontFace: BODY, fontSize: 13, color: INK, isTextBox: true, margin: 0,
  });
  bullets(s, [
    "No graph features → weak on fan-out / bipartite (top of the backlog)",
    "Model decays measurably over ~2 months",
    "Synthetic, instant labels — real SAR outcomes lag by months",
    "Single global threshold — a per-corridor threshold would likely lift recall",
  ], { x: 7.5, y: 3.4, w: CW - 6.9, fontSize: 12 });
  note(s, "Per-typology recall tells investigators what to trust. Strong on structuring/smurfing, weak on fan-out (graph structure). Graph features are top of future work.");
}

/* ------------------------------------------------------- 12. Part 2 overview */
{
  const s = lightSlide(2, "ML lifecycle & MLOps");
  title(s, "Operating the model — the lifecycle loop");
  const stages = ["daily\nbatch", "validate", "features", "score", "alerts", "MONITOR", "retrain\n(rolling)", "challenger\n+ shadow", "approve +\npromote"];
  const n = stages.length, gap = 0.12, bw = (CW - gap * (n - 1)) / n, by = 2.15, bh = 1.15;
  stages.forEach((t, i) => {
    const hot = t === "MONITOR";
    s.addShape(p.ShapeType.roundRect, { x: MX + i * (bw + gap), y: by, w: bw, h: bh, rectRadius: 0.05, fill: { color: hot ? TEAL : NAVY } });
    s.addText(t, { x: MX + i * (bw + gap), y: by, w: bw, h: bh, align: "center", valign: "middle", fontFace: BODY, fontSize: 8.5, bold: hot, color: WHITE, isTextBox: true, margin: 0 });
  });
  s.addText("Part 1 showed the model decays:  PR-AUC 0.68 → 0.54,  alert rate 1.0% → 1.4%,  in two months.", {
    x: MX, y: 3.6, w: CW, h: 0.4, fontFace: BODY, fontSize: 13, italic: true, color: INK, isTextBox: true, margin: 0,
  });
  card(s, MX, 4.2, 6.0, 1.9, "F4F7FC");
  cardHead(s, MX, 4.2, 6.0, "Demonstrated (built, run on Jul–Aug)");
  s.addText("monitoring on every batch  ·  a codified retraining trigger  ·  rolling-window retrain + challenger + gated promotion", { x: MX + 0.3, y: 4.75, w: 5.4, h: 1.2, fontFace: BODY, fontSize: 12, color: INK, isTextBox: true, margin: 0 });
  card(s, MX + 6.3, 4.2, CW - 6.3, 1.9, WHITE);
  cardHead(s, MX + 6.3, 4.2, CW - 6.3, "Conceptual", MUTE);
  s.addText("registry mechanics (MLflow @champion/@challenger aliases)  ·  shadow deployment  ·  delayed & biased labels + reservoir sampling  ·  feature-store parity  ·  governance & audit  ·  incident response", { x: MX + 6.6, y: 4.75, w: 5.0, h: 1.2, fontFace: BODY, fontSize: 11, color: INK, isTextBox: true, margin: 0 });
  note(s, "The model decays — slides 9-10 proved it. Lifecycle runs on proxy signals because true labels (SARs) lag months. Demonstrated: monitoring + trigger + challenger promotion, run on the Jul-Aug batches.");
}

/* ------------------------------------------------- 13. Part 2 monitoring */
{
  const s = lightSlide(2, "ML lifecycle & MLOps");
  title(s, "Monitoring every batch — and what it caught");
  s.addTable(
    [
      [{ text: "Layer", options: { bold: true, color: WHITE, fill: { color: NAVY } } }, { text: "Signal", options: { bold: true, color: WHITE, fill: { color: NAVY } } }, { text: "On the Jul–Aug replay", options: { bold: true, color: WHITE, fill: { color: NAVY } } }],
      ["Data quality", "volume vs. norm · nulls · unseen categories · schema", "flagged the final partial batch (8k vs 30k rows) → metrics gated"],
      ["Input drift", "PSI per feature vs. the training snapshot", "stable (max ≈ 0.13) — except two unbounded counter features"],
      ["Prediction drift", "score distribution · alert rate at the fixed threshold", "alert rate drifts 1.2% → 1.7%, breaches the band from Aug 1"],
      ["Delayed performance", "precision / recall once labels arrive (weeks–months)", "precision 7.4% → 5.5%,  recall 80% → 72%"],
    ],
    { x: MX, y: 2.1, w: CW, colW: [2.1, 4.5, 5.5], rowH: 0.7, fontFace: BODY, fontSize: 10.5, color: INK, valign: "middle", border: { type: "solid", color: LINE, pt: 1 } }
  );
  card(s, MX, 5.5, CW, 1.6, "F4F7FC");
  cardHead(s, MX, 5.5, CW, "The input-drift alarm surfaced a feature-design issue");
  s.addText("sender / receiver_prior_txn_count are unbounded cumulative counters — PSI climbs to 1.4 as calendar time passes, regardless of behaviour. Tagged as known drift, excluded from the alarm, backlogged as a Part 1 fix (cap or window them). Monitoring doing its job.", {
    x: MX + 0.3, y: 6.05, w: CW - 0.6, h: 0.9, fontFace: BODY, fontSize: 10.5, color: INK, isTextBox: true, margin: 0, lineSpacingMultiple: 1.05,
  });
  note(s, "Cheapest signal first, compared to a reference snapshot versioned with the model. Two findings: real prediction drift (alert-rate breach), and a feature bug (unbounded counters). The second is what you deploy monitoring to catch.");
}

/* ------------------------------------------- 14. Part 2 retrain + promote */
{
  const s = lightSlide(2, "ML lifecycle & MLOps");
  title(s, "Trigger → retrain → challenger → gated promotion");
  bullets(s, [
    "Trigger fired 3 Aug — reasons: scheduled monthly floor + alert-rate drift for 3 consecutive days",
    "Retrained on a rolling 8-month window (Dec 2022 – Jul 2023) — track current behaviour, not all history",
    "New model is a challenger: judged on the most recent labelled slice (August), never promoted automatically",
  ], { w: 7.0, y: 2.15, fontSize: 13 });
  s.addTable(
    [
      [{ text: "August", options: { bold: true, color: WHITE, fill: { color: NAVY } } }, { text: "champion", options: { bold: true, color: MUTE } }, { text: "challenger", options: { bold: true, color: NAVY } }],
      ["PR-AUC", "0.45", "0.69"],
      ["recall @ 1% budget", "0.66", "0.82"],
    ],
    { x: 8.0, y: 2.3, w: 4.7, colW: [2.3, 1.2, 1.2], rowH: 0.6, fontFace: BODY, fontSize: 12.5, color: INK, valign: "middle", border: { type: "solid", color: LINE, pt: 1 } }
  );
  card(s, MX, 4.7, CW, 1.9, "F4F7FC");
  cardHead(s, MX, 4.7, CW, "Promotion is a registry operation, not a deploy");
  s.addText("challenger beats champion → registered, alias set to @challenger → awaits model-owner + compliance sign-off → takes @champion.\nScoring only ever asks for @champion, so promotion and rollback are one alias move. Previous version stays loadable.", {
    x: MX + 0.3, y: 5.25, w: CW - 0.6, h: 1.2, fontFace: BODY, fontSize: 11.5, color: INK, isTextBox: true, margin: 0, lineSpacingMultiple: 1.1,
  });
  note(s, "The retrain recovers the lost performance — PR-AUC back to 0.69 on August. But a human signs off before it goes live: regulated control. Rollback is symmetric — move the alias back.");
}

/* ------------------------------------------------------- 14. Part 3 architecture */
{
  const s = lightSlide(3, "Production architecture");
  title(s, "A daily batch pipeline on AWS");
  const steps = ["S3 raw\n(immutable)", "Validate\nGE / Deequ", "Features\nGlue / EMR", "Score\nSageMaker", "Rank + alert\nDynamoDB / SQS", "Monitor\nEvidently"];
  const n = steps.length, gap = 0.2, bw = (CW - gap * (n - 1)) / n, by = 2.4, bh = 1.25;
  steps.forEach((t, i) => {
    s.addShape(p.ShapeType.roundRect, { x: MX + i * (bw + gap), y: by, w: bw, h: bh, rectRadius: 0.05, fill: { color: NAVY } });
    s.addText(t, { x: MX + i * (bw + gap), y: by, w: bw, h: bh, align: "center", valign: "middle", fontFace: BODY, fontSize: 9.5, color: WHITE, isTextBox: true, margin: 0 });
    if (i < n - 1) s.addText("→", { x: MX + i * (bw + gap) + bw - 0.04, y: by, w: gap + 0.08, h: bh, align: "center", valign: "middle", fontFace: BODY, fontSize: 13, color: MUTE, isTextBox: true, margin: 0 });
  });
  s.addText("Orchestration: Step Functions (daily) + a second state machine for retrain → shadow → promote        ·        MLflow registry on Fargate        ·        Terraform", {
    x: MX, y: 4.0, w: CW, h: 0.4, fontFace: BODY, fontSize: 11, color: MUTE, isTextBox: true, margin: 0,
  });
  bullets(s, [
    "Batch, not streaming — the label and the workflow are both post-event; there is no real-time decision. Add streaming only for a specific high-risk corridor if the business needs faster interdiction.",
    "The MLflow registry is the single control point — scoring loads \"the Production model\", never a path. Rollback = revert the stage and re-score.",
    "Raw zone is immutable and date-partitioned → any past batch can be re-scored with any model version.",
  ], { y: 4.6, fontSize: 12.5 });
  note(s, "Talk to the flow left to right. Validation upstream of scoring — a broken feed is the #1 cause of 'model failure'. Registry is the control point. Immutable raw = reproducibility.");
}

/* ------------------------------------------------------- 15. Part 3 trade-offs */
{
  const s = lightSlide(3, "Production architecture");
  title(s, "Technology choices — the trade-offs");
  s.addTable(
    [
      [{ text: "Decision", options: { bold: true, color: WHITE, fill: { color: NAVY } } }, { text: "Choice", options: { bold: true, color: WHITE, fill: { color: NAVY } } }, { text: "Alternative", options: { bold: true, color: WHITE, fill: { color: NAVY } } }, { text: "Why this way", options: { bold: true, color: WHITE, fill: { color: NAVY } } }],
      ["Serving", "Daily batch", "Streaming", "Post-event problem; batch is simpler, no train/serve skew"],
      ["Compute", "SageMaker", "EKS + Kubeflow", "Managed, minimal ops for a small team"],
      ["Registry", "MLflow (self-hosted)", "SageMaker Model Registry", "One tool for tracking + registry; portable; team standard"],
      ["Feature compute", "Glue / EMR Serverless", "Pandas on a big box", "Scales past 10x volume; serverless"],
      ["Feature storage", "S3 table + shared code", "SageMaker Feature Store", "Batch-only — parity without the infrastructure"],
      ["Case management", "Buy", "Build a UI", "Not the differentiation; fits investigator workflow"],
    ],
    { x: MX, y: 2.15, w: CW, colW: [2.2, 2.7, 2.7, 4.5], rowH: 0.6, fontFace: BODY, fontSize: 10.5, color: INK, valign: "middle", border: { type: "solid", color: LINE, pt: 1 } }
  );
  note(s, "Every choice biased toward less operational burden for a small team. Name the alternative and the condition under which I'd switch (EKS if the org already runs k8s; Feature Store if a real-time path appears).");
}

/* ------------------------------------------------------- 16. Part 4 delivery */
{
  const s = lightSlide(4, "Delivery");
  title(s, "PoC → production over 3 / 6 / 9 months");
  s.addText("Principle: reliability → automation → sophistication. It's a regulated control — the risks are operational, not modelling. Model never hard-replaces the rules engine; shadow → parallel → staged retirement.", {
    x: MX, y: 1.95, w: CW, h: 0.7, fontFace: BODY, fontSize: 12, italic: true, color: MUTE, isTextBox: true, margin: 0,
  });
  const cols = [
    ["0–3 mo — make it trustworthy", ["Reliable pipeline + data validation", "One model, fully monitored", "SHADOW mode — action nothing", "Baselines agreed with compliance"], "Can't operate or get sign-off for a control you can't see."],
    ["3–6 mo — make it self-sustaining", ["Automated retraining + challengers", "Investigator feedback loop for labels", "Reservoir sampling for unbiased recall", "Run in PARALLEL with rules"], "Label lag is months — the feedback loop must start early."],
    ["6–9 mo — make it better + auditable", ["Graph features / typology models", "Per-corridor thresholds", "Full model cards, annual validation", "Staged rules retirement"], "Lowest-risk work — only after the foundations hold."],
  ];
  const cw2 = (CW - 0.6) / 3;
  cols.forEach(([h, items, why], i) => {
    const x = MX + i * (cw2 + 0.3);
    card(s, x, 2.8, cw2, 3.7, i === 0 ? "F4F7FC" : WHITE);
    s.addText(h, { x: x + 0.25, y: 3.0, w: cw2 - 0.5, h: 0.7, fontFace: BODY, bold: true, fontSize: 12.5, color: NAVY, isTextBox: true, margin: 0 });
    s.addText(items.map((t, j) => ({ text: t, options: { bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 6 } })).concat([{ text: why, options: { italic: true, color: MUTE, paraSpaceBefore: 6 } }]), {
      x: x + 0.25, y: 3.5, w: cw2 - 0.5, h: 3.0, fontFace: BODY, fontSize: 10.5, color: INK, isTextBox: true, margin: 0,
    });
  });
  note(s, "Reliability first, sophistication last — the opposite of the instinct. Shadow before parallel before retirement. Each stage has an exit criterion (SLA met 4 weeks; 2 automated retrains; etc.).");
}

/* ---- 17. Summary */
{
  const s = darkSlide();
  s.addShape(p.ShapeType.ellipse, { x: -2, y: 4.4, w: 5.5, h: 5.5, fill: { color: INK } });
  s.addText("What I'd want you to take away", { x: MX, y: 0.8, w: CW, h: 0.8, fontFace: HEAD, fontSize: 30, bold: true, color: WHITE, isTextBox: true, margin: 0 });
  s.addText([
    { text: "Framed as alerting, not classification — recall at a fixed workload", options: { bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 10, color: ICE } },
    { text: "Causal features, temporal split, imbalance as a threshold problem — each choice tested against an alternative, not asserted", options: { bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 10, color: ICE } },
    { text: "Model decays measurably in 2 months — Part 2 makes that operable", options: { bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 10, color: ICE } },
    { text: "Production seams already in place: package, ADRs, MLflow, registry", options: { bullet: { code: "2022" }, color: ICE } },
  ], { x: MX, y: 1.9, w: 8.3, h: 3.4, fontFace: BODY, fontSize: 15, isTextBox: true, margin: 0, lineSpacingMultiple: 1.1 });
  s.addText("77%", { x: 8.9, y: 5.0, w: 3.8, h: 1.1, align: "center", fontFace: HEAD, fontSize: 54, bold: true, color: TEAL, isTextBox: true, margin: 0 });
  s.addText("of laundering caught\nat ~400 alerts / day", { x: 8.9, y: 6.1, w: 3.8, h: 0.8, align: "center", fontFace: BODY, fontSize: 12, color: ICE, isTextBox: true, margin: 0 });
  note(s, "Treated it as alerting, made every choice with investigator workload in mind, tested each against an alternative. 77% at 400 alerts/day, and it decays — which is what Part 2 is for.");
}

p.writeFile({ fileName: "AML-Transaction-Monitoring.pptx" }).then((f) => console.log("wrote", f));
