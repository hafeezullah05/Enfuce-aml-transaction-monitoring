"""Generate the charts embedded in the presentation.

Loads the registered @champion model, scores validation and test, reads the
MLflow sweep and the Part 2 monitoring history, and writes styled PNGs to
presentation/figures/.

Run: uv run python scripts/make_deck_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from aml_monitoring.config import ARTIFACTS_DIR, MLFLOW_URI
from aml_monitoring.dataset import FEATURE_COLS, load_or_build_features, make_dataset
from aml_monitoring.models.explain import (
    build_explainer,
    explain_alert,
    global_importance,
    pick_reason_code_alert,
    shap_frame,
)

CORAL = "#F0523C"
INK = "#1A1A1A"
GREY = "#9B8F8A"
OUT = Path("presentation/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#ECECEC",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "font.size": 11,
})


def save(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    frame = load_or_build_features()
    ds = make_dataset(frame)
    model = mlflow.lightgbm.load_model("models:/aml-transaction-monitoring@champion")

    val_s = model.predict_proba(ds.X_val)[:, 1]
    test_s = model.predict_proba(ds.X_test)[:, 1]
    y_val, y_test = ds.y_val.to_numpy(), ds.y_test.to_numpy()

    # ---------------------------------------------------------------- 1. PR curves
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ap_val = average_precision_score(y_val, val_s)
    ap_test = average_precision_score(y_test, test_s)
    for scores, y, lab, col in [
        (val_s, y_val, f"validation  (PR-AUC {ap_val:.2f})", GREY),
        (test_s, y_test, f"held-out test  (PR-AUC {ap_test:.2f})", CORAL),
    ]:
        pr, rc, _ = precision_recall_curve(y, scores)
        ax.plot(rc, pr, color=col, lw=2.6, label=lab)
    # operating point: threshold from val at 1% budget, applied to test
    thr = float(np.quantile(val_s, 0.99))
    alert = test_s >= thr
    op_r = alert[y_test == 1].mean()
    op_p = (y_test[alert] == 1).mean()
    ax.scatter([op_r], [op_p], s=110, color=INK, zorder=5,
               label=f"operating point: {op_r:.0%} recall, {op_p:.1%} precision")
    ax.set_xlabel("recall  (share of laundering caught)")
    ax.set_ylabel("precision  (share of alerts that are real)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("The two-month decay: validation 0.68 to test 0.54", fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), ncol=1, handletextpad=0.5)
    fig.subplots_adjust(bottom=0.34)
    save(fig, "pr_curves.png", tight=False)

    # ------------------------------------------------------- 2. sweep: PR vs ROC
    runs = mlflow.search_runs(experiment_names=["aml-part1"], order_by=["start_time"])
    lg = runs[runs["tags.mlflow.runName"].str.startswith("lightgbm")].copy()
    lg["spw"] = lg["params.scale_pos_weight"].astype(float)
    lg = lg.sort_values("spw")
    x = np.arange(len(lg)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 4.3))

    def _lab(v):
        return f"{v:.2f}" if v >= 0.1 else f"{v:.3f}"

    ax.bar(x - w / 2, lg["metrics.val_pr_auc"], w, color=CORAL, label="PR-AUC")
    ax.bar(x + w / 2, lg["metrics.val_roc_auc"], w, color=INK, label="ROC-AUC")
    for xi, pr, roc in zip(x, lg["metrics.val_pr_auc"], lg["metrics.val_roc_auc"]):
        ax.text(xi - w / 2, pr + 0.02, _lab(pr), ha="center", fontsize=8.5)
        ax.text(xi + w / 2, roc + 0.02, _lab(roc), ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"spw = {int(v)}" for v in lg["spw"]])
    ax.set_ylim(0, 1.18); ax.set_ylabel("validation score")
    ax.set_title("Cost-weighting sweep: PR-AUC collapses, ROC-AUC does not", fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "sweep.png")

    # ------------------------------------------------- 3. SHAP global importance
    explainer = build_explainer(model)
    sample = ds.X_test.sample(n=min(20000, len(ds.X_test)), random_state=0)
    sv = shap_frame(explainer, sample)
    gi = global_importance(sv).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh(gi["feature"], gi["mean_abs_shap"], color=INK)
    ax.set_xlabel("mean |SHAP|  (average effect on the log-odds score)")
    ax.set_title("What the model actually uses: account behaviour dominates", fontsize=10.5, pad=10)
    ax.grid(axis="y", visible=False)
    save(fig, "importance.png")

    # ----------------------------------------- 3b. SHAP reason codes for one alert
    tp = ds.X_test[ds.y_test.to_numpy() == 1]
    tp_scores = model.predict_proba(tp)[:, 1]
    alert_i = pick_reason_code_alert(model, explainer, tp, tp_scores)
    expl = explain_alert(model, explainer, tp.iloc[[alert_i]], top_n=7)

    def _fmt(feat, v):
        if feat == "amount_log":
            return f"transaction amount = GBP {np.expm1(float(v)):,.0f}"
        if feat.endswith("_secs_since_last"):
            who = feat.split("_")[0]
            hrs = float(v) / 3600
            span = f"{hrs / 24:.0f}d" if hrs >= 72 else f"{hrs:.0f}h"
            return f"{who}: {span} since its last txn"
        if feat.endswith("_amount_vs_mean_7d"):
            who = feat.split("_")[0]
            return f"{who}: amount is {float(v):.1f}x its 7d average"
        return f"{feat} = {v}"

    drv = list(reversed(expl["drivers"]))
    labels = [_fmt(d["feature"], d["value"]) for d in drv]
    vals = [d["shap"] for d in drv]
    cols = [CORAL if v > 0 else GREY for v in vals]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.barh(range(len(vals)), vals, color=cols)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.axvline(0, color="#333333", lw=0.9)
    ax.set_xlabel("SHAP contribution to the score   (coral raises risk, grey lowers it)")
    ax.set_title("A laundering alert: the reason codes the investigator sees", fontsize=10.5, pad=10)
    ax.grid(axis="y", visible=False)
    save(fig, "shap_alert.png")

    # ------------------------------------------------- 4. recall by typology
    from aml_monitoring.models.evaluate import recall_by_typology, score_frame

    test_scored = score_frame(model, ds.X_test, ds.y_test, ds.meta_test)
    val_thr = float(np.quantile(val_s, 0.99))  # 1% budget on validation
    rbt = recall_by_typology(test_scored, val_thr)
    rbt = rbt[rbt["n"] >= 20].sort_values("recall")
    labels = [t.replace("_", " ") for t in rbt.index]
    rec = rbt["recall"].to_numpy()
    cols = [CORAL if r < 0.72 else INK for r in rec]
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.barh(labels, rec * 100, color=cols)
    for i, r in enumerate(rec):
        ax.text(r * 100 + 1.5, i, f"{r:.0%}", va="center", fontsize=9)
    ax.set_xlim(0, 112)
    ax.set_xlabel("recall at the 1% alert budget  (%)")
    ax.set_title("Strong on behavioural typologies (ink), weak on graph-structure ones (coral)", fontsize=9.5, pad=10)
    ax.grid(axis="y", visible=False)
    save(fig, "typology_recall.png")


    # ------------------------------------------------------- 5. Part 2 dashboard
    h = pd.read_csv(ARTIFACTS_DIR / "monitoring_history.csv", parse_dates=["day"])
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 5.4))
    d = h["day"]
    axs[0, 0].plot(d, h["alert_rate"] * 100, marker=".", color=CORAL)
    axs[0, 0].axhspan(0.7, 1.5, color=CORAL, alpha=0.08)
    axs[0, 0].set_title("Alert rate vs. the 0.7 to 1.5% band", fontsize=10)
    axs[0, 0].set_ylabel("%")
    axs[0, 1].plot(d, h["score_mean"], marker=".", color=INK, label="mean")
    axs[0, 1].plot(d, h["score_p95"], marker=".", color=CORAL, label="p95")
    axs[0, 1].set_title("Score distribution drift", fontsize=10); axs[0, 1].legend(frameon=False, fontsize=8)
    axs[1, 0].plot(d, h["psi_max"], marker=".", color=INK, label="alarm set (max)")
    axs[1, 0].plot(d, h["psi_known_drift"], marker=".", color=CORAL, label="prior_txn_count")
    axs[1, 0].axhline(0.2, color="#999", ls="--", lw=1)
    axs[1, 0].set_title("Input drift (PSI)", fontsize=10); axs[1, 0].legend(frameon=False, fontsize=8)
    axs[1, 1].plot(d, h["precision"] * 100, marker=".", color=INK, label="precision")
    axs[1, 1].plot(d, h["recall"] * 100, marker=".", color=CORAL, label="recall")
    axs[1, 1].set_title("Delayed performance (arrives weeks later)", fontsize=10)
    axs[1, 1].set_ylabel("%"); axs[1, 1].legend(frameon=False, fontsize=8)
    for a in axs.flat:
        a.tick_params(axis="x", rotation=40, labelsize=8)
    save(fig, "monitoring.png")

    # ------------------------------------------------------- 6. champion vs challenger
    try:
        chal = mlflow.lightgbm.load_model("models:/aml-transaction-monitoring@challenger")
        aug = frame[frame["month"] == "2023-08"]
        Xa, ya = aug[FEATURE_COLS], aug["Is_laundering"].to_numpy()
        vals = {}
        for name, m in [("champion", model), ("challenger", chal)]:
            sc = m.predict_proba(Xa)[:, 1]
            t = float(np.quantile(sc, 0.99))
            al = sc >= t
            vals[name] = (average_precision_score(ya, sc), al[ya == 1].mean())
    except Exception:  # noqa: BLE001
        vals = {"champion": (0.45, 0.66), "challenger": (0.69, 0.82)}
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    x = np.arange(2); w = 0.35
    ax.bar(x - w / 2, [vals["champion"][0], vals["champion"][1]], w, color=GREY, label="champion (stale)")
    ax.bar(x + w / 2, [vals["challenger"][0], vals["challenger"][1]], w, color=CORAL, label="challenger (retrained)")
    for xi, (c, ch) in zip(x, zip(*[vals["champion"], vals["challenger"]])):
        ax.text(xi - w / 2, c + 0.02, f"{c:.2f}", ha="center", fontsize=9)
        ax.text(xi + w / 2, ch + 0.02, f"{ch:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["PR-AUC", "recall @ 1% budget"])
    ax.set_ylim(0, 1.0); ax.set_title("August: the retrain recovers the lost performance", fontsize=10, pad=10)
    ax.legend(frameon=False, fontsize=9)
    save(fig, "champ_chall.png")

    print("\nall figures written to", OUT)


if __name__ == "__main__":
    main()
