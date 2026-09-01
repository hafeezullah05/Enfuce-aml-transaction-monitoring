"""Part 2 demo — lifecycle management of the deployed model.

  1. rebuild the deployment-time reference snapshot
  2. replay Jul + Aug 2023 as daily production batches, monitoring each
  3. evaluate the retraining trigger against the accumulated history
  4. retrain a challenger on a rolling window, compare it to the champion on the
     most recent labelled data, and register it if it wins

Run:  uv run python scripts/run_part2.py
Writes: artifacts/monitoring_history.csv, artifacts/reference.json
"""

from __future__ import annotations

import mlflow
import pandas as pd

from aml_monitoring.config import (
    ALL_MONTHS,
    ARTIFACTS_DIR,
    MLFLOW_URI,
    TRAIN_MONTHS,
    VAL_MONTHS,
)
from aml_monitoring.dataset import load_or_build_features, make_dataset
from aml_monitoring.lifecycle import (
    compare_on_labelled,
    fit_challenger,
    rolling_window,
    should_retrain,
)
from aml_monitoring.monitoring.batch import check_batch, score_batch
from aml_monitoring.monitoring.reference import build_reference

REGISTERED_NAME = "aml-transaction-monitoring"
CHAMPION_URI = f"models:/{REGISTERED_NAME}@champion"  # alias, not a version number
BATCH_MONTHS = ["2023-07", "2023-08"]


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    frame = load_or_build_features()
    champion = mlflow.lightgbm.load_model(CHAMPION_URI)

    # ---- 1. reference snapshot (built from what the model was trained/tuned on) ----
    train_frame = frame[frame["month"].isin(TRAIN_MONTHS)]
    val_frame = frame[frame["month"].isin(VAL_MONTHS)]
    ds = make_dataset(frame)
    val_scores = champion.predict_proba(ds.X_val)[:, 1]
    threshold = float(pd.Series(val_scores).quantile(0.99))  # 1% budget on validation

    ref = build_reference(train_frame, val_frame, val_scores, threshold, model_version="1")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ref.save(ARTIFACTS_DIR / "reference.json")
    print(f"reference: threshold={threshold:.4f}  expected {ref.rows_per_day_mean:.0f} rows/day")

    # ---- 2. replay daily batches ----
    prod = frame[frame["month"].isin(BATCH_MONTHS)].copy()
    prod["day"] = pd.to_datetime(prod["timestamp"]).dt.date.astype(str)

    records = []
    for day, batch in prod.groupby("day", sort=True):
        scores = score_batch(champion, batch)
        records.append(check_batch(batch, scores, ref, day))
    history = pd.DataFrame(records)
    history.to_csv(ARTIFACTS_DIR / "monitoring_history.csv", index=False)

    jul = history[history["day"].str.startswith("2023-07")]
    aug = history[history["day"].str.startswith("2023-08")]
    print(
        f"\nJuly : alert_rate {jul.alert_rate.mean():.2%}  psi_max {jul.psi_max.max():.2f}  "
        f"precision {jul.precision.mean():.2%}  recall {jul.recall.mean():.2%}"
    )
    print(
        f"Aug  : alert_rate {aug.alert_rate.mean():.2%}  psi_max {aug.psi_max.max():.2f}  "
        f"precision {aug.precision.mean():.2%}  recall {aug.recall.mean():.2%}"
    )
    alarmed = history[history["alarm"] != ""]
    print(f"days with an alarm: {len(alarmed)} / {len(history)}")
    if len(alarmed):
        print(f"  first alarm: {alarmed.iloc[0]['day']}  ({alarmed.iloc[0]['alarm']})")
    print(f"  known structural drift (prior_txn_count) PSI: {history['psi_known_drift'].max():.2f}  [excluded from alarm]")

    # ---- 3. retraining trigger ----
    baseline_precision = float(jul["precision"].iloc[:5].mean())
    decision = should_retrain(history, baseline_precision=baseline_precision, days_since_training=60)
    print(f"\nretrain?  {decision.retrain}  reasons={decision.reasons}  first triggered: {decision.trigger_day}")

    if not decision.retrain:
        print("no retrain triggered — stopping")
        return

    # ---- 4. challenger on a rolling window through July, judged on August ----
    months = rolling_window(ALL_MONTHS, through="2023-07", width=8)
    print(f"challenger training window: {months[0]} .. {months[-1]}")
    ch_ds = make_dataset(frame, train_months=months, val_months=["2023-08"], test_months=["2023-08"])
    challenger = fit_challenger(ch_ds)

    labelled_aug = frame[frame["month"] == "2023-08"]
    cmp = compare_on_labelled(champion, challenger, labelled_aug)
    print("\nchampion  vs  challenger  on August:")
    print(f"  PR-AUC          {cmp['champion_pr_auc']:.3f}   {cmp['challenger_pr_auc']:.3f}")
    print(f"  recall@1%budget {cmp['champion_recall_at_budget']:.3f}   {cmp['challenger_recall_at_budget']:.3f}")
    print(f"  challenger wins: {cmp['challenger_wins']}")

    if cmp["challenger_wins"]:
        with mlflow.start_run(run_name="challenger-through-2023-07"):
            mlflow.log_params({"training_window": f"{months[0]}..{months[-1]}", "scale_pos_weight": 1.0})
            mlflow.log_metrics({k: v for k, v in cmp.items() if isinstance(v, float)})
            mlflow.lightgbm.log_model(challenger, name="model")
            run_id = mlflow.active_run().info.run_id
        mv = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_NAME)
        mlflow.MlflowClient().set_registered_model_alias(REGISTERED_NAME, "challenger", mv.version)
        print(
            f"\nregistered v{mv.version} and set alias @challenger "
            f"(awaiting human promotion to @champion; current champion kept for rollback)"
        )


if __name__ == "__main__":
    main()
