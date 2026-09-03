"""Freeze every headline result into ``results/`` as plain files.

The notebooks and MLflow are the source of truth; this script pins the numbers
that appear in the presentation into a small, reviewable folder that goes in the
repo (``artifacts/`` and ``mlflow.db`` are git-ignored and regenerated).

Produces, under ``results/``:
  part1/ranking_metrics.json          PR-AUC / ROC-AUC on validation and test
  part1/operating_points_val.csv      precision / recall / alerts-per-day by budget
  part1/operating_points_test.csv
  part1/recall_by_typology.csv        where recall comes from, at the 1% budget
  part1/sweep_results.csv             the scale_pos_weight sweep
  part1/shap_global_importance.csv    mean |SHAP| per feature
  part1/shap_example_alert.json       reason codes for one laundering alert
  mlflow/experiment_runs.csv          every tracked run in aml-part1
  mlflow/registry.csv                 registered versions + aliases
  part2/monitoring_history.csv        the daily monitoring replay
  part2/champion_vs_challenger.json   the retrain comparison on August

Run:  uv run python scripts/export_results.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from aml_monitoring.config import ALERT_BUDGET_PCT, ARTIFACTS_DIR, MLFLOW_URI, REPO_ROOT
from aml_monitoring.dataset import load_or_build_features, make_dataset
from aml_monitoring.models.evaluate import (
    operating_points,
    ranking_metrics,
    recall_by_typology,
    score_frame,
    threshold_for_budget,
)
from aml_monitoring.models.explain import (
    build_explainer,
    explain_alert,
    global_importance,
    pick_reason_code_alert,
    shap_frame,
)

RESULTS = REPO_ROOT / "results"
REGISTERED_NAME = "aml-transaction-monitoring"


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))
    print("wrote", path.relative_to(REPO_ROOT))


def _write_csv(path: Path, df: pd.DataFrame, index: bool = False) -> None:
    df.to_csv(path, index=index)
    print("wrote", path.relative_to(REPO_ROOT))


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    (RESULTS / "part1").mkdir(parents=True, exist_ok=True)
    (RESULTS / "part2").mkdir(parents=True, exist_ok=True)
    (RESULTS / "mlflow").mkdir(parents=True, exist_ok=True)

    frame = load_or_build_features()
    ds = make_dataset(frame)
    model = mlflow.lightgbm.load_model(f"models:/{REGISTERED_NAME}@champion")

    val = score_frame(model, ds.X_val, ds.y_val, ds.meta_val)
    test = score_frame(model, ds.X_test, ds.y_test, ds.meta_test)

    # ---- Part 1: headline metrics -------------------------------------------
    _write_json(
        RESULTS / "part1" / "ranking_metrics.json",
        {"validation": ranking_metrics(val), "test": ranking_metrics(test)},
    )

    _write_csv(RESULTS / "part1" / "operating_points_val.csv", operating_points(val))
    _write_csv(RESULTS / "part1" / "operating_points_test.csv", operating_points(test))

    # threshold fixed on validation at the 1% budget, then applied to test
    thr = threshold_for_budget(val, ALERT_BUDGET_PCT)
    _write_csv(
        RESULTS / "part1" / "recall_by_typology.csv",
        recall_by_typology(test, thr).reset_index(),
        index=False,
    )

    sweep_src = ARTIFACTS_DIR / "sweep_results.csv"
    if sweep_src.exists():
        shutil.copy(sweep_src, RESULTS / "part1" / "sweep_results.csv")
        print("wrote results/part1/sweep_results.csv")

    # ---- Part 1: SHAP explainability --------------------------------------
    explainer = build_explainer(model)
    sample = ds.X_test.sample(n=min(20000, len(ds.X_test)), random_state=0)
    sv = shap_frame(explainer, sample)
    _write_csv(RESULTS / "part1" / "shap_global_importance.csv", global_importance(sv))

    # the worked example on the explainability slide (shared selection logic)
    tp = ds.X_test[ds.y_test.to_numpy() == 1]
    tp_scores = model.predict_proba(tp)[:, 1]
    alert_i = pick_reason_code_alert(model, explainer, tp, tp_scores)
    _write_json(
        RESULTS / "part1" / "shap_example_alert.json",
        explain_alert(model, explainer, tp.iloc[[alert_i]], top_n=8),
    )

    # ---- MLflow: tracking + registry ------------------------------------
    runs = mlflow.search_runs(experiment_names=["aml-part1"], order_by=["start_time"])
    keep = [c for c in runs.columns if c.startswith(("params.", "metrics.", "tags.mlflow.runName")) or c == "run_id"]
    _write_csv(RESULTS / "mlflow" / "experiment_runs.csv", runs[keep])

    client = mlflow.MlflowClient()
    alias_by_version: dict[str, list[str]] = {}
    for alias, ver in client.get_registered_model(REGISTERED_NAME).aliases.items():
        alias_by_version.setdefault(str(ver), []).append(alias)
    reg_rows = []
    for mv in client.search_model_versions(f"name='{REGISTERED_NAME}'"):
        reg_rows.append(
            {
                "name": mv.name,
                "version": mv.version,
                "aliases": ",".join(sorted(alias_by_version.get(str(mv.version), []))),
                "run_id": mv.run_id,
                "status": mv.status,
            }
        )
    _write_csv(RESULTS / "mlflow" / "registry.csv", pd.DataFrame(reg_rows).sort_values("version"))

    # ---- Part 2: monitoring replay + challenger comparison --------------
    mon = ARTIFACTS_DIR / "monitoring_history.csv"
    if mon.exists():
        shutil.copy(mon, RESULTS / "part2" / "monitoring_history.csv")
        print("wrote results/part2/monitoring_history.csv")

    comparison = {"champion": {}, "challenger": {}, "note": "challenger not registered"}
    try:
        chal = mlflow.lightgbm.load_model(f"models:/{REGISTERED_NAME}@challenger")
        aug = frame[frame["month"] == "2023-08"]
        Xa = aug[ds.X_test.columns]
        ya = aug["Is_laundering"].to_numpy()
        from sklearn.metrics import average_precision_score

        for label, m in [("champion", model), ("challenger", chal)]:
            sc = m.predict_proba(Xa)[:, 1]
            t = float(np.quantile(sc, 1 - ALERT_BUDGET_PCT))
            alert = sc >= t
            comparison[label] = {
                "august_pr_auc": round(float(average_precision_score(ya, sc)), 4),
                "august_recall_at_budget": round(float(alert[ya == 1].mean()), 4),
            }
        comparison["note"] = "threshold set per model at the 1% budget on August scores"
        comparison["challenger_wins"] = (
            comparison["challenger"]["august_pr_auc"] > comparison["champion"]["august_pr_auc"]
        )
    except Exception as exc:  # noqa: BLE001
        comparison["note"] = f"challenger load failed: {exc}"
    _write_json(RESULTS / "part2" / "champion_vs_challenger.json", comparison)

    _readme(comparison)
    print("\nall results written to", RESULTS.relative_to(REPO_ROOT))


def _readme(comparison: dict) -> None:
    (RESULTS / "README.md").write_text(
        "# results/\n\n"
        "Frozen copies of every number quoted in the presentation. Regenerate with\n"
        "`uv run python scripts/export_results.py` after re-running Parts 1 and 2.\n\n"
        "| file | what it is |\n"
        "|---|---|\n"
        "| `part1/ranking_metrics.json` | PR-AUC / ROC-AUC on validation and held-out test |\n"
        "| `part1/operating_points_*.csv` | precision, recall, alerts-per-day across alert budgets |\n"
        "| `part1/recall_by_typology.csv` | per-typology recall at the 1% budget (test) |\n"
        "| `part1/sweep_results.csv` | the scale_pos_weight sweep |\n"
        "| `part1/shap_global_importance.csv` | mean absolute SHAP per feature |\n"
        "| `part1/shap_example_alert.json` | reason codes for one laundering alert |\n"
        "| `mlflow/experiment_runs.csv` | every tracked run in the `aml-part1` experiment |\n"
        "| `mlflow/registry.csv` | registered model versions and their aliases |\n"
        "| `part2/monitoring_history.csv` | the daily July-August monitoring replay |\n"
        "| `part2/champion_vs_challenger.json` | the retrain comparison on August |\n\n"
        f"Champion vs challenger (August): {json.dumps(comparison, indent=2, default=str)}\n"
    )
    print("wrote results/README.md")


if __name__ == "__main__":
    main()
