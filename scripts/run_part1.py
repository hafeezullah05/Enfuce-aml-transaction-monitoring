"""Offline training job for Part 1.

Fits the Logistic Regression baseline and LightGBM at several ``scale_pos_weight``
values, logs every run to MLflow, writes the sweep table to
``artifacts/sweep_results.csv`` and registers the chosen model as
``aml-transaction-monitoring`` in the MLflow model registry.

The notebook loads the registered model; it never retrains. Run:

    uv run python scripts/run_part1.py
"""

from __future__ import annotations

import mlflow
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from aml_monitoring.config import ARTIFACTS_DIR, MLFLOW_URI
from aml_monitoring.dataset import load_or_build_features, make_dataset
from aml_monitoring.models.train import fit_baseline, fit_lightgbm, lgbm_params

EXPERIMENT = "aml-part1"
SWEEP = [1.0, 5.0, 100.0, 1003.0]  # 1003 ~= negatives/positives (the naive default)
SHIP = 1.0
REGISTERED_NAME = "aml-transaction-monitoring"


def _val_metrics(model, ds) -> tuple[float, float]:
    p = model.predict_proba(ds.X_val)[:, 1]
    return average_precision_score(ds.y_val, p), roc_auc_score(ds.y_val, p)


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    ds = make_dataset(load_or_build_features())
    print("train", ds.X_train.shape, "val", ds.X_val.shape, "test", ds.X_test.shape, flush=True)

    rows: list[dict] = []

    with mlflow.start_run(run_name="logreg-baseline"):
        baseline = fit_baseline(ds)
        pr, roc = _val_metrics(baseline, ds)
        mlflow.log_param("model", "logreg")
        mlflow.log_metrics({"val_pr_auc": pr, "val_roc_auc": roc})
        mlflow.sklearn.log_model(baseline, name="model", serialization_format="pickle")
    rows.append({"model": "logreg", "scale_pos_weight": None,
                 "val_pr_auc": round(pr, 4), "val_roc_auc": round(roc, 4)})
    print(f"done logreg  pr_auc={pr:.4f}", flush=True)

    from tqdm.auto import tqdm

    ship_run_id: str | None = None
    for spw in tqdm(SWEEP, desc="scale_pos_weight sweep", unit="model"):
        with mlflow.start_run(run_name=f"lightgbm-spw{spw:g}") as run:
            model = fit_lightgbm(ds, scale_pos_weight=spw, progress=True)
            pr, roc = _val_metrics(model, ds)
            mlflow.log_params(lgbm_params(spw))
            mlflow.log_metrics({
                "val_pr_auc": pr,
                "val_roc_auc": roc,
                "best_iteration": model.best_iteration_ or lgbm_params(spw)["n_estimators"],
            })
            mlflow.lightgbm.log_model(model, name="model")
            if spw == SHIP:
                ship_run_id = run.info.run_id
        rows.append({"model": "lightgbm", "scale_pos_weight": spw,
                     "val_pr_auc": round(pr, 4), "val_roc_auc": round(roc, 4)})
        print(f"done spw={spw:g}  pr_auc={pr:.4f}", flush=True)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(ARTIFACTS_DIR / "sweep_results.csv", index=False)

    assert ship_run_id is not None
    mv = mlflow.register_model(f"runs:/{ship_run_id}/model", REGISTERED_NAME)
    mlflow.MlflowClient().set_registered_model_alias(REGISTERED_NAME, "champion", mv.version)
    print(f"\nregistered {REGISTERED_NAME} v{mv.version} as @champion  (scale_pos_weight={SHIP})")
    print(sweep.to_string(index=False))


if __name__ == "__main__":
    main()
