"""Model training: a linear baseline and the LightGBM model, both tracked in MLflow.

Two models on purpose:
  * Logistic regression -- a transparent reference. If a linear model already
    separates the classes on these features, the features carry real signal and
    any LightGBM gain is incremental, not load-bearing.
  * LightGBM -- the model we would ship: handles the categorical + count feature
    mix natively and the class imbalance via cost-weighting (ADR-0005).

Both log params, metrics and the fitted model to a local MLflow store
(``mlruns/`` in the repo root). MLflow is the experiment-tracking tool named in
the role, so wiring it in from the first training run is deliberate.
"""

from __future__ import annotations

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aml_monitoring.config import MLFLOW_URI, SEED
from aml_monitoring.dataset import CATEGORICAL_COLS, NUMERIC_FEATURES, Dataset

EXPERIMENT = "aml-part1"


def _init_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)


def _pr_auc(model, X, y) -> float:
    """Average precision (area under the precision-recall curve)."""
    scores = model.predict_proba(X)[:, 1]
    return float(average_precision_score(y, scores))


def train_baseline(ds: Dataset) -> Pipeline:
    """Fit the logistic-regression baseline and log it to MLflow."""
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )
    model = Pipeline(
        [
            ("pre", pre),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",  # cheap imbalance handling for the baseline
                    max_iter=1000,
                    random_state=SEED,
                ),
            ),
        ]
    )

    _init_mlflow()
    with mlflow.start_run(run_name="logreg-baseline"):
        model.fit(ds.X_train, ds.y_train)
        val_pr_auc = _pr_auc(model, ds.X_val, ds.y_val)
        mlflow.log_params({"model": "logreg", "class_weight": "balanced"})
        mlflow.log_metric("val_pr_auc", val_pr_auc)
        mlflow.sklearn.log_model(model, name="model", serialization_format="pickle")
        print(f"[logreg] val PR-AUC = {val_pr_auc:.4f}")

    return model


def train_lightgbm(ds: Dataset, scale_pos_weight: float = 1.0) -> lgb.LGBMClassifier:
    """Fit LightGBM and log it to MLflow.

    Args:
        ds: The temporal split.
        scale_pos_weight: Weight on the positive class in the loss. The naive
            default would be negatives/positives (~1000). A sweep on the
            validation month showed that collapses PR-AUC ~20x while ROC-AUC
            barely moves -- the model already ranks well, and heavy weighting
            just floods the top of the alert list. We keep the loss ~unweighted
            and handle imbalance at the threshold (ADR-0005). Kept as an
            argument so the sweep is reproducible.
    """
    params = {
        "objective": "binary",
        "n_estimators": 600,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": -1,
    }
    model = lgb.LGBMClassifier(**params)

    _init_mlflow()
    with mlflow.start_run(run_name="lightgbm"):
        model.fit(
            ds.X_train,
            ds.y_train,
            eval_set=[(ds.X_val, ds.y_val)],
            eval_metric="average_precision",
            categorical_feature=CATEGORICAL_COLS,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        val_pr_auc = _pr_auc(model, ds.X_val, ds.y_val)
        mlflow.log_params(params)
        mlflow.log_metric("best_iteration", model.best_iteration_ or params["n_estimators"])
        mlflow.log_metric("val_pr_auc", val_pr_auc)
        mlflow.lightgbm.log_model(model, name="model")
        print(
            f"[lightgbm] scale_pos_weight={params['scale_pos_weight']}  "
            f"best_iter={model.best_iteration_}  val PR-AUC = {val_pr_auc:.4f}"
        )

    return model
