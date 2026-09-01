"""Model fitting: a linear baseline and the LightGBM model.

These functions are pure -- they fit and return a model, nothing else.
MLflow tracking, the imbalance sweep and model registration live in the
orchestration script (``scripts/run_part1_models.py``). Keeping fitting and
orchestration separate is what lets the notebook load a registered model
without ever importing training code.

Two models on purpose:
  * Logistic regression -- a transparent reference. If a linear model already
    separates the classes on these features, the features carry real signal.
  * LightGBM -- the model we ship: handles the categorical + count feature mix
    natively, and imbalance via cost-weighting (ADR-0005).
"""

from __future__ import annotations

import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aml_monitoring.config import SEED
from aml_monitoring.dataset import CATEGORICAL_COLS, NUMERIC_FEATURES, Dataset


def fit_baseline(ds: Dataset) -> Pipeline:
    """Fit the Logistic Regression baseline (scaled numerics + one-hot categoricals)."""
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),  # NaN = account's first txn
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
                    class_weight="balanced",  # linear analogue of scale_pos_weight
                    max_iter=1000,
                    random_state=SEED,
                ),
            ),
        ]
    )
    model.fit(ds.X_train, ds.y_train)
    return model


def lgbm_params(scale_pos_weight: float = 1.0) -> dict:
    """LightGBM hyperparameters.

    Conservative on purpose: only ~5k positives in training, so a large tree
    (num_leaves=63) memorises them in a single boosting round and early stopping
    fires at iteration 1. Shallow trees + high ``min_child_samples`` + L1/L2
    regularisation force generalisable structure learned over many rounds.

    ``scale_pos_weight``: the naive default is negatives/positives (~1000); a
    sweep on the validation month shows that collapses PR-AUC to below the
    linear baseline. We ship ``1.0`` and handle imbalance at the threshold
    (ADR-0005). Kept as an argument so the sweep is reproducible.
    """
    return {
        "objective": "binary",
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 200,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "scale_pos_weight": scale_pos_weight,
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": -1,
    }


def fit_lightgbm(ds: Dataset, scale_pos_weight: float = 1.0) -> lgb.LGBMClassifier:
    """Fit LightGBM, early-stopping on validation average precision."""
    model = lgb.LGBMClassifier(**lgbm_params(scale_pos_weight))
    model.fit(
        ds.X_train,
        ds.y_train,
        eval_set=[(ds.X_val, ds.y_val)],
        eval_metric="average_precision",
        categorical_feature=CATEGORICAL_COLS,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    return model
