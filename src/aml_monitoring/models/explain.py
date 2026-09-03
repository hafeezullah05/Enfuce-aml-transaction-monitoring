"""Per-alert explanations for the LightGBM model, with SHAP.

Why this module exists: in an AML control every alert that reaches an
investigator needs a reason, and every SAR filed to the regulator needs a
documented justification. Global feature importance ("the model mostly uses
account history") is not enough — model-risk validation asks for the driver of
an *individual* decision.

SHAP gives an additive per-transaction breakdown: base value + sum of
per-feature contributions = the model's margin for that transaction. We use the
tree-path-dependent ``TreeExplainer``, which reads the trained tree structure
directly (no background dataset, exact for GBDTs).

  * ``build_explainer``   -- one explainer, reused.
  * ``shap_frame``        -- SHAP values for a batch of rows as a tidy DataFrame.
  * ``global_importance`` -- mean |SHAP| per feature (the honest importance rank).
  * ``explain_alert``     -- the top signed drivers for a single transaction,
                             i.e. the reason codes an investigator would see.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import shap

from aml_monitoring.dataset import FEATURE_COLS

# shap emits an informational UserWarning about the LightGBM binary output shape;
# _positive_class() handles every shape it can return, so silence the noise.
warnings.filterwarnings(
    "ignore",
    message="LightGBM binary classifier with TreeExplainer",
    category=UserWarning,
)


def build_explainer(model) -> shap.TreeExplainer:
    """A tree-path-dependent SHAP explainer for a fitted LightGBM classifier."""
    return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")


def _positive_class(values) -> np.ndarray:
    """Normalise SHAP output to a single (n_rows, n_features) array for the
    positive class. Across shap/LightGBM versions this comes back as one 2-D
    array, a length-2 list (one array per class), or a 3-D array."""
    if isinstance(values, list):  # [neg_class, pos_class]
        return np.asarray(values[-1])
    arr = np.asarray(values)
    if arr.ndim == 3:
        # (n_classes, n_rows, n_features) or (n_rows, n_features, n_classes)
        return arr[-1] if arr.shape[0] == 2 else arr[:, :, -1]
    return arr


def shap_frame(explainer: shap.TreeExplainer, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP contributions for every row in ``X`` (columns = ``FEATURE_COLS``)."""
    raw = explainer.shap_values(X[FEATURE_COLS], check_additivity=False)
    return pd.DataFrame(_positive_class(raw), columns=FEATURE_COLS, index=X.index)


def global_importance(sv: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute SHAP per feature — the ranking to quote as 'what the model uses'."""
    imp = sv.abs().mean().sort_values(ascending=False)
    return pd.DataFrame({"feature": imp.index, "mean_abs_shap": imp.to_numpy()})


def explain_alert(
    model,
    explainer: shap.TreeExplainer,
    x_row: pd.DataFrame,
    top_n: int = 6,
) -> dict:
    """Reason codes for one transaction.

    ``x_row`` is a single-row DataFrame with the model's feature columns and
    dtypes (e.g. ``X_test.iloc[[i]]``). Returns the model score, the SHAP base
    value, and the ``top_n`` features by absolute contribution with their value
    and signed push (positive = raises laundering risk).
    """
    x = x_row[FEATURE_COLS]
    values = x.iloc[0]

    contribs = _positive_class(explainer.shap_values(x, check_additivity=False))[0]
    base = float(np.ravel(explainer.expected_value)[-1])
    score = float(model.predict_proba(x)[:, 1][0])

    order = np.argsort(np.abs(contribs))[::-1][:top_n]
    drivers = [
        {
            "feature": FEATURE_COLS[i],
            "value": _clean(values.iloc[i]),
            "shap": round(float(contribs[i]), 4),
            "direction": "raises risk" if contribs[i] > 0 else "lowers risk",
        }
        for i in order
    ]
    return {
        "score": round(score, 4),
        "shap_base_value": round(base, 4),
        "margin": round(base + float(contribs.sum()), 4),
        "drivers": drivers,
    }


def _clean(v):
    """JSON-friendly scalar."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 4)
    return str(v)


_AMOUNT_DRIVERS = ("amount_log", "sender_amount_vs_mean_7d", "receiver_amount_vs_mean_7d", "Payment_type")


def pick_reason_code_alert(model, explainer, tp: pd.DataFrame, tp_scores: np.ndarray) -> int:
    """Row index into ``tp`` of a good worked example for the explainability slide.

    Wants a confidently-alerted true positive whose top drivers (a) include a
    clear laundering-shaped signal (an amount spike or a cash payment type) and
    (b) include at least one feature that *lowers* the score, so the two-colour
    raises/lowers encoding is visible rather than a wall of one colour. Falls
    back to the highest-scoring alert if nothing matches.
    """
    order = np.argsort(np.abs(tp_scores - 0.9))
    for cand in order[:200]:
        if not 0.80 <= tp_scores[cand] <= 0.99:
            continue
        drivers = explain_alert(model, explainer, tp.iloc[[int(cand)]], top_n=7)["drivers"]
        has_neg = any(d["shap"] < -0.7 for d in drivers)
        has_amount = any(d["feature"] in _AMOUNT_DRIVERS for d in drivers[:3])
        if has_neg and has_amount:
            return int(cand)
    return int(np.argmax(tp_scores))
