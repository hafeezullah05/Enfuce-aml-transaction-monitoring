"""The reference snapshot a live batch is compared against.

Built once, at deployment, from the data the model was trained + validated on,
and versioned alongside the model. "Drift against what" is then unambiguous:
against the world as it looked when this model went live.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aml_monitoring.dataset import CATEGORICAL_COLS, NUMERIC_FEATURES
from aml_monitoring.monitoring.drift import numeric_bins


@dataclass
class Reference:
    """Everything a batch check needs. Serialisable to JSON."""

    numeric_edges: dict[str, list[float]]          # feature -> bin edges
    numeric_ref: dict[str, list[float]]            # feature -> a sample for PSI (down-sampled)
    categorical_ref: dict[str, dict[str, float]]   # feature -> {value: proportion}
    null_rate: dict[str, float]                    # feature -> null fraction in reference
    score_ref: list[float]                         # model scores on the validation month
    rows_per_day_mean: float                       # expected daily volume
    rows_per_day_std: float
    threshold: float                               # operating threshold (from validation, 1% budget)
    model_version: str

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__))

    @classmethod
    def load(cls, path: Path) -> Reference:
        return cls(**json.loads(Path(path).read_text()))


def build_reference(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    val_scores: np.ndarray,
    threshold: float,
    model_version: str,
    sample: int = 50_000,
) -> Reference:
    """Snapshot the training feature distributions + validation score distribution.

    Args:
        train_frame: feature rows from the training window (for input drift).
        val_frame:   feature rows from the validation month (for volume norm).
        val_scores:  model scores on the validation month (for prediction drift).
        threshold:   the operating threshold chosen on validation.
        model_version: the registry version this reference belongs to.
    """
    rng = np.random.default_rng(0)

    numeric_edges: dict[str, list[float]] = {}
    numeric_ref: dict[str, list[float]] = {}
    for col in NUMERIC_FEATURES:
        edges = numeric_bins(train_frame[col])
        numeric_edges[col] = edges.tolist()
        s = train_frame[col].dropna().to_numpy(dtype="float64")
        if s.size > sample:
            s = rng.choice(s, sample, replace=False)
        numeric_ref[col] = s.tolist()

    categorical_ref = {
        col: train_frame[col].astype(str).value_counts(normalize=True).to_dict()
        for col in CATEGORICAL_COLS
    }

    null_rate = {
        col: float(train_frame[col].isna().mean())
        for col in NUMERIC_FEATURES + CATEGORICAL_COLS
    }

    per_day = val_frame.groupby(pd.to_datetime(val_frame["timestamp"]).dt.date).size()

    scores = np.asarray(val_scores, dtype="float64")
    if scores.size > sample:
        scores = rng.choice(scores, sample, replace=False)

    return Reference(
        numeric_edges=numeric_edges,
        numeric_ref=numeric_ref,
        categorical_ref=categorical_ref,
        null_rate=null_rate,
        score_ref=scores.tolist(),
        rows_per_day_mean=float(per_day.mean()),
        rows_per_day_std=float(per_day.std()),
        threshold=float(threshold),
        model_version=str(model_version),
    )
