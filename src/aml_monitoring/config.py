"""Project configuration: paths, the temporal split, evaluation settings.

Kept as plain module constants — small project, no need for a config framework.
Every other module imports from here so paths and split boundaries live in one place.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "Dataset" / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"  # cached features, trained models, plots
MLFLOW_URI = f"file://{REPO_ROOT / 'mlruns'}"

SEED = 42

# --- Temporal split -------------------------------------------------------
# Months are the natural batch unit in this dataset. Train on the past,
# validate on the next month, test on the final two (also reused as
# "production batches" in Part 2).
TRAIN_MONTHS = [
    "2022-10", "2022-11", "2022-12", "2023-01",
    "2023-02", "2023-03", "2023-04", "2023-05",
]
VAL_MONTHS = ["2023-06"]
TEST_MONTHS = ["2023-07", "2023-08"]

# --- Columns -------------------------------------------------------------
TARGET = "Is_laundering"
LEAKY_COLS = ["Laundering_type"]  # target-derived; keep for analysis, never for training (ADR-0003)

# --- Evaluation --------------------------------------------------------
ALERT_BUDGET_PCT = 0.01  # investigators review the top 1% of each day's scores