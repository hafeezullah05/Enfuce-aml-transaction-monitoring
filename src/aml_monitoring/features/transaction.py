"""Transaction-level features: derived purely from a single row.

No history, no windows -> no leakage risk. Per the EDA, corridor features
(cross-border, currency mismatch) and the large-amount flag carry real signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Low-cardinality columns fed straight to the model as categoricals.
CATEGORICAL_COLS = [
    "Payment_type",
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
]

# Numeric transaction-level features this module produces.
TRANSACTION_FEATURES = [
    "amount_log",
    "amount_is_small",
    "amount_is_large",
    "cross_border",
    "currency_mismatch",
    "hour",
    "day_of_week",
]


def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add single-row derived features. Returns a new frame (input not mutated)."""
    out = df.copy()

    # Amount is skewed and bimodal for laundering -> log + shape flags.
    out["amount_log"] = np.log1p(out["Amount"])
    out["amount_is_small"] = (out["Amount"] < 1_000).astype("int8")      # smurfing / structuring
    out["amount_is_large"] = (out["Amount"] > 100_000).astype("int8")    # single-large typology

    # Corridor features -- strongest cheap signal in the EDA.
    out["cross_border"] = (
        out["Sender_bank_location"].astype(str) != out["Receiver_bank_location"].astype(str)
    ).astype("int8")
    out["currency_mismatch"] = (
        out["Payment_currency"].astype(str) != out["Received_currency"].astype(str)
    ).astype("int8")

    # Time-of-arrival: keep raw hour / weekday for the tree to split on;
    # the boolean night/weekend versions showed no signal in EDA.
    out["hour"] = out["timestamp"].dt.hour.astype("int8")
    out["day_of_week"] = out["timestamp"].dt.dayofweek.astype("int8")

    return out