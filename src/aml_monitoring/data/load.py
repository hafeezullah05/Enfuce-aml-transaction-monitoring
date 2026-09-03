"""Load the SAML-D monthly transaction files into one tidy, time-ordered frame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aml_monitoring.config import RAW_DIR

# Explicit dtypes: half the memory, and the schema intent is visible in code.
DTYPES = {
    "Sender_account": "int64",
    "Receiver_account": "int64",
    "Amount": "float64",
    "Payment_currency": "category",
    "Received_currency": "category",
    "Sender_bank_location": "category",
    "Receiver_bank_location": "category",
    "Payment_type": "category",
    "Is_laundering": "int8",
    "Laundering_type": "category",
}


def _month_path(month: str) -> Path:
    """'2023-06' -> <repo>/Dataset/data/transactions_2023-06.csv.gz"""
    return RAW_DIR / f"transactions_{month}.csv.gz"


def load_months(months: list[str]) -> pd.DataFrame:
    """Read the given months and return rows sorted by arrival time.

    A single global sort by ``timestamp`` is a hard requirement for the causal
    entity features built next: every rolling window must see transactions in
    the exact order they arrived, or future information leaks into the past.

    Args:
        months: e.g. ``["2022-10", "2022-11"]``.

    Returns:
        DataFrame with an added ``timestamp`` (datetime) and ``month`` (str) column,
        sorted ascending by ``timestamp``, index reset.
    """
    from tqdm.auto import tqdm

    frames = []
    for m in tqdm(months, desc="loading months", unit="file"):
        part = pd.read_csv(_month_path(m), dtype=DTYPES)
        part["month"] = m
        frames.append(part)

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%Y-%m-%d %H:%M:%S"
    )
    return df.sort_values("timestamp").reset_index(drop=True)