"""One-off exploratory data analysis for the SAML-D monthly transaction files.

Goal: answer the questions we need before making Part 1 modelling decisions:
  - how severe is the class imbalance, and is it stable over time?
  - what is the date range / are the monthly files clean?
  - cardinality of the categorical + account-id columns (affects encoding + entity features)
  - is `Laundering_type` label-derived (i.e. must be dropped as a feature)?
  - do a couple of cheap features (cross-border, currency mismatch) actually separate the classes?
  - do accounts recur? (entity-level aggregate features only make sense if they do)

Run:  uv run python scripts/01_eda.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path("Dataset/data")
FILES = sorted(RAW_DIR.glob("transactions_*.csv.gz"))  # 11 monthly gzipped CSVs

# Explicit dtypes: keeps memory down and makes the schema intent obvious.
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


def load_all() -> pd.DataFrame:
    """Read every monthly file, tag it with its month, build a single timestamp column."""
    frames = []
    for f in FILES:
        part = pd.read_csv(f, dtype=DTYPES)
        part["month"] = f.stem.replace("transactions_", "").replace(".csv", "")
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    # Date is 'YYYY-MM-DD', Time is 'HH:MM:SS' -> one proper datetime for ordering / windows
    df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y-%m-%d %H:%M:%S")
    return df


def main() -> None:
    df = load_all()

    print(f"\nrows: {len(df):,}   cols: {df.shape[1]}")
    print(f"date range: {df['timestamp'].min()}  ->  {df['timestamp'].max()}")
    print(f"months: {sorted(df['month'].unique())}")

    # 1. Class balance -- overall and per month (is the imbalance stable? any drift?)
    print("\n--- prevalence overall ---")
    print(df["Is_laundering"].value_counts())
    print(f"positive rate: {df['Is_laundering'].mean():.4%}")

    print("\n--- prevalence per month ---")
    print(df.groupby("month")["Is_laundering"].agg(n="count", positives="sum", rate="mean"))

    # 2. Missingness
    print("\n--- missing values per column ---")
    print(df.isna().sum())

    # 3. Cardinality -- drives encoding choice and whether entity features are viable
    print("\n--- cardinality ---")
    for c in [
        "Sender_account", "Receiver_account", "Payment_currency", "Received_currency",
        "Sender_bank_location", "Receiver_bank_location", "Payment_type", "Laundering_type",
    ]:
        print(f"{c:>24}: {df[c].nunique():,}")

    # 4. Amount distribution by class
    print("\n--- amount by class ---")
    print(df.groupby("Is_laundering")["Amount"].describe())

    # 5. Laundering_type -- if positives and negatives have disjoint label sets,
    #    the column is derived from the target and must NOT be a feature.
    print("\n--- Laundering_type for positives (Is_laundering == 1) ---")
    print(df.loc[df["Is_laundering"] == 1, "Laundering_type"].value_counts())
    print("\n--- Laundering_type for negatives (Is_laundering == 0), top 10 ---")
    print(df.loc[df["Is_laundering"] == 0, "Laundering_type"].value_counts().head(10))

    # 6. Cheap candidate features -- do they separate the classes at all?
    df["cross_border"] = (df["Sender_bank_location"] != df["Receiver_bank_location"]).astype(int)
    df["currency_mismatch"] = (df["Payment_currency"] != df["Received_currency"]).astype(int)
    print("\n--- cross_border rate by class ---")
    print(df.groupby("Is_laundering")["cross_border"].mean())
    print("\n--- currency_mismatch rate by class ---")
    print(df.groupby("Is_laundering")["currency_mismatch"].mean())
    print("\n--- payment_type share within each class ---")
    print(pd.crosstab(df["Payment_type"], df["Is_laundering"], normalize="columns"))

    # 7. Account recurrence -- entity aggregates need accounts to appear multiple times
    print("\n--- account recurrence ---")
    sender_counts = df["Sender_account"].value_counts()
    receiver_counts = df["Receiver_account"].value_counts()
    print(f"unique senders: {len(sender_counts):,}   unique receivers: {len(receiver_counts):,}")
    print(f"senders appearing >1x: {(sender_counts > 1).mean():.1%}")
    print(f"max transactions for one sender: {sender_counts.max():,}")
    # Do laundering accounts also send normal traffic? (matters for entity-level labelling)
    laundering_senders = set(df.loc[df["Is_laundering"] == 1, "Sender_account"])
    share_mixed = df.loc[df["Sender_account"].isin(laundering_senders), "Is_laundering"].mean()
    print(f"among senders who ever launder, laundering share of their txns: {share_mixed:.2%}")


if __name__ == "__main__":
    main()
