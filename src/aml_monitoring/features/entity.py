"""Entity-level behavioural features (per sender account and per receiver account).

These describe *how the account has been transacting up to — but not including —
the current transaction*. This is where the signal for the behavioural typologies
lives (Structuring, Smurfing, Fan-in/out, Gather-Scatter).

LEAKAGE PREVENTION — the single most important thing in this file:
  * The frame must already be globally sorted by ``timestamp`` (load_months does this).
  * Rolling windows use ``closed="left"`` -> the window is [t - window, t), which
    EXCLUDES the current transaction and anything after it.
  * ``groupby(key).cumcount()`` counts only prior rows for that entity.
  * A unit test (tests/test_entity_features.py) asserts the first row for an entity
    sees an empty history.
"""

from __future__ import annotations

import pandas as pd

WINDOWS = ["1D", "7D", "30D"]

ENTITY_FEATURES: list[str] = []
for _role in ("sender", "receiver"):
    ENTITY_FEATURES.append(f"{_role}_prior_txn_count")
    ENTITY_FEATURES.append(f"{_role}_secs_since_last")
    for _w in WINDOWS:
        ENTITY_FEATURES.append(f"{_role}_cnt_{_w.lower()}")
    ENTITY_FEATURES.append(f"{_role}_sum_7d")
    ENTITY_FEATURES.append(f"{_role}_mean_7d")
    ENTITY_FEATURES.append(f"{_role}_amount_vs_mean_7d")


def add_entity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add causal (past-only) per-account behavioural features.

    Args:
        df: Must be sorted ascending by ``timestamp`` and contain
            ``Sender_account``, ``Receiver_account``, ``Amount``, ``timestamp``.

    Returns:
        New frame with the columns listed in ``ENTITY_FEATURES`` added.
    """
    out = df.copy()

    # Rolling-on-time needs a DatetimeIndex; keep the original order to restore later.
    out = out.set_index("timestamp")

    for role, key in (("sender", "Sender_account"), ("receiver", "Receiver_account")):
        grp = out.groupby(key, sort=False)

        # Number of PRIOR transactions by this entity (0 for its first ever row).
        out[f"{role}_prior_txn_count"] = grp.cumcount()

        # Seconds since this entity's previous transaction (velocity).
        prev_ts = grp["Amount"].transform(lambda s: s.index.to_series().shift(1))
        out[f"{role}_secs_since_last"] = (out.index.to_series().values - prev_ts.values)
        out[f"{role}_secs_since_last"] = out[f"{role}_secs_since_last"].dt.total_seconds()

        # Windowed activity. closed="left" => excludes the current transaction.
        roll = grp["Amount"].rolling(window="7D", closed="left")
        out[f"{role}_cnt_7d"] = roll.count().to_numpy()
        out[f"{role}_sum_7d"] = roll.sum().to_numpy()
        out[f"{role}_mean_7d"] = roll.mean().to_numpy()

        for w in ("1D", "30D"):
            out[f"{role}_cnt_{w.lower()}"] = (
                grp["Amount"].rolling(window=w, closed="left").count().to_numpy()
            )

        # How large is this transaction vs the entity's recent norm? (spike detection)
        out[f"{role}_amount_vs_mean_7d"] = out["Amount"] / (out[f"{role}_mean_7d"] + 1.0)

    return out.reset_index()