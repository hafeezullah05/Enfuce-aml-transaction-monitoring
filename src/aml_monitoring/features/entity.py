"""Entity-level behavioural features (per sender account and per receiver account).

Every feature describes how the account was transacting BEFORE the current
transaction — never including it. This is where the signal for the behavioural
typologies lives (Structuring, Smurfing, Fan-in/out, Gather-Scatter).

Leakage prevention (this is the point of the file):
  * we work on a copy sorted by (entity, timestamp);
  * expanding features use ``cumcount`` / diff -> prior rows only;
  * windowed features use ``rolling(w, closed="left")`` -> window [t - w, t),
    which excludes the current transaction and everything after it;
  * results are mapped back to the caller's row order via an explicit position
    column, never by a (non-unique) timestamp index;
  * tests/test_entity_features.py locks this behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ROLES = (("sender", "Sender_account"), ("receiver", "Receiver_account"))

ENTITY_FEATURES: list[str] = []
for _r, _ in ROLES:
    ENTITY_FEATURES += [
        f"{_r}_prior_txn_count",     # how many txns this account has had before
        f"{_r}_secs_since_last",     # velocity: seconds since its previous txn
        f"{_r}_cnt_1d",              # txns in the trailing 24h
        f"{_r}_cnt_7d",              # txns in the trailing 7d
        f"{_r}_sum_7d",              # summed amount, trailing 7d
        f"{_r}_mean_7d",             # mean amount, trailing 7d
        f"{_r}_amount_vs_mean_7d",   # this amount / recent mean  (spike detection)
    ]


def add_entity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add causal (past-only) per-account behavioural features.

    Args:
        df: Needs columns ``Sender_account``, ``Receiver_account``, ``Amount``,
            ``timestamp``. Input row order is irrelevant — it is restored on return.

    Returns:
        A new frame with the columns in ``ENTITY_FEATURES`` added, in input row order.
    """
    out = df.reset_index(drop=True).copy()
    out["_pos"] = np.arange(len(out))  # remember original order

    for role, key in ROLES:
        # Group this entity's rows together, ordered in time within each group.
        s = out.sort_values([key, "timestamp"], kind="stable").copy()
        g = s.groupby(key, sort=False)

        # --- expanding: all prior history ---
        prior_count = g.cumcount().to_numpy()               # 0 on an entity's first txn
        secs = s["timestamp"].diff().dt.total_seconds().to_numpy()
        secs[prior_count == 0] = np.nan                     # diff across a group boundary is invalid

        # --- windowed: rolling on a DatetimeIndex, current row excluded ---
        rgrp = s.set_index("timestamp").groupby(key, sort=False)["Amount"]
        # An empty trailing window means "no recent activity" -> 0 for counts and
        # summed volume; the mean stays NaN (there is genuinely no recent mean).
        cnt_1d = np.nan_to_num(rgrp.rolling("1D", closed="left").count().to_numpy())
        r7 = rgrp.rolling("7D", closed="left")
        cnt_7d = np.nan_to_num(r7.count().to_numpy())
        sum_7d = np.nan_to_num(r7.sum().to_numpy())
        mean_7d = r7.mean().to_numpy()
        # ^ output rows are in the same order as `s` (groups in first-appearance
        #   order = ascending key; time-ordered within group), so positional
        #   assignment below is correct.

        s[f"{role}_prior_txn_count"] = prior_count
        s[f"{role}_secs_since_last"] = secs
        s[f"{role}_cnt_1d"] = cnt_1d
        s[f"{role}_cnt_7d"] = cnt_7d
        s[f"{role}_sum_7d"] = sum_7d
        s[f"{role}_mean_7d"] = mean_7d
        s[f"{role}_amount_vs_mean_7d"] = s["Amount"].to_numpy() / (mean_7d + 1.0)

        # Map back to the caller's row order.
        s = s.sort_values("_pos")
        for feat in ENTITY_FEATURES:
            if feat.startswith(role):
                out[feat] = s[feat].to_numpy()

    return out.drop(columns="_pos")