"""Locks the causal (no-leakage) behaviour of the entity features."""

import pandas as pd

from aml_monitoring.features.entity import add_entity_features


def _toy() -> pd.DataFrame:
    # Sender 1 has three txns on consecutive days; sender 2 has one.
    return pd.DataFrame(
        {
            "Sender_account": [1, 1, 1, 2],
            "Receiver_account": [9, 9, 8, 7],
            "Amount": [100.0, 200.0, 300.0, 50.0],
            "timestamp": pd.to_datetime(
                [
                    "2023-01-01 10:00",
                    "2023-01-02 10:00",
                    "2023-01-03 10:00",
                    "2023-01-01 11:00",
                ]
            ),
        }
    )


def test_first_txn_sees_no_history() -> None:
    out = add_entity_features(_toy())
    first = out[out["Sender_account"] == 1].sort_values("timestamp").iloc[0]
    assert first["sender_prior_txn_count"] == 0
    assert first["sender_cnt_7d"] == 0
    assert pd.isna(first["sender_secs_since_last"])
    assert pd.isna(first["sender_mean_7d"])


def test_windows_are_causal() -> None:
    out = add_entity_features(_toy())
    a = out[out["Sender_account"] == 1].sort_values("timestamp").reset_index(drop=True)
    assert list(a["sender_prior_txn_count"]) == [0, 1, 2]
    assert list(a["sender_cnt_7d"]) == [0, 1, 2]
    assert a.loc[1, "sender_secs_since_last"] == 86400.0      # exactly one day
    assert a.loc[2, "sender_sum_7d"] == 300.0                 # 100 + 200, current (300) excluded