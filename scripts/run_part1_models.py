"""Run Section 6-7 end to end on the real split and print the comparison table."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from aml_monitoring.dataset import load_or_build_features, make_dataset
from aml_monitoring.models.train import train_baseline, train_lightgbm


def main() -> None:
    ds = make_dataset(load_or_build_features())
    print("train", ds.X_train.shape, "val", ds.X_val.shape, "test", ds.X_test.shape, flush=True)

    train_baseline(ds)

    rows = []
    for spw in [1.0, 5.0, 100.0, 1003.0]:
        m = train_lightgbm(ds, scale_pos_weight=spw)
        p = m.predict_proba(ds.X_val)[:, 1]
        rows.append({
            "scale_pos_weight": spw,
            "val_pr_auc": round(average_precision_score(ds.y_val, p), 4),
            "val_roc_auc": round(roc_auc_score(ds.y_val, p), 4),
        })
        print("done spw", spw, rows[-1], flush=True)

    print("\n=== COMPARISON ===")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
