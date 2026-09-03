# results/

Frozen copies of every number quoted in the presentation. Regenerate with
`uv run python scripts/export_results.py` after re-running Parts 1 and 2.

| file | what it is |
|---|---|
| `part1/ranking_metrics.json` | PR-AUC / ROC-AUC on validation and held-out test |
| `part1/operating_points_*.csv` | precision, recall, alerts-per-day across alert budgets |
| `part1/recall_by_typology.csv` | per-typology recall at the 1% budget (test) |
| `part1/sweep_results.csv` | the scale_pos_weight sweep |
| `part1/shap_global_importance.csv` | mean absolute SHAP per feature |
| `part1/shap_example_alert.json` | reason codes for one laundering alert |
| `mlflow/experiment_runs.csv` | every tracked run in the `aml-part1` experiment |
| `mlflow/registry.csv` | registered model versions and their aliases |
| `part2/monitoring_history.csv` | the daily July-August monitoring replay |
| `part2/champion_vs_challenger.json` | the retrain comparison on August |

Champion vs challenger (August): {
  "champion": {
    "august_pr_auc": 0.4543,
    "august_recall_at_budget": 0.6614
  },
  "challenger": {
    "august_pr_auc": 0.6851,
    "august_recall_at_budget": 0.8193
  },
  "note": "threshold set per model at the 1% budget on August scores",
  "challenger_wins": true
}
