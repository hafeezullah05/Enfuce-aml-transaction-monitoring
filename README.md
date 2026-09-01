# Enfuce — AML Transaction Monitoring (technical case)

An ML-based transaction-monitoring solution, built with a production / MLOps
focus. The emphasis is on **technical decisions and their reasoning**, not model
score.

## Stack

Python 3.11 · uv · pandas · scikit-learn / LightGBM · MLflow (tracking + registry)
· pandera-style typed loading · Ruff · pytest

## Layout

| Path | What |
|---|---|
| `src/aml_monitoring/` | The pipeline as an installable package |
| `src/aml_monitoring/data/` | Loading the monthly SAML-D files, time-ordered |
| `src/aml_monitoring/features/` | Transaction-level + **causal** entity-behaviour features (leakage-tested) |
| `src/aml_monitoring/dataset.py` | Feature build + temporal split + parquet cache |
| `src/aml_monitoring/models/` | Pure fit functions + the evaluation module |
| `scripts/run_part1_models.py` | Offline training job — sweeps, logs to MLflow, registers the model |
| `notebooks/main.ipynb` | The Part 1 narrative — loads the registered model, evaluates |
| `docs/` | Model write-up, architecture, delivery plan, ADRs |
| `presentation/` | Deck + speaker notes + generator |
| `tests/` | Leakage tests for the entity features |

## Run

```bash
uv sync
# put the SAML-D monthly CSVs in Dataset/data/  (not committed)
uv run python scripts/run_part1_models.py      # Part 1: train sweep + register @champion
uv run jupyter lab notebooks/main.ipynb        # Part 1: evaluate
uv run python scripts/run_part2.py             # Part 2: monitor Jul-Aug, trigger, challenger
uv run jupyter lab notebooks/part2.ipynb       # Part 2: dashboard + narrative
uv run pytest
```

## Parts

| Part | Status | Where |
|---|---|---|
| **1 — Model development & evaluation** | ✅ built | `src/`, `notebooks/main.ipynb`, `docs/part1-model.md`, ADRs 0001–0005 |
| **2 — ML lifecycle & MLOps** | ✅ built (monitoring + retraining trigger + challenger promotion, run on Jul–Aug) | `src/aml_monitoring/monitoring/`, `src/aml_monitoring/lifecycle.py`, `scripts/run_part2.py`, `notebooks/part2.ipynb`, `docs/part2-lifecycle.md` |
| **3 — Production architecture** | design + trade-offs + Terraform sketch | `docs/architecture.md` |
| **4 — Delivery (3 / 6 / 9 months)** | design | `docs/delivery-plan.md` |

## Headline result

Held-out test (Jul–Aug 2023): PR-AUC **0.54**. At a 1% daily alert budget —
~**400 alerts/day, 77% of laundering caught**. The model loses ~0.14 PR-AUC over
two months, which motivates the Part 2 monitoring + retraining design.
