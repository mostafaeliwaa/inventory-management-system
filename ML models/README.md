# ML Models — Demand Classification & Dead-Stock Prediction

Two production-ready scikit-learn models served via FastAPI for the inventory management system. Trained on the 2,428-product snapshot in `../data/cleaned_sales_stock.csv` (joined with pricing).

## What this delivers

| Model | Type | Output |
|---|---|---|
| **Demand Classifier** | 3-class (slow / medium / fast) | label + class probabilities |
| **Dead-Stock Predictor** | binary | `is_dead_stock` + probability |

Both models are trained from the same engineered feature table (one row per product, 12 features) and are exposed through a single FastAPI service.

## Folder layout

```
ML models/
├── README.md                    ← you are here
├── api.py                       ← FastAPI service
├── requirements.txt
├── src/
│   ├── data_loader.py           ← reads CSVs, builds the 12-feature table
│   ├── labels.py                ← rule-based labels used as training ground truth
│   ├── train_demand.py          ← trains Random Forest demand classifier
│   ├── train_dead_stock.py      ← trains Gradient Boosting dead-stock model
│   ├── evaluate.py              ← robustness eval (hides rule-defining features)
│   └── predict.py               ← inference helpers + per-product lookup
├── artifacts/                   ← serialized .joblib models (created by training)
└── reports/                     ← JSON evaluation reports (created by training)
```

## Quick start

```bash
cd "ML models"
pip install -r requirements.txt

# 1. Train both models (writes artifacts/ + reports/)
python -m src.train_demand
python -m src.train_dead_stock

# 2. Run robustness evaluation (optional but recommended)
python -m src.evaluate

# 3. Start the API
uvicorn api:app --host 0.0.0.0 --port 8003 --reload
```

Open http://localhost:8003/docs for interactive Swagger UI.

## Features (per product)

Computed in `src/data_loader.py::build_feature_table`:

| Feature | Meaning |
|---|---|
| `total_sales`, `total_stock` | sum across the 13 branches |
| `branches_with_sales`, `branches_with_stock` | branch coverage |
| `max_branch_sales`, `max_branch_stock` | top branch values |
| `sales_concentration` | `max_branch_sales / total_sales` (0..1) |
| `stock_to_sales_ratio` | `total_stock / total_sales` |
| `sell_through` | `total_sales / (total_sales + total_stock)` |
| `cost_price`, `retail_price`, `margin_pct` | from `cleaned_pricing.csv` |

## Labels (training ground truth)

We have no expert-tagged data, so labels are derived deterministically from the feature table (see `src/labels.py`):

- **Demand** — quantile cuts on `total_sales`. Bottom 50% → `slow`, next 30% → `medium`, top 20% → `fast`. Zero-sales products are forced to `slow`.
- **Dead stock** — `(stock > 0 AND sales == 0)` OR (top-decile `stock_to_sales_ratio` AND `sell_through < 5%`).

## Evaluation

### Standard hold-out (25% test split)

| Model | Accuracy | Macro F1 / F1 | ROC-AUC |
|---|---|---|---|
| Demand classifier | 1.000 | 1.000 | — |
| Dead-stock predictor | 1.000 | 1.000 | 1.000 |

Perfect scores are **expected**: the labels are deterministic functions of features the model can see (`total_sales`, `stock_to_sales_ratio`, etc.). The model trivially learns the rule. This is fine in production — the model returns calibrated probabilities and generalizes to new products with the same feature space — but it's not a meaningful generalization test.

### Robustness evaluation (rule-defining features hidden)

To measure real generalization we re-train with the rule-defining features removed (`src/evaluate.py`). The remaining features must reconstruct the label.

| Model | Hidden features | Accuracy | F1 |
|---|---|---|---|
| Demand classifier | `total_sales`, `max_branch_sales` | 0.909 | macro **0.866** |
| Dead-stock predictor | `total_sales`, `total_stock`, `stock_to_sales_ratio`, `sell_through` | 0.997 | dead-class **0.967** (AUC 0.999) |

Reads: branch coverage and pricing alone recover demand class with 87% macro F1, and predict dead-stock with 97% F1 — strong signal that the models learn meaningful structure beyond the labeling rule.

Reports are written to `reports/*.json` after each run.

## API reference

Base URL: `http://localhost:8003`

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness probe |
| GET | `/metrics` | training + robustness reports as JSON |
| POST | `/predict/demand` | classify from raw 12-feature payload |
| POST | `/predict/dead-stock` | dead-stock from raw 12-feature payload |
| GET | `/predict/by-code/{code}` | look up a known product code, return both predictions |
| POST | `/predict/batch` | batch lookup by codes (max 500) |
| GET | `/codes/sample?limit=20` | sample known product codes for testing |

### Example: predict from features

```bash
curl -X POST http://localhost:8003/predict/demand \
  -H "Content-Type: application/json" \
  -d '{
    "total_sales": 50, "total_stock": 10,
    "branches_with_sales": 5, "branches_with_stock": 3,
    "max_branch_sales": 20, "max_branch_stock": 5,
    "sales_concentration": 0.4, "stock_to_sales_ratio": 0.2,
    "sell_through": 0.83,
    "cost_price": 30, "retail_price": 60, "margin_pct": 0.5
  }'
```

```json
{"label": "fast", "probabilities": {"fast": 0.99, "medium": 0.01, "slow": 0.00}}
```

### Example: predict by product code

```bash
curl http://localhost:8003/predict/by-code/10000019
```

```json
{
  "code": "10000019",
  "name": "...",
  "found": true,
  "features": { "...": 0 },
  "demand":     {"label": "fast", "probabilities": {...}},
  "dead_stock": {"is_dead_stock": false, "probability": 0.0}
}
```

### Backend integration

The API is a stateless HTTP service — call it from any backend:

```python
import requests
r = requests.post("http://ml-api:8003/predict/batch", json={"codes": ["10000019", "1000006"]})
for item in r.json()["results"]:
    print(item["code"], item["demand"]["label"], item["dead_stock"]["probability"])
```

For Docker / Kubernetes: bind `0.0.0.0:8003`, mount `artifacts/` as a volume, and re-run the training scripts whenever the underlying CSVs are refreshed.

## Retraining

The models are deterministic given the input CSVs and `random_state=42`. To retrain after a data refresh:

```bash
python -m src.train_demand && python -m src.train_dead_stock && python -m src.evaluate
```

Artifacts are overwritten in place; restart the API to pick up the new models (the `lru_cache` in `predict.py` only loads at process start).

## Limitations

- **Snapshot data, no time dimension** — these models classify *current* demand state, not future demand. True demand forecasting (ARIMA / Prophet / LSTM) requires transaction-level data with timestamps; see the root `CLAUDE.md`.
- **Rule-derived labels** — labels are heuristics, not human ground truth. The robustness evaluation gives the most honest read on model quality.
- **Class imbalance for dead-stock** — ~5% positive rate; the model uses `GradientBoosting` and is evaluated on F1/AUC rather than accuracy.
