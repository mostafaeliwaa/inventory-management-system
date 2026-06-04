# Forecasting Module — Plan

**Branch:** `forecasting`
**Target:** Daily revenue per branch (sum of `total_value`)
**Horizon:** 7 days ahead
**Models compared:** Seasonal-naive baseline · Prophet (per branch) · LightGBM (global, lag features)

---

## 1. Why this scope

Data we have (`data/merged_new_sales.csv`):
- ~144 days × 8 branches of receipt-level POS, 2026-01-01 → 2026-05-24
- Receipt-level only — **no SKU/product key**, so SKU-level forecasting is impossible
- Refund rows (negative `total_value`) are legitimate and are netted into daily revenue

Forecasting fundamentals this data forces:
- Branch-level is the finest granularity available
- 144 daily points per branch is enough for **classical + light GBM**, too few for **LSTM/TFT/N-BEATS** (they will overfit)
- Weekly seasonality is the dominant signal; Egyptian holidays (Ramadan 2026: Feb 18 → Mar 19, Eid al-Fitr Mar 20–22) sit inside the training window and must be modeled

We pick **7-day horizon** because it matches the weekly cycle and keeps error bands honest given the short history.

---

## 2. Module layout

```
forecasting/
├── PLAN.md                          # this file
├── README.md                        # written last, after results are in
├── requirements.txt                 # prophet, lightgbm, holidays, scikit-learn, joblib, pandas, plotly
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # read merged_new_sales.parquet → daily branch panel
│   ├── features.py                  # lags, rolling means, calendar, Ramadan/Eid, weekend
│   ├── splits.py                    # walk-forward CV (expanding window, 7-day folds)
│   ├── metrics.py                   # MAE, MAPE, WAPE, sMAPE, vs-baseline lift
│   ├── models/
│   │   ├── baseline.py              # seasonal-naive (lag-7) + naive (lag-1)
│   │   ├── prophet_model.py         # one Prophet per branch, with holiday regressors
│   │   └── lgbm_model.py            # global LightGBM, branch as categorical
│   ├── train.py                     # fit all models on full history, save artifacts
│   ├── evaluate.py                  # run walk-forward CV, emit reports/results.json + plots
│   └── forecast.py                  # produce next-7-day forecast per branch (inference)
│
├── notebooks/
│   └── 01_eda_and_baseline.ipynb    # sanity-check the panel + baseline before modeling
│
├── artifacts/                       # serialized models (.joblib / .json), gitignored
├── reports/                         # results.json + per-branch plots, gitignored except results.json
└── api.py                           # FastAPI service on port 8004, /forecast/{branch}, /metrics
```

Convention follows the existing `chatbot/` and `analysis/` folders. The old `forcasting models/` folder (typo, empty) stays untouched for now; we don't reuse the name.

---

## 3. Data pipeline (`src/data_loader.py`, `src/features.py`)

### 3.1 Load → daily panel

```
read merged_new_sales.parquet
↓ filter: posting_date NOT NULL, branch_code IN known 8
↓ group by (branch_code, posting_date)
↓ agg: revenue = sum(total_value), tx_count = count(*), cash_share = sum(cash_paid)/sum(total_value)
↓ reindex each branch onto full date range (fill missing days with 0 revenue, 0 tx_count)
→ panel: ~144 days × 8 branches = ~1,152 rows
```

Output columns: `posting_date, branch_code, branch_name_en, revenue, tx_count, cash_share`.

### 3.2 Feature engineering (LightGBM only — Prophet handles its own)

Per (branch, date):
- **Lags:** revenue_lag_{1, 2, 3, 7, 14, 28}
- **Rolling means (shifted by 1 to avoid leakage):** revenue_rmean_{7, 14, 28}
- **Rolling std:** revenue_rstd_{7, 14}
- **Calendar:** dow (0=Mon), is_weekend (Fri=4 / Sat=5 in Egypt), day_of_month, week_of_month, month
- **Holiday flags:** is_ramadan, ramadan_day_index (1..30, else 0), is_eid_fitr, days_to_eid, is_eid_adha (if in window), is_public_holiday (Sinai Liberation Apr 25, Labour Day May 1, Revolution Day Jul 23 — only those in our window matter)
- **Branch encoding:** branch_code as `category` (LightGBM native handling)

Rows with NaN lags (first 28 days per branch) are dropped from training but kept for predicting the next horizon (they become tail history).

---

## 4. Validation: walk-forward CV (`src/splits.py`)

Expanding window, 7-day forecast horizon, refit at every fold.

```
Total days per branch: ~144
First train cutoff:    day 60   (enough for 28-day lags + warmup)
Step:                  7 days
Number of folds:       ~12   ((144 - 60) // 7)
```

For each fold `i`:
- Train: days [0, cutoff_i)
- Test: days [cutoff_i, cutoff_i + 7)
- Predict 7 days ahead in one shot (no rolling update — that's the deployment shape)

Report per-branch and overall: MAE (EGP), MAPE (%), WAPE (%), sMAPE (%), and **lift vs seasonal-naive** (the number that actually matters).

---

## 5. Models

### 5.1 Baseline (`models/baseline.py`)

- **Seasonal-naive:** ŷ_t = y_{t-7}. This is the bar to beat.
- **Naive:** ŷ_t = y_{t-1}. Sanity floor.

No hyperparameters. Fits per branch.

### 5.2 Prophet (`models/prophet_model.py`)

One model per branch (8 total). Use:
- `weekly_seasonality=True`, `yearly_seasonality=False` (we have 5 months), `daily_seasonality=False`
- Holidays dataframe: Ramadan (as a 30-day window), Eid al-Fitr (3 days), Egyptian public holidays in window
- `seasonality_mode='multiplicative'` (revenue scales with level; will verify with `additive` in CV)
- `changepoint_prior_scale=0.05` default, tune in [0.01, 0.1, 0.5] via CV

### 5.3 LightGBM (`models/lgbm_model.py`)

**One global model** across all 8 branches, branch as categorical feature. Pools data → effectively 8× more rows.

- `objective='regression_l1'` (MAE; robust to refund spikes)
- `num_leaves=31`, `learning_rate=0.05`, `n_estimators=500` with early stopping on a held-out tail
- `categorical_feature=['branch_code', 'dow']`
- Predict 7 days ahead: **recursive** strategy — predict day t+1, append, recompute lags, predict t+2, … Document the alternative (direct multi-output) but ship recursive first; it's simpler and competitive on short horizons.

---

## 6. Evaluation report (`src/evaluate.py`)

Outputs `reports/results.json`:

```json
{
  "config": {"horizon": 7, "target": "revenue", "folds": 12, "cv_start": "2026-03-02"},
  "per_branch": {
    "Asyut": {"baseline_wape": 0.31, "prophet_wape": 0.22, "lgbm_wape": 0.19, "winner": "lgbm"},
    ...
  },
  "overall": {"baseline_wape": 0.29, "prophet_wape": 0.21, "lgbm_wape": 0.18},
  "lift_vs_baseline": {"prophet": 0.27, "lgbm": 0.38}
}
```

And per-branch Plotly HTML plots in `reports/plots/{branch}.html` showing actual vs each model's forecast across folds.

---

## 7. Inference (`src/forecast.py`, `api.py`)

- `forecast.py`: load best model per branch (from `results.json`), produce next-7-day forecast from latest data, write `reports/next_7d.json`.
- `api.py` (FastAPI, **port 8004** — 8002 is chatbot, 8003 was the old ML models service):
  - `GET /forecast/{branch}` → next 7 days for one branch
  - `GET /forecast` → all 8 branches
  - `GET /metrics` → latest CV results
  - `GET /health`

---

## 8. Implementation order (each step is independently runnable)

1. **Scaffold:** create `__init__.py`, `requirements.txt`, empty module skeletons
2. **`data_loader.py`** + a quick notebook cell verifying the panel shape and date completeness
3. **`features.py`** + unit-sanity (no NaN lag leakage, holiday flags hit the right days)
4. **`splits.py`** + **`metrics.py`** + **baseline** — get the bar-to-beat number first
5. **Prophet** model + plug into walk-forward, compare to baseline
6. **LightGBM** model + plug into walk-forward, compare to both
7. **`evaluate.py`** end-to-end → write `results.json`, render plots
8. **`forecast.py`** — pick winner per branch, produce next-7-day forecast
9. **`api.py`** on port 8004
10. **`README.md`** with the headline numbers and how to run
11. **EDA notebook** (`notebooks/01_eda_and_baseline.ipynb`) — optional, can be done after step 4

Stop after step 4 to review baseline numbers before investing in Prophet/LightGBM. If seasonal-naive WAPE is already <15%, the fancy models may not earn their complexity.

---

## 9. What's out of scope (explicitly)

- SKU/product-level forecasting — requires line-item exports we don't have
- Deep learning (LSTM/TFT/N-BEATS) — 144 days is not enough
- Probabilistic intervals from LightGBM (quantile regression) — defer to v2
- Auto-retraining schedule / Airflow — manual `python -m forecasting.src.train` for now
- Hooking forecasts into the existing Supabase dashboard — separate ticket

---

## 10. Open questions for the user

1. **Refunds policy:** model raw `total_value` (refunds net out naturally) or filter them out (refund-free demand signal)? Default: raw, net.
2. **Missing days:** treat as 0 revenue (assumes branch was closed/no sales recorded) or as missing (interpolate / mask out)? Default: 0, with a `was_open` flag feature.
3. **First branch to validate on:** Asyut (largest row count) or Nakheel (highest revenue)? Default: Asyut.
4. **Holiday list source:** hard-code Egyptian public holidays + Ramadan windows for 2026, or pull from the `holidays` Python package? Default: `holidays` package with `country='EG'`, plus a manual Ramadan/Eid override since `holidays` doesn't always carry Islamic dates correctly.
