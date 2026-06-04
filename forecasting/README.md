# Forecasting Module

Branch-revenue forecasts at two granularities, built from `data/merged_new_sales.csv`
(receipt-level POS, 8 branches, 144 days, 2026-01-01 → 2026-05-24).

## Two products, two horizons

| Product | Target | Horizon | Best WAPE | Use case |
|---|---|---|---:|---|
| **Daily**  | daily revenue per branch  | 7 days  | **0.479** | Daily ops, staffing, today's cash |
| **Weekly** | weekly revenue per branch | 4 weeks | **0.229** | Cash-flow planning, reorder, manager KPIs |

Weekly is **~52% more accurate** than daily because aggregating across a week averages out random walk-in noise while keeping the trend + Ramadan signal intact. The production weekly model is a 3-model ensemble (Prophet + LightGBM + naive_mean4).

---

## Daily — walk-forward CV (12 folds, 7-day horizon)

| Model | WAPE | MAE (EGP) | vs baseline |
|---|---:|---:|---:|
| seasonal-naive (baseline) | 0.623 | 6,472 | — |
| naive | 0.708 | 7,361 | worse |
| Prophet (additive, tuned per branch) | 0.525 | 5,460 | −15.8% |
| LightGBM (recursive, log target) | 0.518 | 5,384 | −16.9% |
| LightGBM direct (per-horizon) | 0.534 | 5,552 | −14.3% |
| **Ensemble (Prophet + LGBM mean)** | **0.479** | **4,981** | **−23.1%** |
| smart_ensemble (CV-best per branch) | 0.506 | 5,256 | −18.8% |

All 8 branches beat baseline. Asyut improved most (0.846 → 0.623).

## Weekly — walk-forward CV (8 folds, 4-week horizon, 21 weekly obs per branch)

| Model | WAPE | MAE (EGP) |
|---|---:|---:|
| naive lag-1 | 0.369 | 27,688 |
| naive mean-4 (baseline) | 0.314 | 23,506 |
| Prophet weekly | 0.316 | 23,709 |
| LightGBM weekly (direct) | 0.355 | 26,590 |
| Ensemble weekly (P + L) | 0.250 | 18,543 |
| **Ensemble3 weekly (P + L + naive_mean4)** | **0.229** | **16,789** |

`ensemble3` is the **production** weekly model — averaging Prophet, LightGBM, and naive_mean4 trims the variance of each individual model and beats the 2-model ensemble by ~8% relative.

### Per-branch weekly WAPE (ensemble3) vs daily

| Branch | Weekly | Daily |
|---|---:|---:|
| Sheraton | 0.154 | 0.573 |
| Madinaty | 0.167 | 0.434 |
| Mehrem | 0.182 | 0.374 |
| Maadi | 0.220 | 0.492 |
| October | 0.223 | 0.410 |
| Nakheel | 0.271 | 0.487 |
| MiniStore_Mehrem | 0.283 | 0.454 |
| Asyut | 0.295 | 0.623 |

**All 8 branches finish below 0.30 WAPE weekly.** Sheraton (the hardest branch on the daily product, WAPE 0.573) becomes the *most accurate* weekly branch at 0.154 — a 73% relative improvement.

---

## Why this stack

- **8 branches × 144 days** of receipt-level POS is enough for classical + light-gradient-boosting, **not** enough for LSTM/TFT.
- Strong **weekly cycle** plus the Ramadan/Eid 2026 shock (Feb 18 – Mar 22) sits in the middle of the training window — both products model it explicitly.
- **No SKU/product key** in the source receipts → branch-level only.

## What's in the module

```
forecasting/
├── PLAN.md / README.md
├── requirements.txt
├── api.py                          # FastAPI on port 8004
├── src/
│   ├── data_loader.py              # merged_new_sales → daily (branch, day) panel
│   ├── features.py                 # lags, calendar, Ramadan/Eid shape, cross-target lags
│   ├── splits.py / metrics.py
│   ├── models/
│   │   ├── baseline.py             # seasonal-naive + naive
│   │   ├── prophet_model.py        # per-branch additive Prophet
│   │   ├── lgbm_model.py           # global LGBM, recursive
│   │   └── lgbm_direct.py          # global LGBM, one model per horizon day
│   ├── evaluate.py                 # daily walk-forward CV → results.json
│   ├── forecast.py                 # next-7-day daily forecast → next_7d.json
│   └── weekly.py                   # full weekly pipeline (aggregation + CV + forecast)
├── reports/
│   ├── results.json                # daily CV
│   ├── results_weekly.json         # weekly CV
│   ├── next_7d.json                # latest 7-day forecast
│   ├── next_4w.json                # latest 4-week forecast
│   └── plots/*.html                # one per-branch chart (daily models)
└── artifacts/                      # reserved for serialized models
```

## How to run

```bash
# from the repo root
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r forecasting/requirements.txt

# Daily pipeline
python -m forecasting.src.evaluate     # CV → reports/results.json + plots
python -m forecasting.src.forecast     # next 7 days → reports/next_7d.json

# Weekly pipeline (one script does both CV and forecast)
python -m forecasting.src.weekly       # → reports/results_weekly.json + next_4w.json

# Serve everything via FastAPI on port 8004
uvicorn forecasting.api:app --port 8004 --reload
# → http://localhost:8004/docs
```

## API endpoints (port 8004)

| Endpoint | Returns |
|---|---|
| `GET /health` | service status + which forecast files are present |
| `GET /forecast` | next 7 days for all 8 branches |
| `GET /forecast/{branch}` | next 7 days for one branch (case-insensitive) |
| `GET /forecast/weekly` | next 4 weeks for all 8 branches |
| `GET /forecast/weekly/{branch}` | next 4 weeks for one branch |
| `GET /metrics` | daily CV results |
| `GET /metrics/weekly` | weekly CV results |

## Latest forecast snapshots

**Daily (May 25–31, 2026):** company total ~495K EGP. Top branch Nakheel 90K.
**Weekly ensemble3 (May 25 – Jun 15, 2026):** company total ~2.27M EGP over 4 weeks. Top branch Nakheel 424K.

The API exposes `ensemble3` predictions as the headline `forecast` field on weekly endpoints; the individual `prophet`, `lgbm`, `naive_mean4`, and 2-model `ensemble` arrays are still returned alongside for transparency.

## What's modeled / not modeled

**Modeled:**
- Weekly seasonality (Fri/Sat weekend in Egypt) — daily product only
- Ramadan 2026 (Feb 18 – Mar 19) with first-10 / mid-10 / last-10 / last-3 day shape features
- Pre-Ramadan + pre-Eid shopping weeks
- Eid al-Fitr (Mar 20–22) with ±1 day window
- Egyptian public holidays in window: Coptic Christmas, Revolution Day, Sinai Liberation, Labour Day
- **Egyptian payday window** (25th of month → 5th of next) — `payday_days` & `payday_intensity` regressors (weekly)
- **School holiday windows** (mid-year break, end-of-year break) — `school_holiday_days` & `school_holiday_intensity` (weekly)
- **Intra-week volatility** (`intraweek_std`) — captures branches that swing harder day-to-day
- **Peer median lag** — cross-branch signal: median of *other* branches' last week as a regressor
- Refund spikes dampened (winsorized at 1st percentile per branch in LGBM)
- Cross-target lags: `tx_count`, `cash_share`, `was_open` — extra signal for LGBM

**Not modeled (out of scope):**
- SKU/product-level demand — receipts don't carry product codes
- Weather, promo calendar, payday weeks — no external regressors yet
- Probabilistic intervals — point forecasts only

## Known limitations

- **Daily WAPE ~48% is high in absolute terms** but normal for small-store retail with 144 days of history dominated by Ramadan. Aggregating to weekly drops it to ~23%.
- **Monthly forecasting is not supported** — only 5 monthly observations per branch is below any reasonable modeling threshold. Sum the weekly forecast (~30 days) for a monthly-style number instead.
- **LGBM direct underperformed LGBM recursive on daily** — training data splits 7 ways, each per-horizon model is starved at this scale.
- **smart_ensemble (per-branch CV-winner) overfit CV** vs the steady Prophet+LGBM average — same pattern observed when we tried per-branch weighted weekly ensembles. The simple equal-weight `ensemble3` is the production choice.
- **Realistic floor ~0.18-0.20 WAPE weekly** with 21 weekly observations. Further improvements would require more history or external regressors (weather, promo calendar, payday dollar volume).

## Notes on dependencies

`prophet` and `lightgbm` are heavy installs. On Anaconda Windows behind a corporate proxy, `pip` may hit
SSL errors — use:
```
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org prophet lightgbm
```
