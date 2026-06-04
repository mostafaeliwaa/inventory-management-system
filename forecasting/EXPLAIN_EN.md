# Forecasting Module — Detailed Explanation (EN)

A walkthrough of **everything** we built in the `forecasting/` module of this graduation project, in the order it happened. Use it for the discussion / viva.

> **TL;DR — final result**
> - Daily 7-day forecast: WAPE **0.479** (ensemble Prophet + LightGBM)
> - **Weekly 4-week forecast: WAPE 0.229 (ensemble3 = Prophet + LightGBM + naive_mean4) — production model**
> - Best branch weekly: **Sheraton 0.154** (down from 0.573 daily)
> - All forecasts served via FastAPI on port 8004

For the Arabic version, see `EXPLAIN_AR.md`.

---

## 0. The data we started from

Receipt-level POS transactions for **8 branches** across Egypt, covering **144 days** (2026-01-01 → 2026-05-24). The data was already merged in `data/merged_new_sales.csv` (the file is named `merged_new_sales` because it's the cleaned, joined version of the raw sales dumps).

What's in each row: a receipt header — `posting_date`, `branch_code`, `branch_name_en`, `tx_count` (number of items), `revenue` (EGP), `cash_share` (cash vs card mix), `was_open` flag. **No SKU / product key** in the receipts — so we can only model **branch-level revenue**, not per-product demand.

**Why this matters:** with only 144 days × 8 branches and no product key:
- LSTM / Transformer-style sequence models are off the table — not enough data
- SKU-level forecasting is impossible
- Classical statistical models + lightweight tree boosters are the right tool

---

## Phase 1 — Daily product, baseline + first stack

### What we built

A daily-revenue-per-branch forecaster with a 7-day horizon (predict the next week, day by day). Module pipeline:

```
forecasting/src/
├── data_loader.py    → builds the (branch, day) panel from merged_new_sales.csv
├── features.py       → lags 1/2/3/7/14/28, rolling means/std, calendar, Ramadan shape
├── splits.py         → walk-forward CV (12 folds, expanding window, 7-day step)
├── metrics.py        → WAPE (headline), MAE, sMAPE
├── models/
│   ├── baseline.py       → seasonal_naive (y_{t-7}) and naive (y_{t-1})
│   ├── prophet_model.py  → per-branch additive Prophet with custom holidays
│   ├── lgbm_model.py     → global LightGBM with log1p target, recursive 7-day prediction
│   └── lgbm_direct.py    → global LightGBM, one model per horizon day (direct multi-output)
├── evaluate.py       → daily walk-forward CV → reports/results.json
└── forecast.py       → next 7 days → reports/next_7d.json
```

**Cross-validation:** *walk-forward* (also called *expanding window* / *time-series split*). Starting from day 60, we train on everything before, predict the next 7 days, slide 7 days forward, repeat. That gives **12 folds**. We never use future data to predict past data — same protocol as you'd see in real-world deployment.

**Headline metric: WAPE** (Weighted Absolute Percentage Error). Formula:

$$\text{WAPE} = \frac{\sum_t |y_t - \hat{y}_t|}{\sum_t |y_t|}$$

Why WAPE instead of MAPE: MAPE blows up when the true value is small (division by zero issue) — common in low-traffic branches like MiniStore_Mehrem. WAPE pools the numerator and denominator over the whole series, so it stays well-defined.

### Phase-1 results

| Model | WAPE | MAE (EGP) | vs baseline |
|---|---:|---:|---:|
| seasonal-naive (baseline) | 0.623 | 6,472 | — |
| naive (y_{t-1}) | 0.708 | 7,361 | worse |
| Prophet | 0.525 | 5,460 | −15.8% |
| LightGBM (recursive) | 0.518 | 5,384 | −16.9% |
| LightGBM (direct) | 0.534 | 5,552 | −14.3% |
| **Ensemble (Prophet + LGBM mean)** | **0.479** | **4,981** | **−23.1%** |
| smart_ensemble (per-branch CV-winner) | 0.506 | 5,256 | −18.8% |

Ensemble wins. Two interesting findings:

1. **LightGBM recursive > LightGBM direct.** Direct trains 7 separate models (one per horizon day) — at 144 days × 8 branches that splits the training data 7 ways, each model gets starved.
2. **smart_ensemble overfit CV.** We tried picking the best model *per branch* based on CV performance — but the per-branch CV scores are unstable at this sample size, so the picker latches onto noise. Plain averaging beats it.

**Per-branch:** all 8 branches beat baseline. Asyut improved most (0.846 → 0.623). But the absolute WAPE — ~48% — is still high. That motivated phase 2.

### Features we engineered

- **Calendar:** day-of-week one-hots, day-of-month, week-of-year, month
- **Lags:** revenue at t-1, t-2, t-3, t-7, t-14, t-28
- **Rolling stats:** mean & std over windows of 7, 14, 28 days
- **Ramadan shape (2026-02-18 → 2026-03-19):** `is_ramadan`, `ramadan_first10`, `ramadan_mid10`, `ramadan_last10`, `ramadan_last3` — Ramadan retail isn't flat; the last 10 days (and especially last 3) spike for Eid shopping
- **Eid al-Fitr (2026-03-20 → 22):** `is_eid`, ±1 day window
- **Egyptian public holidays:** Coptic Christmas, Revolution Day, Sinai Liberation, Labour Day
- **Pre-Ramadan / pre-Eid shopping weeks:** flags for the 7 days before each event
- **Cross-target lags:** lag-1 of `tx_count`, `cash_share`, `was_open` — these give the LGBM extra signal that's correlated with revenue but isn't revenue itself (so it dodges the leakage problem when forecasting recursively)

In Prophet's case, these are passed as **regressors** + **custom holidays**. In LGBM's case they're just additional columns in the training matrix.

---

## Phase 2 — Switch to weekly granularity

### Why

48% WAPE on daily is *too high to plan on*. Two options:
1. Get more data — out of scope; we have what we have.
2. Aggregate to a coarser time grain — sum daily revenue into ISO weeks.

Aggregation does two helpful things at the same time:
- **Averages out walk-in noise.** A daily forecast has to predict whether Monday or Tuesday gets the customer. A weekly forecast just asks: how much do you make this week? The within-week shuffle becomes irrelevant.
- **Keeps the long-running signal.** The trend, Ramadan, payday cycles — all of these are still in the weekly aggregate. Only the high-frequency noise disappears.

The cost: 144 days → ~21 weekly observations per branch. That's *very few* — we have to be careful with feature count.

### What we built

`forecasting/src/weekly.py` — a single script that does it all:

1. **`build_weekly_panel()`** — aggregates the daily panel to ISO weeks (Monday-start). Each row has:
   - `revenue` (sum over the week)
   - `tx_count` (sum)
   - `ramadan_days`, `eid_days` (count of Ramadan / Eid days in the week)
   - `payday_days` ✱ — *(added in phase 3 — mentioned here for context)*
   - `school_holiday_days` ✱
   - `intraweek_std` ✱ — std of daily revenue inside the week — measures how bumpy the week was
2. **`weekly_prophet()`** — weekly Prophet. We use `changepoint_prior_scale=0.1` *fixed* (not per-branch tuned) because per-branch tuning over-fit — see "what didn't work" below.
3. **`weekly_lgbm()`** — LightGBM direct multi-output: 4 models, one per horizon week. ~22 features.
4. **`evaluate_weekly()`** — walk-forward CV: 8 folds, 4-week horizon, first train on 10 weeks.
5. **`next_4_weeks_forecast()`** — produces the next 4 weeks → `reports/next_4w.json`.

### Phase-2 result (before phase-3 enhancements)

| Model | WAPE | MAE (EGP) |
|---|---:|---:|
| naive lag-1 | 0.369 | 27,688 |
| naive mean-4 (baseline) | 0.314 | 23,506 |
| Prophet weekly | 0.353 | 26,450 |
| LightGBM weekly (direct) | 0.376 | 28,210 |
| **Ensemble weekly (P + L)** | **0.270** | **20,055** |

Weekly ensemble landed at **WAPE 0.270** — a **43% relative improvement over the daily ensemble (0.479)**, just by changing time granularity. Worth highlighting: individually, Prophet weekly and LGBM weekly are *worse* than naive_mean4 (the smarter weekly baseline). The ensemble *combination* still beats everything — the two models are wrong in different directions, and averaging them cancels noise.

### What didn't work in phase 2

- **Per-branch Prophet changepoint tuning.** Tried picking `cps ∈ {0.05, 0.1, 0.2}` per branch by CV — WAPE went **up** from 0.353 to 0.592, and Sheraton specifically went to 0.721. With only 21 weekly obs, the per-branch CV score is too noisy to pick a hyperparameter reliably. **Lesson:** when data is scarce, prefer global hyperparameters with light regularization over per-branch tuning.
- **Monthly aggregation (proposed but rejected).** We considered going one step further and aggregating to monthly. But 144 days = only 5 monthly observations per branch. That's below any reasonable modeling threshold. Decision: **weekly is the right floor for this dataset; monthly is not supported.** Sum 4 weekly forecasts if you need a monthly-style number.

---

## Phase 3 — Enhanced weekly: extra regressors + 3-model ensemble

### The motivation

At the end of phase 2 we had WAPE 0.270. Two branches were still rough — Asyut (0.350) and Sheraton (0.332) — because they're hard cases: Asyut has high refund volatility, Sheraton sits in a transient/hotel area with payday-sensitive customers. We asked: what *real-world signal* is missing from the model?

Answer: salary cycles and school holidays. Egyptian retail customers, especially in middle-income areas, shop heavily right after payday (25th of each month, with most cash flowing 25th → 5th). School holidays change foot traffic patterns. Neither was in the model.

### What we added

In `forecasting/src/weekly.py`:

1. **Payday window** — Egypt-specific: most salaries land 25th of each month, cash flow continues until ~5th of next. Two features:
   - `payday_days` — count of payday-window days in the week
   - `payday_intensity` — same thing normalized to 0..1 (used by LGBM as a continuous feature)

2. **School holidays** — mid-year break (~Jan 23–Feb 3, 2026) and end-of-year break (~Jun 1+). Two features: `school_holiday_days`, `school_holiday_intensity`.

3. **Intra-week volatility** — `intraweek_std` = std of daily revenue inside the week. Branches with high std are bursty (Asyut, MiniStore_Mehrem); branches with low std are steady (Madinaty, October). This lets the model condition its forecast on the *kind* of branch each row represents.

4. **Cross-branch (peer) signal: `peer_median_lag1`** — for each (branch, week), the median revenue across *other* branches in the previous week. This captures system-wide shocks (e.g., a hot weekend, an Eid bump) that affect every branch.

5. **Branch identity hint: `branch_mean_logrev`** — the mean of log-revenue for that branch up to (but not including) the current week. This gives the global LGBM a numeric "this is the size of this branch" feature.

### The 3-model ensemble

In phase 2 the ensemble was `mean(prophet, lgbm)`. In phase 3 we added a third model into the mix: `naive_mean4` — the mean of the last 4 weeks' revenue. Why?

Because naive_mean4 turned out to be **really hard to beat** at the per-branch level on this dataset — Prophet beats it on some branches, LGBM beats it on others, but neither beats it everywhere. Averaging all three pulls predictions toward the most accurate-on-average baseline while still letting Prophet and LGBM contribute trend + seasonality signal.

Formula: `ensemble3 = mean(prophet, lgbm, naive_mean4)`.

### Phase-3 final results

**Overall (8 branches, 8 folds):**

| Model | WAPE | MAE (EGP) |
|---|---:|---:|
| naive lag-1 | 0.369 | 27,688 |
| naive mean-4 | 0.314 | 23,506 |
| Prophet weekly | 0.316 | 23,709 |
| LightGBM weekly (direct) | 0.355 | 26,590 |
| ensemble (P + L) — phase-2 production | 0.250 | 18,543 |
| **ensemble3 (P + L + naive_mean4) — phase-3 production** | **0.229** | **16,789** |

**Per-branch (ensemble3 vs daily ensemble):**

| Branch | Daily WAPE | Weekly ensemble3 WAPE | Δ |
|---|---:|---:|---:|
| Sheraton | 0.573 | **0.154** | −73% |
| Madinaty | 0.434 | 0.167 | −61% |
| Mehrem | 0.374 | 0.182 | −51% |
| Maadi | 0.492 | 0.220 | −55% |
| October | 0.410 | 0.223 | −46% |
| Nakheel | 0.487 | 0.271 | −44% |
| MiniStore_Mehrem | 0.454 | 0.283 | −38% |
| Asyut | 0.623 | 0.295 | −53% |

Every branch finished below **0.30 WAPE** weekly. Sheraton — the worst on the daily product — became the *most accurate* branch weekly, with a 73% drop. That's the single biggest improvement and it's mostly attributable to payday + school-holiday regressors capturing what was previously labeled "noise" in Sheraton's daily series.

### What didn't work in phase 3

- **Per-branch weighted weekly ensembles.** Tried `smart_ensemble_weekly` (per-branch CV-winner among Prophet, LGBM, naive_mean4, ensemble3). Same overfit pattern as smart_ensemble on daily — the per-branch CV scores were too noisy with only 21 weekly obs. Plain equal-weight ensemble3 won. **Lesson confirmed twice now: with this much data, simple averaging > per-branch selection.**
- **Recommendation we *didn't* take: stacking / meta-learners.** Could in principle train a linear stacker on top of the 3 base models, but with 21 weekly obs that stacker has ~5–10 effective training points after CV — would absolutely overfit.

---

## API

All forecasts are served through FastAPI on **port 8004**:

```bash
uvicorn forecasting.api:app --port 8004 --reload
# Browse: http://localhost:8004/docs
```

| Endpoint | What it returns |
|---|---|
| `GET /health` | service status + which forecast files exist |
| `GET /forecast` | next 7 days for all branches (daily) |
| `GET /forecast/{branch}` | next 7 days for one branch |
| `GET /forecast/weekly` | next 4 weeks for all branches — **`forecast` field = ensemble3 (production)** |
| `GET /forecast/weekly/{branch}` | next 4 weeks for one branch |
| `GET /metrics` | daily CV results |
| `GET /metrics/weekly` | weekly CV results |

**Important production wiring:** the weekly endpoints have an `_annotate_weekly()` helper that **promotes ensemble3** to the headline `forecast` field. The raw `prophet`, `lgbm`, `naive_mean4`, and 2-model `ensemble` series are still in the response for transparency / sanity-checking — but downstream consumers should read `forecast` and `forecast_total_4w`.

The `/forecast/weekly` routes are declared *before* the dynamic `/forecast/{branch}` route — otherwise FastAPI would have matched `weekly` as a branch name and 404'd.

---

## What's modeled vs not modeled

### Modeled ✅

- Weekly seasonality (Fri/Sat = Egyptian weekend) — daily product only
- Ramadan 2026 (Feb 18 – Mar 19) with first-10 / mid-10 / last-10 / last-3 day shape features
- Pre-Ramadan + pre-Eid shopping weeks (7-day flags)
- Eid al-Fitr (Mar 20–22) with ±1 day window
- Egyptian public holidays: Coptic Christmas, Revolution Day, Sinai Liberation, Labour Day
- Egyptian payday window (25th → 5th of next month) — weekly only
- School holidays (mid-year, end-of-year) — weekly only
- Intra-week volatility — weekly only
- Cross-branch peer median lag — weekly only
- Cross-target lags (`tx_count`, `cash_share`, `was_open`) — LGBM only
- Refund spikes dampened via 1st-percentile winsorization per branch — LGBM only

### Not modeled (out of scope) ❌

- **SKU / product-level demand** — receipts don't carry product codes
- **Weather** — would need external data feed
- **Promo calendar / discount days** — not in the source data
- **Probabilistic intervals** — point forecasts only. Adding Prophet's uncertainty interval would be ~30 lines of code if needed for risk-aware planning.
- **External macro signals** — FX rate, inflation, fuel prices. Could matter for Asyut (more macro-sensitive) but no data source.

---

## Files in the module

```
forecasting/
├── EXPLAIN_EN.md           ← this file (English)
├── EXPLAIN_AR.md           ← Arabic version
├── PLAN.md                 ← original kickoff plan
├── README.md               ← module documentation (numbers, how-to-run)
├── requirements.txt
├── api.py                  ← FastAPI on port 8004
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── splits.py
│   ├── metrics.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── prophet_model.py
│   │   ├── lgbm_model.py
│   │   └── lgbm_direct.py
│   ├── evaluate.py         ← daily CV
│   ├── forecast.py         ← daily next-7-day forecast
│   └── weekly.py           ← weekly pipeline (CV + forecast, phase 2 & 3)
├── notebooks/
│   └── results.ipynb       ← visualizations of every metric, with methodology section
├── reports/
│   ├── results.json
│   ├── results_weekly.json
│   ├── next_7d.json
│   ├── next_4w.json
│   └── plots/
└── artifacts/              ← reserved for serialized models
```

---

## How to reproduce

```bash
# from repo root, after pip install -r forecasting/requirements.txt

# Daily pipeline
python -m forecasting.src.evaluate     # 12-fold CV → reports/results.json
python -m forecasting.src.forecast     # next 7 days → reports/next_7d.json

# Weekly pipeline (CV + forecast in one shot)
python -m forecasting.src.weekly       # → reports/results_weekly.json + next_4w.json

# Serve via FastAPI on port 8004
uvicorn forecasting.api:app --port 8004 --reload
```

---

## Discussion talking points

If asked **"why ensemble of three?"** → because no single model dominates per-branch, and equal-weight averaging is the safest combination given how few weekly observations we have. We tried smarter combinations (per-branch weighting); they overfit.

If asked **"why weekly and not daily?"** → daily forecasting at 144 days of history with no SKU codes hits a ~48% WAPE ceiling. Aggregation removes the within-week noise without losing the long-running signal. We get to ~23% WAPE for free.

If asked **"why not LSTM / Transformer?"** → sequence models need ~10× more data than we have. 21 weekly obs × 8 branches is firmly in the "classical + light-GBM" regime.

If asked **"what would lift the floor below 0.23 WAPE?"** → another 6–12 months of history would let the model learn yearly seasonality. Adding weather, promo calendar, or payday *volume* (not just window) data could shave a few more points. Realistic floor with the current data is ~0.18–0.20 WAPE.

If asked **"why is Sheraton the best now when it was the worst before?"** → the daily Sheraton series was dominated by walk-in randomness from a transient hotel-area customer base. At weekly resolution, that randomness averages out, and what's left is a payday-driven trend that the new `payday_*` regressors capture cleanly.

If asked **"why aggregate to weekly instead of training daily with more features?"** → we did both. Daily with all the same regressors still hits 0.479. The bottleneck isn't features — it's the signal-to-noise ratio of the target itself at the daily grain.

If asked **"is `naive_mean4` cheating in the ensemble?"** → no. It's a model. A simple one — predict next 4 weeks as the average of the last 4 weeks — but it has a well-defined training procedure (none) and inference procedure (compute the mean). It happens to be a strong baseline on this dataset because branch revenue is fairly stationary outside Ramadan, so the recent mean is a decent prior. Ensembling it with Prophet and LGBM is just acknowledging that for this data, the mean is hard to beat — and combining models that lean different ways reduces variance.

---

**Last updated:** 2026-05-27 — End of phase 3. Model is frozen pending user approval to commit.
