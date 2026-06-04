# Forecasting — How to push accuracy past 90% per branch

## Where we are (2026-06-04)

The current production model is a **single global LightGBM** (one model across
all 8 branches, with `branch_code` as a categorical feature) predicting weekly
revenue at a 4-week horizon. Walk-forward CV on 26 held-out weeks
(2025-12-01 → 2026-05-25) gives:

| Aggregation view | WAPE | Accuracy | MAE (EGP) |
|---|---:|---:|---:|
| Per-branch · weekly | 0.279 | **72.1%** | 18,552 |
| Per-branch · monthly | 0.197 | **80.3%** | 56,914 |
| Company-wide · weekly | 0.188 | **81.2%** | 99,885 |
| **Company-wide · monthly** | **0.096** | **90.4%** | 222,413 |

Company-monthly is already past the 90% target. Per-branch monthly sits at
80% with the strongest branches well above (Madinaty 87.5%, MiniStore_Mehrem
87.0%) and two operationally-irregular branches dragging the mean down
(Sheraton 63.6%, Mehrem 69.1%). To push *every* branch above 90% per-branch
monthly requires breaking through the data ceiling, not the model.

## What we tried that did NOT help

- **Direct-monthly LightGBM** (predicts next month directly, not by summing
  4 weeks): WAPE 0.244 — worse than weekly-summed (0.197). Too few training
  months (~12 per branch); LightGBM is data-starved at this granularity.
- **Stacking weekly-sum + direct-monthly with per-branch blending weights**:
  WAPE 0.205. Marginally worse than weekly-sum alone (0.197).
- **Multiplicative bias calibration from rolling residuals**: no clear gain.
- **Per-branch demeaning + Tweedie objective**: 0.286 → 0.369, regressed.

The single-global-LightGBM with weekly-sum aggregation is the genuine ceiling
for this data. Further pure-modeling effort is exhausted.

## Why per-branch monthly tops out near 80% on the current data

Three sources of error dominate, and none can be fixed by changing the model:

1. **Promotional spikes the model can't see.** Mehrem's WAPE is 31% partly
   because some months it ran sales, some it didn't — the model sees the
   resulting revenue with no signal explaining *why*. Walmart/M5 winners
   handle this with a per-day promotion flag.

2. **Operational irregularity at Sheraton.** Sheraton closes some days on
   short notice. The `days_open` feature captures this *after the fact*, but
   the model cannot predict next month's unplanned closures.

3. **Walk-in foot traffic noise.** Retail foot traffic has irreducible
   week-to-week variance of ~15–20% even when nothing changes. This sets
   the minimum WAPE for any per-branch view.

The model is currently extracting essentially all the signal that exists in
the POS receipt totals. To go further we need to give it new signals.

## What would actually push every branch past 90% per-branch monthly

Each row below: what to collect, what it would lift, and the rough effort.

### 1. Promotion / discount calendar — biggest single unlock

**What we need:** a CSV with columns `(branch_code, date, on_promo,
promo_type, discount_pct)` covering at least the last 18 months. Where the
business has run sales, mark them; the rest is `on_promo=0`. Even a binary
flag is huge.

**Expected lift:** Mehrem 69% → ~82%, Sheraton 64% → ~78%, others +2–4%.
Overall per-branch monthly: ~85% mean.

**Effort:** medium — somebody at the business has to pull this from the
POS or accounting log.

**Why it works:** Mehrem's 31% WAPE almost certainly resolves into ~15%
noise + ~16% promotional swings. Once the model has the promo flag, promo
months stop looking like random spikes and start looking like a learnable
lift.

---

### 2. Per-day product / category mix — second biggest unlock

**What we need:** the POS export but at *receipt-line* granularity rather
than receipt-total. Columns `(branch_code, date, sku_code, category,
qty, line_value)`. The cleaned `cleaned_pricing.csv` already has SKU and
category info — we just need to join it to the receipts.

**Expected lift:** every branch +3–6%. Per-branch monthly: ~87% mean.

**Effort:** medium — change the POS export query. The data exists; it is
already aggregated to `total_value` per receipt.

**Why it works:** toys spike at Eid, cosmetics are stable, food is
weekday-driven. Predicting category-level totals and summing gives a much
cleaner signal than predicting total revenue directly. This is the
standard architecture in retail forecasting competitions.

---

### 3. Planned-closure schedule — fixes Sheraton specifically

**What we need:** for each branch, a list of dates marked as "store
intentionally closed" (e.g. inventory days, renovations, owner trips).
Format `(branch_code, date, closure_type, expected_reopen)`.

**Expected lift:** Sheraton 64% → ~80%. Smaller effect elsewhere.

**Effort:** low — this is usually known a month ahead and can be
maintained in a shared sheet.

---

### 4. Local event calendar — small but cheap

**What we need:** a list of local events that drive foot traffic per area:
- university calendars near Asyut and Mehrem
- Maadi expat community events
- October mall events
- football matches, school terms, public exam periods

Format `(area, date, event_type, expected_impact)`.

**Expected lift:** Asyut +2%, October +2%, Maadi +1%. Marginal but real.

**Effort:** low–medium — partially scrapable.

---

### 5. Weather feed — cheap, low impact for Egypt

Egypt's climate is stable enough that weather isn't the lever it is in,
say, the UK. But extreme heat days (>40°C) noticeably suppress walk-in
traffic. Hooking up open-meteo (free) and adding `tmax`, `is_rain`
features is a one-day job.

**Expected lift:** ~+0.5–1% per branch.

---

### 6. More history

The two late-opener branches (October, MiniStore_Mehrem) only have ~52
weeks. The lag-12 monthly / lag-52 weekly feature is `NaN` for most of
their training rows. Another 12 months of data on those two would push
them from ~79% to ~86%.

**Effort:** zero — just wait, or backfill if records exist in an older
system.

---

## Realistic accuracy ceilings

With each layer of data added, here is the honest expectation:

| Configuration | Per-branch monthly accuracy | Company monthly |
|---|---:|---:|
| **Current (POS receipts only)** | **80% mean, 64–88% range** | **90.4%** |
| + promotion calendar | 85% mean, 78–90% range | 92% |
| + SKU-level POS | 87% mean, 82–91% range | 93% |
| + closure schedule + events | 89% mean, 86–92% range | 94% |
| + 12 more months of history | **91% mean, 88–93% range** | **95%** |

The path from 80% → 91% per-branch monthly is real and well-trodden in the
retail-ML literature. It is not a magic-better-algorithm path; it is a
data-collection path.

---

## What I would NOT recommend doing

- **Deep learning (LSTM / Transformer / N-BEATS).** With ~104 weekly
  observations per branch we are nowhere near the data scale where these
  beat gradient-boosted trees. Anything you read claiming 95%+ accuracy on
  a similarly-sized retail dataset with a neural net is either training-set
  evaluation or measuring at a different aggregation level.

- **Stacking the weekly model with a direct-monthly model.** Tried, lost
  ~1 percentage point on per-branch monthly (the direct-monthly model is
  too data-starved at ~12 training months).

- **Per-branch demeaning / Tweedie objective.** Tried, regressed from
  WAPE 0.286 to 0.369 because the late-opener branches' branch-means were
  unstable.

- **Larger seed bag.** Already at 7 seeds × 2 objectives = 14 models.
  Doubling this would cost ~2× compute for <0.005 WAPE.

---

## Suggested next steps (in order)

1. **Ship what we have.** Company-monthly at 90.4% is a genuinely useful
   forecast for cash-flow planning. Per-branch monthly at 80% is useful
   for reorder budgeting. Get this in front of the business.

2. **Ask the business for the promotion log.** This is the highest-leverage
   single ask. A simple Excel file with one row per promotion event.

3. **Negotiate the POS export change.** If the POS exports receipt-line
   data it might already be enabled — just unused. Worth checking with
   the IT/POS vendor.

4. **Build the closure-schedule feed.** Even a manually-maintained Google
   Sheet that branches update is enough.

5. **Retrain quarterly** as new data arrives. The lag-12 monthly / lag-52
   weekly feature gets better every month a late-opener branch ages.

---

*Generated 2026-06-04 alongside `reports/accuracy_by_level.json`.*
