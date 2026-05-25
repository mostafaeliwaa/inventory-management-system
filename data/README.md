# `/data` — Datasets Reference

This folder holds every dataset used by the project. Two generations live here:

1. **Original aggregated snapshots** (`cleaned_*.csv`) — branch-level totals **without dates**. Used by the existing dashboards in `/analysis`.
2. **New transaction-level sales** (`merged_new_sales.csv` / `.parquet`) — receipt-level POS data **with dates**. This unlocks forecasting and is the focus of this README.

---

## 1. `merged_new_sales.csv` — the new transactional file

### What this file is

This is the **clean, merged version of the eight raw branch workbooks** in `/new data/*.xlsx`. Each row is one **receipt / POS transaction** at one of our branches, with the date it was posted. The file was produced by `cleaning/merge_new_data.py`.

| Property | Value |
|---|---|
| Rows | **53,819** transactions |
| Columns | 19 |
| Branches | **8** (Asyut, October, Maadi, Nakheel, Sheraton, Mehrem, MiniStore-Mehrem, Madinaty) |
| Date range | **2026-01-01 → 2026-05-24** (~144 unique days, ~141 days per branch) |
| Encoding | UTF-8 with BOM (opens correctly in Excel with Arabic) |
| Twin file | `merged_new_sales.parquet` — same data, smaller & faster for Python/Spark |

### Branches present in this file

| branch_code | branch_name (AR) | branch_name_en | Rows |
|---:|---|---|---:|
| 2  | النخيل           | Nakheel           | 7,972 |
| 4  | المعادى          | Maadi             | 7,417 |
| 5  | أكتوبر           | October           | 5,846 |
| 6  | شيراتون          | Sheraton          | 4,946 |
| 7  | مينى ستور - محرم | MiniStore_Mehrem  | 6,427 |
| 8  | محرم             | Mehrem            | 5,164 |
| 9  | اسيوط            | Asyut             | 9,774 |
| 13 | مدينتى           | Madinaty          | 6,273 |

> The original aggregated `cleaned_*.csv` files mention 13 branches; the new dump covers 8 of them. Five branches (e.g. التجمع, العباسيه, الاداره) are not included in this batch.

### Column dictionary

Every column, what it means, and what the team should do with it.

| Column | Type | Arabic source | Meaning |
|---|---|---|---|
| `posting_date` | date | `تاريخ الترحيل` | The day the transaction was posted to the system. **This is the time index for forecasting.** Date-only (time stripped). |
| `branch_code` | int | `كود المخزن` | Internal numeric code of the branch / store. **Use this as the join/group key** (it's stable across spelling variants). |
| `branch_name` | text (AR) | `اسم المخزن` | Branch name as it appears in the source file (Arabic). |
| `branch_name_en` | text | derived | Standardised English branch name (e.g. `Maadi`, `Nakheel`). Use this in charts/labels. |
| `tx_type` | text (AR) | `نوع الحركة` | Transaction type as written in the source: `نقدى` (cash), `بطاقات إئتمان` (credit card), `آجل` (deferred / on-account). |
| `tx_type_en` | text | derived | English: `cash`, `card`, `credit`, or `other`. Use this in code. |
| `doc_no` | int | `رقم المستند` | Receipt / document number assigned by the POS for that branch+day. Not globally unique. |
| `system_serial` | int | `مسلسل الحاسب` | Computer serial counter for the transaction. Mostly mirrors `doc_no`. |
| `total_value` | float (EGP) | `القيمة` | **Gross transaction value (incl. tax & discount).** This is the primary revenue figure. Can be **negative** (refund / `مردود`). |
| `net_value` | float (EGP) | `قيمة الحركة` | Net value of the movement (after discount, before/after tax depending on POS config). Very close to `total_value` for most rows. |
| `cash_paid` | float (EGP) | `المدفوع نقدي` | Amount paid in cash for this transaction. `0` for card/credit rows. Sum of this per day = daily cash drawer. |
| `returns` | float (EGP) | `مردود` | Amount returned by the customer in this transaction. Usually `0`; non-zero means a refund/exchange leg. |
| `discount` | float (EGP) | `خصم / فاتورة` | Discount applied to this invoice. |
| `sales_tax` | float (EGP) | `ضريبة مبيعات` | Sales tax / VAT amount embedded in the transaction. |
| `salesperson` | text | `المندوب` | Sales rep / cashier who entered the transaction, with branch hint (e.g. `يوسف - اسيوط`). Useful for staff-level analytics; sparse in some files. |
| `source_file` | text | derived | The original Excel file the row came from (e.g. `اسيوط.xlsx`). Lets you trace any row back to its raw source. |
| `service_fee` | float | `خدمة` | Service charge field. **Mostly null** — kept for completeness, do not rely on it. |
| `customer_name` | text | `اسم العميل` | Customer name / channel tag (often `فيزا`, `انستا`, etc., used by the POS to mark the payment channel rather than a real person). Sparse. |
| `customer_code` | int | `كود العميل` | Numeric code paired with `customer_name`. Sparse. |

### How to read a row

A typical row says:
> *On `posting_date`, branch `branch_name_en` posted transaction `doc_no` of type `tx_type_en`. The customer was charged `total_value` EGP, of which `cash_paid` EGP was in cash, `discount` was discounted, and `sales_tax` was tax. The sale was rung up by `salesperson`.*

### Worked example

```text
posting_date     : 2026-01-01
branch_code      : 9
branch_name_en   : Asyut
tx_type_en       : card
doc_no           : 1
total_value      : 300
net_value        : 300
cash_paid        : 0
discount         : 0
sales_tax        : 0
salesperson      : يوسف - اسيوط
source_file      : اسيوط.xlsx
```
→ On 2026-01-01, Asyut (branch 9) sold receipt #1 for 300 EGP, all paid by Visa card, by salesperson Youssef.

### Cleaning already applied (so you don't have to)

The merge script (`cleaning/merge_new_data.py`) performs the following:

1. **Renamed** Arabic column headers to English.
2. **Dropped subtotal rows** — the `اجمالي المخزن` lines that the source includes after each branch group (no date, no branch). About 1,500 rows removed.
3. **Parsed dates** (`posting_date`) and stripped the time component.
4. **Normalised branch names** — `المعادي` / `المعادى` → `Maadi`, etc. → `branch_name_en`.
5. **Mapped transaction types** to English in `tx_type_en`.
6. **Coerced numerics**; missing money fields → `0`.
7. **Dropped 25 corrupted Sheraton rows** on **2026-02-06 / 2026-02-07** (values from 10⁶ up to ~3 billion EGP — a POS-sync error). Sanity cap: `|value| > 100,000 EGP`.
8. **Deduplicated** exact repeats on `(branch_code, posting_date, doc_no, total_value)`.
9. **Reordered** columns and **tagged each row** with its `source_file`.

### Known data caveats — please read before modelling

- **No SKU / product information.** This dataset is **receipt-level only**; there is no product code or product name in the source workbooks. SKU-level demand forecasting is **not possible** with this file alone. To forecast at the product level you must collect the line-item export.
- **Negative `total_value` is legitimate.** Refund/return rows show as negative. Examples: Asyut −20,000, Nakheel −3,330 — keep them, they are real reversals.
- **`customer_name` is often the payment-channel tag** (e.g. `فيزا`, `انستا`), not a real customer. Treat it as a dimension, not a customer key.
- **Salesperson labels are free-text** and sometimes carry an Arabic branch suffix (e.g. `يوسف - اسيوط`) — split on `-` if you need clean rep names.
- **Sheraton is now sane** (post-clean). Before cleaning, total revenue was reported as 626 M EGP because of 25 corrupt rows; after the filter it's ~818 k EGP, in line with the other branches.

### Quick sanity stats (after cleaning)

| Branch | Rows | Total revenue (EGP) | Mean ticket | Max ticket |
|---|---:|---:|---:|---:|
| Asyut | 9,774 | 1,311,030 | 134 | 20,030 |
| Maadi | 7,417 | 1,779,975 | 240 | 10,305 |
| Madinaty | 6,273 | 1,692,845 | 270 | 5,065 |
| Mehrem | 5,164 | 937,730 | 182 | 3,670 |
| MiniStore_Mehrem | 6,427 | 1,048,530 | 163 | 3,120 |
| Nakheel | 7,972 | 2,083,675 | 261 | 6,040 |
| October | 5,846 | 1,251,380 | 214 | 8,450 |
| Sheraton | 4,946 | 817,870 | 165 | 3,760 |

### Reading the file in Python

```python
import pandas as pd
df = pd.read_csv("data/merged_new_sales.csv", parse_dates=["posting_date"])
# or, faster:
df = pd.read_parquet("data/merged_new_sales.parquet")
```

For Excel: just double-click `merged_new_sales.csv` — UTF-8 BOM ensures Arabic renders.

---

## 2. What this data lets us predict

With one row per receipt and a real date column, we can finally do **time-series forecasting**. Recommended targets, in order of business value:

| Predict | Horizon | Business use |
|---|---|---|
| **Daily revenue per branch** (sum of `total_value`) | 7–30 days | Cash-flow planning, manager KPI targets |
| **Daily transaction count per branch** | 7–14 days | Shift scheduling, cashier headcount |
| **Cash share vs. card share per branch** | 7 days | Cash-pickup frequency, treasury planning |
| **Weekly returns amount** | 1–4 weeks | Quality / anomaly alerts |
| **Average ticket size per branch** | 14 days | Promo / pricing evaluation |

**Limitations:** ~141 days × 8 branches is enough for branch-level forecasting (LightGBM, Prophet, SARIMA), but **not** for deep-learning models (LSTM/TFT). And again, **no SKU forecasting** until product-level exports arrive.

---

## 3. Original aggregated files (no dates — pre-existing)

These are kept for backwards compatibility with `/analysis/EXP.ipynb` and the dashboards already published in `/analysis/output/`. **They are snapshots without time** and cannot be used for forecasting.

| File | Records | Contents |
|---|---:|---|
| `cleaned_sales_stock.csv` | 2,428 products | Product codes/names, sales & stock by branch |
| `cleaned_pricing.csv` | products | Cost & retail price, category |
| `cleaned_suppliers.csv` | suppliers | Supplier info |
| `cleaned_warehouse.csv` | warehouse | Inventory levels by warehouse |

For the legacy ETL that produced the PostgreSQL schema from these files, see [`/cleaning/`](../cleaning/) and the legacy notes at the bottom of this file.

---

## 4. How to regenerate `merged_new_sales.csv`

Whenever a new monthly drop lands in `/new data/`, simply re-run:

```bash
python cleaning/merge_new_data.py
```

The script will:
1. read every `.xlsx` in `/new data/`,
2. apply the same cleaning rules listed above,
3. overwrite `data/merged_new_sales.csv` and `.parquet`,
4. print per-branch row counts and a list of any rows it dropped as corrupted.

---

## 5. Legacy ETL note (PostgreSQL normalisation)

The pre-existing ETL pipeline (`etl_pipeline.py`) loads the aggregated `cleaned_*.csv` files into PostgreSQL with this schema: `products`, `suppliers`, `product_suppliers`, `branches`, `branch_sales`, `branch_stock`, `warehouses`, `warehouse_stock`, `pricing`. It does **not** consume `merged_new_sales.csv` — that file is intended for the forecasting pipeline.

### Database Schema Diagram

![Inventory Management Schema](inventory-management.png)
