"""
Preprocess & merge new branch-level transaction data (2026-01 .. 2026-05).

Inputs : new data/*.xlsx  (one workbook per branch, Arabic headers)
Outputs:
  data/transactions_clean.csv   - cleaned row-per-transaction
  data/transactions_clean.parquet
  data/daily_sales_by_branch.csv  - daily aggregate ready for forecasting

Usage  : python cleaning/preprocess_new_sales.py
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "new data"
OUT_DIR = ROOT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Arabic -> English column map (every source column we care about)
COL_MAP = {
    "تاريخ الترحيل": "posting_date",
    "اسم المخزن": "branch_name",
    "كود المخزن": "branch_code",
    "القيمة": "total_value",         # gross incl. tax
    "قيمة الحركة": "net_value",
    "المدفوع نقدي": "cash_paid",
    "مردود": "returns",
    "خصم / فاتورة": "discount",
    "ضريبة مبيعات": "sales_tax",
    "خدمة": "service_fee",
    "نوع الحركة": "tx_type",
    "رقم المستند": "doc_no",
    "مسلسل الحاسب": "system_serial",
    "اسم العميل": "customer_name",
    "كود العميل": "customer_code",
    "المندوب": "salesperson",
}

# Canonical tx_type values (Arabic -> English)
TX_TYPE_MAP = {
    "نقدى": "cash",
    "بطاقات إئتمان": "card",
    "آجل": "credit",
}

# Branch-name normalisation (different spellings -> canonical)
BRANCH_NAME_MAP = {
    "اسيوط": "Asyut",
    "أكتوبر": "October",
    "اكتوبر": "October",
    "المعادى": "Maadi",
    "المعادي": "Maadi",
    "النخيل": "Nakheel",
    "شيراتون": "Sheraton",
    "مينى ستور - محرم": "MiniStore_Mehrem",
    "محرم": "Mehrem",
    "مدينتى": "Madinaty",
    "مدينتي": "Madinaty",
}

NUMERIC_COLS = [
    "total_value", "net_value", "cash_paid", "returns",
    "discount", "sales_tax", "service_fee",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_one(path: Path) -> pd.DataFrame:
    """Load one branch workbook and apply column renaming."""
    df = pd.read_excel(path)
    df = df.rename(columns=COL_MAP)
    df["source_file"] = path.name
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply common cleaning steps to a transaction frame."""
    # 1. Drop subtotal rows (no date / no branch)
    before = len(df)
    df = df.dropna(subset=["posting_date", "branch_code"]).copy()
    dropped = before - len(df)
    print(f"   dropped {dropped} subtotal/blank rows")

    # 2. Parse date
    df["posting_date"] = pd.to_datetime(df["posting_date"], errors="coerce")
    df = df.dropna(subset=["posting_date"])
    df["posting_date"] = df["posting_date"].dt.normalize()  # strip time

    # 3. Normalise branch names
    df["branch_name"] = df["branch_name"].astype(str).str.strip()
    df["branch_name_en"] = df["branch_name"].map(BRANCH_NAME_MAP).fillna(df["branch_name"])

    # 4. Coerce numerics and fill NaN with 0 (no charge of that type)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 5. Normalise tx_type
    if "tx_type" in df.columns:
        df["tx_type_en"] = df["tx_type"].map(TX_TYPE_MAP).fillna("other")

    # 6. Outlier flagging (do not drop — keep, but flag).
    #    Use a generous IQR rule per branch on absolute total_value.
    grp = df.groupby("branch_code")["total_value"]
    q1 = grp.transform(lambda s: s.quantile(0.25))
    q3 = grp.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    upper = q3 + 3 * iqr           # 3*IQR -> only extreme outliers
    lower = q1 - 3 * iqr
    df["is_outlier"] = (df["total_value"] > upper) | (df["total_value"] < lower)

    # 7. Drop high-null / low-value columns
    drop_cols = [c for c in ["service_fee", "customer_name", "customer_code"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # 8. De-duplicate exact dupes (same branch, date, doc_no, value)
    dup_keys = ["branch_code", "posting_date", "doc_no", "total_value"]
    dup_keys = [k for k in dup_keys if k in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=dup_keys)
    print(f"   dropped {before - len(df)} exact duplicates")

    return df.reset_index(drop=True)


def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions to daily-per-branch panel + calendar features."""
    # Only keep non-outlier rows for the aggregate target (configurable)
    work = df[~df["is_outlier"]].copy()

    g = work.groupby(["posting_date", "branch_code", "branch_name_en"], dropna=False)
    daily = g.agg(
        revenue=("total_value", "sum"),
        net_revenue=("net_value", "sum"),
        cash_revenue=("cash_paid", "sum"),
        returns_amt=("returns", "sum"),
        discount_amt=("discount", "sum"),
        tax_amt=("sales_tax", "sum"),
        tx_count=("total_value", "size"),
        avg_ticket=("total_value", "mean"),
    ).reset_index()

    # Transaction-mix shares
    mix = (
        work.groupby(["posting_date", "branch_code", "tx_type_en"])["total_value"]
            .sum()
            .unstack(fill_value=0.0)
            .add_prefix("rev_")
            .reset_index()
    )
    daily = daily.merge(mix, on=["posting_date", "branch_code"], how="left").fillna(0.0)

    # Reindex to full date range per branch (fill gaps with 0 sales)
    all_dates = pd.date_range(daily["posting_date"].min(),
                              daily["posting_date"].max(), freq="D")
    branches = daily[["branch_code", "branch_name_en"]].drop_duplicates()
    full_idx = (branches.assign(key=1)
                          .merge(pd.DataFrame({"posting_date": all_dates, "key": 1}), on="key")
                          .drop(columns="key"))
    daily = full_idx.merge(daily, on=["posting_date", "branch_code", "branch_name_en"], how="left")
    daily = daily.fillna(0.0)

    # ---- calendar features ---------------------------------------------
    d = daily["posting_date"]
    daily["dayofweek"] = d.dt.dayofweek            # Mon=0 ... Sun=6
    daily["is_weekend"] = d.dt.dayofweek.isin([4, 5]).astype(int)  # Fri+Sat in EG
    daily["day"] = d.dt.day
    daily["month"] = d.dt.month
    daily["week"] = d.dt.isocalendar().week.astype(int)
    daily["is_month_start"] = d.dt.is_month_start.astype(int)
    daily["is_month_end"] = d.dt.is_month_end.astype(int)

    # Ramadan 2026: Feb 18 – Mar 19. Eid al-Fitr: Mar 20-22.
    ramadan = (d >= "2026-02-18") & (d <= "2026-03-19")
    eid_fitr = (d >= "2026-03-20") & (d <= "2026-03-22")
    sham_elnessim = d == "2026-04-21"
    labor_day = d == "2026-05-01"
    daily["is_ramadan"] = ramadan.astype(int)
    daily["is_eid_fitr"] = eid_fitr.astype(int)
    daily["is_holiday"] = (sham_elnessim | labor_day | eid_fitr).astype(int)
    daily["days_since_payday"] = ((d - d.dt.to_period("M").dt.to_timestamp() -
                                   pd.Timedelta(days=24)).dt.days % 30)

    # ---- lag + rolling features (per branch) ---------------------------
    daily = daily.sort_values(["branch_code", "posting_date"]).reset_index(drop=True)
    by = daily.groupby("branch_code")["revenue"]
    for lag in (1, 7, 14, 28):
        daily[f"rev_lag_{lag}"] = by.shift(lag)
    for win in (7, 14, 28):
        daily[f"rev_roll_mean_{win}"] = by.shift(1).rolling(win).mean()
        daily[f"rev_roll_std_{win}"]  = by.shift(1).rolling(win).std()

    return daily


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Force UTF-8 stdout on Windows for Arabic logging
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    files = sorted(glob.glob(str(SRC_DIR / "*.xlsx")))
    if not files:
        raise SystemExit(f"No .xlsx files found in {SRC_DIR}")

    frames: list[pd.DataFrame] = []
    for f in files:
        path = Path(f)
        print(f"-> loading {path.name}")
        df = load_one(path)
        df = clean(df)
        print(f"   kept {len(df):>6} rows  |  branches: {df['branch_name_en'].unique().tolist()}"
              f"  |  date {df['posting_date'].min().date()} .. {df['posting_date'].max().date()}")
        frames.append(df)

    tx = pd.concat(frames, ignore_index=True)
    print(f"\nMerged transactions: {len(tx):,} rows, "
          f"{tx['branch_code'].nunique()} branches, "
          f"{tx['posting_date'].nunique()} unique days")
    print(f"Outliers flagged: {tx['is_outlier'].sum():,} ({tx['is_outlier'].mean():.2%})")

    # Save tx-level
    tx_csv = OUT_DIR / "transactions_clean.csv"
    tx_pq  = OUT_DIR / "transactions_clean.parquet"
    tx.to_csv(tx_csv, index=False, encoding="utf-8-sig")
    try:
        tx.to_parquet(tx_pq, index=False)
        print(f"wrote {tx_csv.name} and {tx_pq.name}")
    except Exception as e:
        print(f"wrote {tx_csv.name} (parquet skipped: {e})")

    # Daily panel
    daily = build_daily(tx)
    daily_csv = OUT_DIR / "daily_sales_by_branch.csv"
    daily.to_csv(daily_csv, index=False, encoding="utf-8-sig")
    print(f"wrote {daily_csv.name}  ({len(daily):,} rows, "
          f"{daily['branch_code'].nunique()} branches x {daily['posting_date'].nunique()} days)")
    print("\nSample of daily panel:")
    print(daily.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
