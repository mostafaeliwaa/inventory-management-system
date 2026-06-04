"""
Merge every Excel file in `new data/` into ONE clean CSV.

- Reads each branch workbook
- Renames Arabic columns to English
- Drops subtotal rows (rows with no date/branch)
- Parses dates
- Normalises branch names and transaction types
- Adds `source_file`
- Writes data/merged_new_sales.csv  (+ .parquet if pyarrow available)

Run: python cleaning/merge_new_data.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "new data"
OUT_DIR = ROOT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COL_MAP = {
    "تاريخ الترحيل": "posting_date",
    "اسم المخزن":   "branch_name",
    "كود المخزن":   "branch_code",
    "القيمة":        "total_value",
    "قيمة الحركة":  "net_value",
    "المدفوع نقدي": "cash_paid",
    "مردود":        "returns",
    "خصم / فاتورة": "discount",
    "ضريبة مبيعات": "sales_tax",
    "خدمة":         "service_fee",
    "نوع الحركة":   "tx_type",
    "رقم المستند":  "doc_no",
    "مسلسل الحاسب": "system_serial",
    "اسم العميل":   "customer_name",
    "كود العميل":   "customer_code",
    "المندوب":      "salesperson",
}

TX_TYPE_MAP = {"نقدى": "cash", "بطاقات إئتمان": "card", "آجل": "credit"}

BRANCH_NAME_MAP = {
    "اسيوط": "Asyut", "أكتوبر": "October", "اكتوبر": "October",
    "المعادى": "Maadi", "المعادي": "Maadi", "النخيل": "Nakheel",
    "شيراتون": "Sheraton", "مينى ستور - محرم": "MiniStore_Mehrem",
    "محرم": "Mehrem", "مدينتى": "Madinaty", "مدينتي": "Madinaty",
}

NUMERIC_COLS = ["total_value", "net_value", "cash_paid",
                "returns", "discount", "sales_tax", "service_fee"]

# Hard sanity cap on per-transaction value (EGP). The 99th percentile of every
# legitimate branch is ~1,500 and absolute max is ~20k. Anything above this
# threshold (or below its negative) is a known POS-sync corruption — we have
# seen 10^6 .. 10^9 entries on Sheraton on 2026-02-06/07. Drop them.
MAX_ABS_TX_VALUE = 100_000  # EGP


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    files = sorted(glob.glob(str(SRC_DIR / "*.xlsx")) +
                   glob.glob(str(SRC_DIR / "*.xls")))
    if not files:
        raise SystemExit(f"No .xlsx/.xls files in {SRC_DIR}")

    frames = []
    for f in files:
        path = Path(f)
        # Some exports come as cp1256-encoded TSV with an .xls extension
        # (they're not OLE compound files). Detect via magic bytes and parse as TSV.
        try:
            with open(path, "rb") as fh:
                magic = fh.read(8)
            is_real_xls = magic.startswith(b"\xd0\xcf\x11\xe0")
            is_xlsx = magic.startswith(b"PK\x03\x04")
            if path.suffix.lower() == ".xls" and not is_real_xls and not is_xlsx:
                df = pd.read_csv(path, sep="\t", encoding="cp1256", engine="python")
            else:
                df = pd.read_excel(path)
        except Exception as e:
            print(f"  WARN: could not read {path.name} ({e}); skipping")
            continue
        df = df.rename(columns=COL_MAP)
        df["source_file"] = path.name

        # drop subtotal rows (no date / no branch)
        df = df.dropna(subset=["posting_date", "branch_code"]).copy()

        # parse date, normalise text. Use dayfirst=True because cp1256 TSV exports
        # write dates as dd/mm/yyyy (xlsx files come through as real datetimes already).
        df["posting_date"] = pd.to_datetime(
            df["posting_date"], errors="coerce", dayfirst=True
        ).dt.normalize()
        df = df.dropna(subset=["posting_date"])
        df["branch_name"] = df["branch_name"].astype(str).str.strip()
        df["branch_name_en"] = df["branch_name"].map(BRANCH_NAME_MAP).fillna(df["branch_name"])
        if "tx_type" in df.columns:
            df["tx_type_en"] = df["tx_type"].map(TX_TYPE_MAP).fillna("other")

        # numeric coercion + NaN -> 0
        for c in NUMERIC_COLS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        print(f"{path.name:30s}  rows={len(df):>5}  "
              f"branch={df['branch_name_en'].iloc[0]}  "
              f"{df['posting_date'].min().date()} .. {df['posting_date'].max().date()}")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    # ---- drop corrupted POS rows (values outside a realistic envelope) ----
    val_cols = [c for c in ("total_value", "net_value", "cash_paid") if c in merged.columns]
    bad_mask = False
    for c in val_cols:
        bad_mask = bad_mask | (merged[c].abs() > MAX_ABS_TX_VALUE)
    if hasattr(bad_mask, "sum") and bad_mask.sum():
        bad = merged[bad_mask]
        print(f"\nDropping {len(bad)} corrupted rows (|value| > {MAX_ABS_TX_VALUE:,} EGP):")
        print(bad.groupby(["branch_name_en", "posting_date"]).size()
                 .rename("rows").to_string())
        merged = merged[~bad_mask].reset_index(drop=True)

    # de-dup exact rows (same branch, date, doc, value)
    dup_keys = [k for k in ("branch_code", "posting_date", "doc_no", "total_value")
                if k in merged.columns]
    before = len(merged)
    merged = merged.drop_duplicates(subset=dup_keys).reset_index(drop=True)

    # tidy column order
    front = ["posting_date", "branch_code", "branch_name", "branch_name_en",
             "tx_type", "tx_type_en", "doc_no", "system_serial",
             "total_value", "net_value", "cash_paid", "returns",
             "discount", "sales_tax", "salesperson", "source_file"]
    cols = [c for c in front if c in merged.columns] + \
           [c for c in merged.columns if c not in front]
    merged = merged[cols]

    out_csv = OUT_DIR / "merged_new_sales.csv"
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nMerged {len(files)} files -> {out_csv}")
    print(f"  rows={len(merged):,}  duplicates removed={before - len(merged):,}")
    print(f"  branches={merged['branch_name_en'].nunique()}  "
          f"days={merged['posting_date'].nunique()}  "
          f"range={merged['posting_date'].min().date()} .. {merged['posting_date'].max().date()}")

    try:
        out_pq = OUT_DIR / "merged_new_sales.parquet"
        # cast object cols to string so parquet doesn't trip on mixed types
        for c in merged.select_dtypes(include="object").columns:
            merged[c] = merged[c].astype("string")
        merged.to_parquet(out_pq, index=False)
        print(f"  also wrote {out_pq.name}")
    except Exception as e:
        print(f"  (parquet skipped: {e})")


if __name__ == "__main__":
    main()
