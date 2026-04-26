"""
Retail Data Cleaning & Merging Pipeline
========================================
Loads 4 retail datasets (Sales & Stock, Pricing, Warehouse Inventory, Suppliers),
cleans them, and merges into a single Master DataFrame keyed on Item Code (الكود).
"""

import os
import re
import warnings

os.environ["PYTHONUTF8"] = "1"
warnings.filterwarnings("ignore")

import pandas as pd


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def _clean_item_code(series: pd.Series) -> pd.Series:
    """Convert item codes to clean string: remove .0 suffix, strip whitespace."""
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s


def _is_numeric_code(series: pd.Series) -> pd.Series:
    """Return boolean mask: True where the code consists entirely of digits."""
    return series.str.fullmatch(r"\d+", na=False)


def _strip_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply .str.strip() to every object (text) column."""
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


def _clamp_negatives(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Replace negative values with 0 in the specified numeric columns."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] < 0, col] = 0
    return df


def _fill_numeric_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN with 0 for all numeric columns."""
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(0)
    return df


# ──────────────────────────────────────────────
# Per-file cleaning
# ──────────────────────────────────────────────

def _clean_file1_sales(path: str) -> pd.DataFrame:
    """
    File 1 — Transactional Sales & Stock (مبيعات + استوكات خلال فترة).

    Row 0 contains sub-headers (مبيعات / استوك); actual data starts at row 1.
    Each branch occupies two adjacent columns: sales then stock.
    """
    raw = pd.read_excel(path)
    raw.columns = raw.columns.str.strip()
    raw_cols = raw.columns.tolist()

    # Actual data rows (skip the sub-header row)
    df = raw.iloc[1:].copy().reset_index(drop=True)

    # ── Item code ──
    df["الكود"] = _clean_item_code(df["الكود"])

    # ── Rename item name column to a standard name ──
    if "الصنف" in df.columns:
        df.rename(columns={"الصنف": "اسم الصنف"}, inplace=True)

    # ── Parse branch sales/stock pairs into clearly named columns ──
    branch_pair_indices = [
        (2, 3, "اسيوط"),
        (4, 5, "اكتوبر"),
        (6, 7, "العباسيه"),
        (8, 9, "معادى"),
        (10, 11, "تجمع"),
        (12, 13, "توى المعادى"),
        (14, 15, "توى التجمع"),
        (16, 17, "شيراتون"),
        (18, 19, "محرم"),
        (20, 21, "مدينتى"),
        (22, 23, "مينى ستور محرم"),
    ]

    numeric_cols = []
    for sales_idx, stock_idx, branch_name in branch_pair_indices:
        sales_src = raw_cols[sales_idx]
        stock_src = raw_cols[stock_idx]
        s_col = f"{branch_name}_مبيعات"
        k_col = f"{branch_name}_استوك"
        df[s_col] = pd.to_numeric(df[sales_src], errors="coerce").fillna(0)
        df[k_col] = pd.to_numeric(df[stock_src], errors="coerce").fillna(0)
        numeric_cols.extend([s_col, k_col])

    # Totals
    for col in ["اجمالى مبيعات", "اجمالى استوك"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        numeric_cols.append(col)

    # Drop the original unnamed / raw branch columns that we already parsed
    cols_to_drop = [c for c in df.columns if c.startswith("Unnamed") or c in
                    [raw_cols[i] for pair in branch_pair_indices for i in (pair[0], pair[1])
                     if raw_cols[i] not in ("اجمالى مبيعات", "اجمالى استوك", "الكود", "اسم الصنف", "الصنف")]]
    # Deduplicate and only drop what actually exists
    cols_to_drop = list(set(cols_to_drop) & set(df.columns))
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # ── Business validations ──
    df = _clamp_negatives(df, numeric_cols)
    df = _strip_text_columns(df)
    df = _fill_numeric_nans(df)

    # ── Drop invalid codes ──
    df = df[df["الكود"].notna() & (df["الكود"] != "") & (df["الكود"] != "nan")]
    df = df[_is_numeric_code(df["الكود"])].reset_index(drop=True)

    return df


def _clean_file2_pricing(path: str) -> pd.DataFrame:
    """
    File 2 — Pricing & Total Branch Stock (استوك اجمالى الفروع).

    Columns: مجموعه الصنف, كود الصنف, اسم الصنف, branch stock columns...,
             سعر التكلفه (Cost Price), سعر القطعة (Selling Price).
    """
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    # ── Standardize join key ──
    df.rename(columns={"كود الصنف": "الكود"}, inplace=True)
    df["الكود"] = _clean_item_code(df["الكود"])

    # ── Numeric coercion for prices and branch stock columns ──
    price_cols = ["سعر التكلفه", "سعر القطعة"]
    branch_stock_cols = [
        "الاداره", "النخيل", "هوالك", "المعادي", "اكتوبر",
        "شيراتون", "سبورتينج", "محرم", "اسيوط",
        "توى لاند- المعادي", "توى لاند النخيل", "مدينتي",
    ]
    numeric_cols = price_cols + branch_stock_cols
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _fill_numeric_nans(df)

    # ── Business validation: cost > selling price → drop ──
    df = df[~(df["سعر التكلفه"] > df["سعر القطعة"])]

    # ── Clamp negatives in stock / price columns ──
    df = _clamp_negatives(df, numeric_cols)

    df = _strip_text_columns(df)

    # ── Drop invalid codes ──
    df = df[df["الكود"].notna() & (df["الكود"] != "") & (df["الكود"] != "nan")]
    df = df[_is_numeric_code(df["الكود"])].reset_index(drop=True)

    return df


def _clean_file3_warehouse(path: str) -> pd.DataFrame:
    """
    File 3 — Main Warehouse Stock (stock.xls).

    Columns: سعر بيع الوحدة, المجموعه الرئيسيه, الكمية, الوحدة,
             اسم الصنف, كود الصنف, مجموعة الصنف, المخزن.
    """
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    # ── Standardize join key ──
    df.rename(columns={"كود الصنف": "الكود"}, inplace=True)
    df["الكود"] = _clean_item_code(df["الكود"])

    # ── Numeric coercion ──
    qty_cols = ["الكمية", "سعر بيع الوحدة"]
    for col in qty_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = _fill_numeric_nans(df)
    df = _clamp_negatives(df, qty_cols)

    df = _strip_text_columns(df)

    # ── Drop invalid codes ──
    df = df[df["الكود"].notna() & (df["الكود"] != "") & (df["الكود"] != "nan")]
    df = df[_is_numeric_code(df["الكود"])].reset_index(drop=True)

    return df


def _clean_file4_suppliers(path: str) -> pd.DataFrame:
    """
    File 4 — Suppliers (شيت بالاصناف و الموردين).

    Columns: المجموعة, الكود, اسم الصنف, المورد.
    """
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    df["الكود"] = _clean_item_code(df["الكود"])

    df = _strip_text_columns(df)

    # ── Drop invalid codes ──
    df = df[df["الكود"].notna() & (df["الكود"] != "") & (df["الكود"] != "nan")]
    df = df[_is_numeric_code(df["الكود"])].reset_index(drop=True)

    # ── Deduplicate: keep one supplier entry per item code ──
    # Multiple rows per code exist (e.g. different transactions); keep the first.
    df = df.drop_duplicates(subset=["الكود"], keep="first").reset_index(drop=True)

    return df


# ──────────────────────────────────────────────
# Merge pipeline
# ──────────────────────────────────────────────

def _resolve_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    After merging, if overlapping columns produced _x / _y suffixes,
    keep the _x variant (from the left/base table) and drop _y.
    """
    cols_to_drop = []
    rename_map = {}

    for col in df.columns:
        if col.endswith("_x"):
            base_name = col[:-2]
            y_col = f"{base_name}_y"
            if y_col in df.columns:
                # Fill gaps in _x from _y, then drop _y
                df[col] = df[col].fillna(df[y_col])
                cols_to_drop.append(y_col)
            rename_map[col] = base_name

    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    df.rename(columns=rename_map, inplace=True)
    return df


def load_clean_and_merge(
    file1_path: str,
    file2_path: str,
    file3_path: str,
    file4_path: str,
    output_dir: str | None = None,
) -> pd.DataFrame:
    """
    Main entry point.

    Parameters
    ----------
    file1_path : path to Sales & Stock file       (مبيعات + استوكات خلال فترة)
    file2_path : path to Pricing / Branch Stock    (استوك اجمالى الفروع)
    file3_path : path to Main Warehouse Stock      (stock)
    file4_path : path to Suppliers                 (شيت بالاصناف و الموردين)
    output_dir : directory to save cleaned CSVs (individual + master).
                 If None, no intermediate files are saved.

    Returns
    -------
    pd.DataFrame — the merged Master DataFrame.
    """

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Clean each file independently ──
    print("[1/4] Cleaning File 1 (Sales & Stock)...")
    df1 = _clean_file1_sales(file1_path)
    print(f"       → {len(df1):,} rows after cleaning")

    print("[2/4] Cleaning File 2 (Pricing & Branch Stock)...")
    df2 = _clean_file2_pricing(file2_path)
    print(f"       → {len(df2):,} rows after cleaning")

    print("[3/4] Cleaning File 3 (Warehouse Inventory)...")
    df3 = _clean_file3_warehouse(file3_path)
    print(f"       → {len(df3):,} rows after cleaning")

    print("[4/4] Cleaning File 4 (Suppliers)...")
    df4 = _clean_file4_suppliers(file4_path)
    print(f"       → {len(df4):,} rows after cleaning")

    # ── Save individual cleaned files before merging ──
    if output_dir:
        cleaned_files = {
            "cleaned_sales_stock.csv": df1,
            "cleaned_pricing.csv": df2,
            "cleaned_warehouse.csv": df3,
            "cleaned_suppliers.csv": df4,
        }
        print("\nSaving individual cleaned files...")
        for filename, df in cleaned_files.items():
            path = os.path.join(output_dir, filename)
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"  → {filename} ({len(df):,} rows)")

    # ── Step 2: Sequential left merges on الكود ──
    print("\nMerging into Master DataFrame...")

    # Base = File 1 (Sales & Stock)
    master = df1.copy()

    # Merge File 2 (Pricing)
    master = pd.merge(master, df2, on="الكود", how="left", suffixes=("", "_f2"))

    # Merge File 3 (Warehouse)
    master = pd.merge(master, df3, on="الكود", how="left", suffixes=("", "_f3"))

    # Merge File 4 (Suppliers)
    master = pd.merge(master, df4, on="الكود", how="left", suffixes=("", "_f4"))

    # ── Resolve overlapping columns ──
    master = _resolve_duplicate_columns(master)

    # Drop any remaining duplicate-suffixed columns
    suffix_cols = [c for c in master.columns if c.endswith(("_f2", "_f3", "_f4"))]
    if suffix_cols:
        master.drop(columns=suffix_cols, inplace=True, errors="ignore")

    # ── Final fill for any NaN introduced by left joins ──
    master = _fill_numeric_nans(master)
    text_cols = master.select_dtypes(include="object").columns
    master[text_cols] = master[text_cols].fillna("")

    print(f"\nMaster DataFrame ready: {master.shape[0]:,} rows × {master.shape[1]} columns")
    return master


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data")

    OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")

    master_df = load_clean_and_merge(
        file1_path=os.path.join(DATA_DIR, "مبيعات + استوكات خلال فترة.xls"),
        file2_path=os.path.join(DATA_DIR, "استوك اجمالى الفروع.xls"),
        file3_path=os.path.join(DATA_DIR, "stock.xls"),
        file4_path=os.path.join(DATA_DIR, "شيت بالاصناف و الموردين.xls"),
        output_dir=OUTPUT_DIR,
    )

    # Save master
    output_path = os.path.join(OUTPUT_DIR, "master_dataframe.csv")
    master_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved to: {output_path}")

    # Quick summary
    print("\n" + "=" * 60)
    print("Column Summary:")
    print("=" * 60)
    for col in master_df.columns:
        dtype = master_df[col].dtype
        non_null = master_df[col].notna().sum()
        print(f"  {col:<35} {str(dtype):<10} ({non_null:,} non-null)")
