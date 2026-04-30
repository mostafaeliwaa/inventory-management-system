"""Load and engineer features from cleaned inventory CSVs.

The sales/stock dataset is wide: each branch contributes a `_مبيعات` (sales)
and `_استوك` (stock) column. We aggregate these into per-product features
that both the demand classifier and the dead-stock predictor consume.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

SALES_SUFFIX = "_مبيعات"
STOCK_SUFFIX = "_استوك"

CODE_COL = "الكود"
NAME_COL = "اسم الصنف"
TOTAL_SALES_COL = "اجمالى مبيعات"
TOTAL_STOCK_COL = "اجمالى استوك"


def load_sales_stock(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "cleaned_sales_stock.csv"
    df = pd.read_csv(path)
    df[CODE_COL] = df[CODE_COL].astype(str)
    return df


def load_pricing(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "cleaned_pricing.csv"
    df = pd.read_csv(path)
    df["الكود"] = df["الكود"].astype(str)
    keep = ["الكود", "مجموعه الصنف", "سعر التكلفه", "سعر القطعة"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].drop_duplicates(subset=["الكود"])


def _branch_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    sales_cols = [c for c in df.columns if c.endswith(SALES_SUFFIX) and c != TOTAL_SALES_COL]
    stock_cols = [c for c in df.columns if c.endswith(STOCK_SUFFIX) and c != TOTAL_STOCK_COL]
    return sales_cols, stock_cols


def build_feature_table() -> pd.DataFrame:
    """Return one row per product with engineered features.

    Features:
      total_sales, total_stock      - raw totals
      branches_with_sales           - count of branches with sales > 0
      branches_with_stock           - count of branches with stock > 0
      max_branch_sales              - top-performing branch sales
      sales_concentration           - max_branch / total (0..1)
      stock_to_sales_ratio          - total_stock / max(total_sales, 1)
      sell_through                  - total_sales / (total_sales + total_stock)
      cost_price, retail_price, margin_pct (when pricing available)
    """
    df = load_sales_stock()
    sales_cols, stock_cols = _branch_columns(df)

    feats = pd.DataFrame({
        CODE_COL: df[CODE_COL],
        NAME_COL: df[NAME_COL],
        "total_sales": df[sales_cols].sum(axis=1),
        "total_stock": df[stock_cols].sum(axis=1),
        "branches_with_sales": (df[sales_cols] > 0).sum(axis=1),
        "branches_with_stock": (df[stock_cols] > 0).sum(axis=1),
        "max_branch_sales": df[sales_cols].max(axis=1),
        "max_branch_stock": df[stock_cols].max(axis=1),
    })

    feats["sales_concentration"] = np.where(
        feats["total_sales"] > 0,
        feats["max_branch_sales"] / feats["total_sales"],
        0.0,
    )
    feats["stock_to_sales_ratio"] = feats["total_stock"] / feats["total_sales"].replace(0, np.nan)
    feats["stock_to_sales_ratio"] = feats["stock_to_sales_ratio"].fillna(feats["total_stock"] * 1.0 + 100)
    denom = feats["total_sales"] + feats["total_stock"]
    feats["sell_through"] = np.where(denom > 0, feats["total_sales"] / denom, 0.0)

    try:
        pricing = load_pricing()
        feats = feats.merge(pricing, left_on=CODE_COL, right_on="الكود", how="left")
        feats = feats.rename(columns={
            "سعر التكلفه": "cost_price",
            "سعر القطعة": "retail_price",
            "مجموعه الصنف": "category",
        })
        feats["cost_price"] = pd.to_numeric(feats.get("cost_price"), errors="coerce").fillna(0.0)
        feats["retail_price"] = pd.to_numeric(feats.get("retail_price"), errors="coerce").fillna(0.0)
        feats["margin_pct"] = np.where(
            feats["retail_price"] > 0,
            (feats["retail_price"] - feats["cost_price"]) / feats["retail_price"],
            0.0,
        )
    except FileNotFoundError:
        feats["cost_price"] = 0.0
        feats["retail_price"] = 0.0
        feats["margin_pct"] = 0.0
        feats["category"] = "unknown"

    return feats


FEATURE_COLUMNS = [
    "total_sales",
    "total_stock",
    "branches_with_sales",
    "branches_with_stock",
    "max_branch_sales",
    "max_branch_stock",
    "sales_concentration",
    "stock_to_sales_ratio",
    "sell_through",
    "cost_price",
    "retail_price",
    "margin_pct",
]
