"""Load merged_new_sales and reduce it to a daily branch panel."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "merged_new_sales.parquet"
CSV_FALLBACK = Path(__file__).resolve().parents[2] / "data" / "merged_new_sales.csv"


def load_transactions(path: Path | str | None = None) -> pd.DataFrame:
    p = Path(path) if path else DATA_PATH
    if p.exists():
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(CSV_FALLBACK, parse_dates=["posting_date"])
    df = df.dropna(subset=["posting_date", "branch_code"]).copy()
    df["posting_date"] = pd.to_datetime(df["posting_date"]).dt.normalize()
    df["branch_code"] = df["branch_code"].astype(int)
    return df


def build_daily_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per (branch, day). Missing days are filled with 0 and was_open=0."""
    agg = (
        df.groupby(["branch_code", "branch_name_en", "posting_date"], as_index=False)
          .agg(
              revenue=("total_value", "sum"),
              tx_count=("total_value", "size"),
              cash_paid=("cash_paid", "sum"),
          )
    )
    agg["cash_share"] = np.where(agg["revenue"] != 0, agg["cash_paid"] / agg["revenue"], 0.0)

    full_range = pd.date_range(df["posting_date"].min(), df["posting_date"].max(), freq="D")
    branches = agg[["branch_code", "branch_name_en"]].drop_duplicates()
    grid = branches.assign(key=1).merge(
        pd.DataFrame({"posting_date": full_range, "key": 1}), on="key"
    ).drop(columns="key")

    panel = grid.merge(agg, on=["branch_code", "branch_name_en", "posting_date"], how="left")
    panel["was_open"] = panel["revenue"].notna().astype(int)
    panel[["revenue", "tx_count", "cash_paid", "cash_share"]] = (
        panel[["revenue", "tx_count", "cash_paid", "cash_share"]].fillna(0.0)
    )
    panel = panel.sort_values(["branch_code", "posting_date"]).reset_index(drop=True)
    return panel


def get_panel(path: Path | str | None = None) -> pd.DataFrame:
    return build_daily_panel(load_transactions(path))


if __name__ == "__main__":
    panel = get_panel()
    print(f"panel shape: {panel.shape}")
    print(f"date range: {panel['posting_date'].min().date()} to {panel['posting_date'].max().date()}")
    print(f"branches: {panel['branch_name_en'].nunique()}")
    print(panel.groupby("branch_name_en").agg(
        days=("posting_date", "nunique"),
        days_open=("was_open", "sum"),
        total_revenue=("revenue", "sum"),
    ).round(0))
