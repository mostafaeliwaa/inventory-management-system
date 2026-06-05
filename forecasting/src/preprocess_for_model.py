"""Transform raw sales data into model-ready features for the global LightGBM forecaster.

This module is the single entry point for the production preprocessing pipeline.
It takes raw transactional data (same schema as ``merged_new_sales.csv``) and produces
the feature-engineered DataFrame that the serialized LightGBM models expect.

Flow::

    Raw DB rows  (posting_date, branch_code, branch_name_en, total_value, cash_paid)
        → validate & clean
        → build_daily_panel          (one row per branch × day)
        → build_weekly_panel         (one row per branch × ISO-week, with holiday day-counts)
        → build_global_features      (lags, rolling stats, Fourier, momentum, peer, calendar)
        → model-ready DataFrame

Usage::

    # As a library
    from forecasting.src.preprocess_for_model import preprocess
    features_df = preprocess(raw_df)

    # As a CLI
    python -m forecasting.src.preprocess_for_model --input data.csv --output features.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from forecasting.src.data_loader import build_daily_panel
from forecasting.src.weekly import build_weekly_panel
from forecasting.src.weekly_global import build_global_features, feature_columns

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Minimum columns required from the raw database export.
REQUIRED_COLUMNS = ["posting_date", "branch_code", "branch_name_en", "total_value", "cash_paid"]

# Minimum rows per branch to produce meaningful lag features.
MIN_DAYS_PER_BRANCH = 30


def validate_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean raw transactional data.

    Args:
        df: Raw DataFrame with at least the columns in ``REQUIRED_COLUMNS``.
            Accepts ``branch_id`` as an alias for ``branch_code``.

    Returns:
        Cleaned DataFrame ready for panel aggregation.

    Raises:
        ValueError: If required columns are missing or data is insufficient.
    """
    df = df.copy()

    # Handle DB column name differences: branch_id -> branch_code.
    if "branch_id" in df.columns and "branch_code" not in df.columns:
        df = df.rename(columns={"branch_id": "branch_code"})
        logger.info("Renamed 'branch_id' -> 'branch_code'.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Expected: {REQUIRED_COLUMNS}"
        )

    # Parse and normalize dates.
    df["posting_date"] = pd.to_datetime(df["posting_date"], errors="coerce")
    initial_len = len(df)
    df = df.dropna(subset=["posting_date", "branch_code"])
    dropped = initial_len - len(df)
    if dropped > 0:
        logger.warning("Dropped %d rows with null posting_date or branch_code.", dropped)

    df["posting_date"] = df["posting_date"].dt.normalize()
    df["branch_code"] = df["branch_code"].astype(int)

    # Ensure numeric columns are actually numeric.
    for col in ["total_value", "cash_paid"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Validate we have enough data.
    date_range = (df["posting_date"].max() - df["posting_date"].min()).days
    if date_range < MIN_DAYS_PER_BRANCH:
        logger.warning(
            "Only %d days of history — lag features will be sparse. "
            "Recommend >= %d days for reliable forecasts.",
            date_range, MIN_DAYS_PER_BRANCH,
        )

    n_branches = df["branch_code"].nunique()
    n_rows = len(df)
    logger.info(
        "Validated: %d rows, %d branches, date range %s to %s (%d days).",
        n_rows, n_branches,
        df["posting_date"].min().date(), df["posting_date"].max().date(),
        date_range,
    )
    return df


def preprocess(
    raw_df: pd.DataFrame,
    *,
    return_weekly_panel: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Full preprocessing pipeline: raw transactions → model-ready features.

    Args:
        raw_df: Raw transactional DataFrame (same schema as merged_new_sales.csv).
        return_weekly_panel: If True, also return the intermediate weekly panel.

    Returns:
        Model-ready feature DataFrame. If ``return_weekly_panel`` is True, returns
        a tuple of ``(features_df, weekly_panel_df)``.
    """
    logger.info("Step 1/4: Validating raw data...")
    clean_df = validate_raw_data(raw_df)

    logger.info("Step 2/4: Building daily panel...")
    daily_panel = build_daily_panel(clean_df)
    logger.info(
        "  Daily panel: %d rows (%d branches × %d days).",
        len(daily_panel),
        daily_panel["branch_code"].nunique(),
        daily_panel["posting_date"].nunique(),
    )

    logger.info("Step 3/4: Building weekly panel...")
    weekly_panel = build_weekly_panel(daily_panel)
    logger.info(
        "  Weekly panel: %d rows (%d branches × %d weeks).",
        len(weekly_panel),
        weekly_panel["branch_code"].nunique(),
        weekly_panel["week_start"].nunique(),
    )

    logger.info("Step 4/4: Engineering features...")
    features_df = build_global_features(weekly_panel)
    logger.info(
        "  Feature matrix: %d rows × %d columns.",
        len(features_df), len(features_df.columns),
    )

    if return_weekly_panel:
        return features_df, weekly_panel
    return features_df


def preprocess_from_csv(
    csv_path: str | Path,
    *,
    return_weekly_panel: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: read CSV then preprocess.

    Args:
        csv_path: Path to CSV file with raw transactional data.
        return_weekly_panel: If True, also return the intermediate weekly panel.

    Returns:
        Model-ready feature DataFrame.
    """
    logger.info("Reading %s ...", csv_path)
    raw_df = pd.read_csv(csv_path, parse_dates=["posting_date"])
    return preprocess(raw_df, return_weekly_panel=return_weekly_panel)


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature column names the model expects.

    This is the canonical source of truth for feature ordering in production.
    """
    return feature_columns()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess raw sales data into model-ready features."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to input CSV. Defaults to data/merged_new_sales.csv.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save feature CSV. If omitted, prints summary only.",
    )
    args = parser.parse_args()

    input_path = args.input
    if input_path is None:
        input_path = Path(__file__).resolve().parents[2] / "data" / "merged_new_sales.csv"

    features_df = preprocess_from_csv(input_path)

    cols = get_feature_columns()
    available = [c for c in cols if c in features_df.columns]
    missing_feats = [c for c in cols if c not in features_df.columns]

    print(f"\n--- Preprocessing Summary ---")
    print(f"Input:            {input_path}")
    print(f"Output rows:      {len(features_df):,}")
    print(f"Feature columns:  {len(available)} / {len(cols)} available")
    if missing_feats:
        print(f"Missing features: {missing_feats}")
    print(f"NaN counts (top 5):")
    nan_counts = features_df[available].isna().sum().sort_values(ascending=False).head(5)
    for col_name, count in nan_counts.items():
        print(f"  {col_name}: {count}")

    if args.output:
        output_path = Path(args.output)
        features_df.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")
