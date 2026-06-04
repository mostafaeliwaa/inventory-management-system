"""Feature engineering for the LightGBM path. Prophet builds its own features."""
from __future__ import annotations

import pandas as pd

# Egypt Ramadan/Eid windows — manually curated for the years our data spans
# (the `holidays` package doesn't carry Islamic dates reliably). With ~2 years of
# data we need 2024, 2025, and 2026 ranges so every training fold sees them.
RAMADAN_WINDOWS = [
    (pd.Timestamp("2024-03-11"), pd.Timestamp("2024-04-09")),  # Ramadan 2024
    (pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-29")),  # Ramadan 2025
    (pd.Timestamp("2026-02-18"), pd.Timestamp("2026-03-19")),  # Ramadan 2026
]
EID_FITR_WINDOWS = [
    [pd.Timestamp("2024-04-10"), pd.Timestamp("2024-04-11"), pd.Timestamp("2024-04-12")],
    [pd.Timestamp("2025-03-30"), pd.Timestamp("2025-03-31"), pd.Timestamp("2025-04-01")],
    [pd.Timestamp("2026-03-20"), pd.Timestamp("2026-03-21"), pd.Timestamp("2026-03-22")],
]
# Kept for backward compat (single-year holiday frame used by Prophet for the current cycle).
RAMADAN_START = RAMADAN_WINDOWS[-1][0]
RAMADAN_END   = RAMADAN_WINDOWS[-1][1]
EID_FITR      = EID_FITR_WINDOWS[-1]

EG_PUBLIC_HOLIDAYS = [
    # Coptic Christmas (Jan 7), Revolution Day (Jan 25), Sinai Liberation (Apr 25), Labour Day (May 1)
    *[pd.Timestamp(f"{y}-01-07") for y in (2024, 2025, 2026)],
    *[pd.Timestamp(f"{y}-01-25") for y in (2024, 2025, 2026)],
    *[pd.Timestamp(f"{y}-04-25") for y in (2024, 2025, 2026)],
    *[pd.Timestamp(f"{y}-05-01") for y in (2024, 2025, 2026)],
]

# With >1 year of history we can use a 364-day lag (same day-of-week last year)
# and a 364-day rolling mean to capture annual seasonality.
LAGS = [1, 2, 3, 7, 14, 28, 364]
ROLL_MEANS = [7, 14, 28]
ROLL_STDS  = [7, 14]
CROSS_TARGETS = ["tx_count", "cash_share", "was_open"]
CROSS_LAGS = [1, 7]


def _calendar(df: pd.DataFrame) -> pd.DataFrame:
    d = df["posting_date"]
    df["dow"] = d.dt.dayofweek                    # 0=Mon … 4=Fri, 5=Sat
    df["is_weekend"] = df["dow"].isin([4, 5]).astype(int)  # EG weekend
    df["day_of_month"] = d.dt.day
    df["week_of_month"] = ((d.dt.day - 1) // 7 + 1).astype(int)
    df["month"] = d.dt.month
    return df


def _holidays(df: pd.DataFrame) -> pd.DataFrame:
    d = df["posting_date"]

    # is_ramadan and ramadan_day_index reflect *whichever* Ramadan window each date
    # falls inside, across all years (2024-2026).
    is_ramadan = pd.Series(0, index=df.index, dtype=int)
    ramadan_day_index = pd.Series(0, index=df.index, dtype=int)
    pre_ramadan_week = pd.Series(0, index=df.index, dtype=int)
    for start, end in RAMADAN_WINDOWS:
        mask = (d >= start) & (d <= end)
        is_ramadan = is_ramadan | mask.astype(int)
        ramadan_day_index = ramadan_day_index.where(~mask, (d - start).dt.days + 1)
        pre_ramadan_week = pre_ramadan_week | (
            ((d >= start - pd.Timedelta(days=7)) & (d < start)).astype(int)
        )
    df["is_ramadan"] = is_ramadan
    df["ramadan_day_index"] = ramadan_day_index.astype(int)
    df["pre_ramadan_week"] = pre_ramadan_week

    # Ramadan trajectory: first 10 / middle 10 / last 10 / last 3 (peak).
    df["ramadan_first_10"] = ((df["ramadan_day_index"] >= 1) & (df["ramadan_day_index"] <= 10)).astype(int)
    df["ramadan_mid_10"]   = ((df["ramadan_day_index"] >= 11) & (df["ramadan_day_index"] <= 20)).astype(int)
    df["ramadan_last_10"]  = ((df["ramadan_day_index"] >= 21) & (df["ramadan_day_index"] <= 30)).astype(int)
    df["ramadan_last_3"]   = ((df["ramadan_day_index"] >= 28) & (df["ramadan_day_index"] <= 30)).astype(int)

    # Eid: flag day-of, the week before, and the week after, across all years.
    all_eid_days = [day for window in EID_FITR_WINDOWS for day in window]
    df["is_eid_fitr"] = d.isin(all_eid_days).astype(int)
    pre_eid_week  = pd.Series(0, index=df.index, dtype=int)
    post_eid_week = pd.Series(0, index=df.index, dtype=int)
    for window in EID_FITR_WINDOWS:
        pre_eid_week = pre_eid_week | (
            ((d >= window[0] - pd.Timedelta(days=7)) & (d < window[0])).astype(int)
        )
        post_eid_week = post_eid_week | (
            ((d > window[-1]) & (d <= window[-1] + pd.Timedelta(days=7))).astype(int)
        )
    df["pre_eid_week"]  = pre_eid_week
    df["post_eid_week"] = post_eid_week

    # days_to_eid uses the NEXT upcoming Eid in any year — signed and clipped.
    eid_first_days = sorted({w[0] for w in EID_FITR_WINDOWS})
    def _days_to_next_eid(ts: pd.Timestamp) -> int:
        future = [(e - ts).days for e in eid_first_days if e >= ts]
        if future:
            return min(max(future[0], -14), 30)
        # past the last known Eid: signed distance from the most recent one
        return max((eid_first_days[-1] - ts).days, -14)
    df["days_to_eid"] = d.map(_days_to_next_eid).astype(int)

    df["is_public_holiday"] = d.isin(EG_PUBLIC_HOLIDAYS).astype(int)
    return df


def _lag_and_roll(df: pd.DataFrame, target: str = "revenue") -> pd.DataFrame:
    df = df.sort_values(["branch_code", "posting_date"]).copy()
    g = df.groupby("branch_code", sort=False)[target]
    for k in LAGS:
        df[f"{target}_lag_{k}"] = g.shift(k)
    # Roll on the *shifted* series so the current day's value doesn't leak in.
    shifted = g.shift(1)
    for w in ROLL_MEANS:
        df[f"{target}_rmean_{w}"] = shifted.groupby(df["branch_code"]).transform(
            lambda s: s.rolling(window=w, min_periods=max(2, w // 2)).mean()
        )
    for w in ROLL_STDS:
        df[f"{target}_rstd_{w}"] = shifted.groupby(df["branch_code"]).transform(
            lambda s: s.rolling(window=w, min_periods=max(2, w // 2)).std()
        )
    return df


def _cross_target_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["branch_code", "posting_date"]).copy()
    for col in CROSS_TARGETS:
        if col not in df.columns:
            continue
        g = df.groupby("branch_code", sort=False)[col]
        for k in CROSS_LAGS:
            df[f"{col}_lag_{k}"] = g.shift(k)
    return df


def build_features(panel: pd.DataFrame, target: str = "revenue") -> pd.DataFrame:
    df = panel.copy()
    df = _calendar(df)
    df = _holidays(df)
    df = _lag_and_roll(df, target=target)
    df = _cross_target_lags(df)
    return df


def feature_columns(target: str = "revenue") -> list[str]:
    lag_cols  = [f"{target}_lag_{k}" for k in LAGS]
    mean_cols = [f"{target}_rmean_{w}" for w in ROLL_MEANS]
    std_cols  = [f"{target}_rstd_{w}"  for w in ROLL_STDS]
    calendar  = ["dow", "is_weekend", "day_of_month", "week_of_month", "month"]
    holidays  = [
        "is_ramadan", "ramadan_day_index",
        "ramadan_first_10", "ramadan_mid_10", "ramadan_last_10", "ramadan_last_3",
        "pre_ramadan_week", "pre_eid_week", "is_eid_fitr", "post_eid_week",
        "days_to_eid", "is_public_holiday",
    ]
    cross_cols = [f"{c}_lag_{k}" for c in CROSS_TARGETS for k in CROSS_LAGS]
    return ["branch_code", *calendar, *holidays, *lag_cols, *mean_cols, *std_cols, *cross_cols]


if __name__ == "__main__":
    from forecasting.src.data_loader import get_panel

    panel = get_panel()
    feat = build_features(panel)
    print(f"feature df: {feat.shape}")
    print(f"NaN by column (top 8):")
    print(feat.isna().sum().sort_values(ascending=False).head(8))
    # Sanity: lag_28 should be NaN exactly for first 28 days per branch.
    n_nan_lag28 = feat.groupby("branch_code")["revenue_lag_28"].apply(lambda s: s.isna().sum())
    print(f"\nlag_28 NaN count per branch (expect 28 each): {n_nan_lag28.tolist()}")
    # Spot-check holiday flags
    ram_days = feat.loc[feat["is_ramadan"] == 1, "posting_date"].dt.date.unique()
    print(f"\nRamadan days flagged: {len(ram_days)} (expect 30); first={ram_days.min()}, last={ram_days.max()}")
    eid_days = feat.loc[feat["is_eid_fitr"] == 1, "posting_date"].dt.date.unique()
    print(f"Eid al-Fitr days flagged: {sorted(eid_days)}")
