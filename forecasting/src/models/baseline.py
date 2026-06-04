"""Trivial baselines. Beating these is the bar."""
from __future__ import annotations

import pandas as pd


def seasonal_naive_forecast(panel: pd.DataFrame, test_dates: pd.DatetimeIndex, target: str = "revenue") -> pd.DataFrame:
    """Forecast y_t = y_{t-7} per branch."""
    out = []
    for code, g in panel.groupby("branch_code", sort=False):
        g = g.set_index("posting_date").sort_index()
        for d in test_dates:
            ref = d - pd.Timedelta(days=7)
            y_hat = g.loc[ref, target] if ref in g.index else 0.0
            out.append({"branch_code": code, "posting_date": d, "y_pred": float(y_hat)})
    return pd.DataFrame(out)


def naive_forecast(panel: pd.DataFrame, test_dates: pd.DatetimeIndex, target: str = "revenue") -> pd.DataFrame:
    """Forecast y_t = y_{t-1} per branch. Repeats the last training day across the horizon."""
    out = []
    horizon_start = test_dates.min()
    last_train_day = horizon_start - pd.Timedelta(days=1)
    for code, g in panel.groupby("branch_code", sort=False):
        g = g.set_index("posting_date").sort_index()
        y_last = g.loc[last_train_day, target] if last_train_day in g.index else 0.0
        for d in test_dates:
            out.append({"branch_code": code, "posting_date": d, "y_pred": float(y_last)})
    return pd.DataFrame(out)
