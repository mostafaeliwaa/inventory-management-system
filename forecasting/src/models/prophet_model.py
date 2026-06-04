"""One Prophet model per branch, with Ramadan + Eid + EG public holidays."""
from __future__ import annotations

import logging
import warnings

import pandas as pd

from forecasting.src.features import (
    EG_PUBLIC_HOLIDAYS,
    EID_FITR_WINDOWS,
    RAMADAN_WINDOWS,
)

# Prophet is loud — silence cmdstanpy at import time.
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)


def _holiday_frame() -> pd.DataFrame:
    """Union of every Ramadan/Eid window we have ground truth for (2024-2026)."""
    ramadan_days = pd.concat([
        pd.Series(pd.date_range(s, e, freq="D")) for s, e in RAMADAN_WINDOWS
    ], ignore_index=True)
    ramadan = pd.DataFrame({
        "holiday": "ramadan",
        "ds": ramadan_days,
        "lower_window": 0,
        "upper_window": 0,
    })
    eid_days = pd.to_datetime([d for w in EID_FITR_WINDOWS for d in w])
    eid = pd.DataFrame({
        "holiday": "eid_fitr",
        "ds": eid_days,
        "lower_window": -1,
        "upper_window": 1,
    })
    public = pd.DataFrame({
        "holiday": "eg_public",
        "ds": pd.to_datetime(EG_PUBLIC_HOLIDAYS),
        "lower_window": 0,
        "upper_window": 0,
    })
    return pd.concat([ramadan, eid, public], ignore_index=True)


# Additive mode (handles refund days). With ~2 years of data we can be a bit more
# aggressive on changepoint scale.
CHANGEPOINT_GRID = [0.05, 0.1, 0.2, 0.3]


def _fit_predict(train: pd.DataFrame, future_dates: pd.DatetimeIndex,
                 changepoint_prior_scale: float, holidays_df: pd.DataFrame,
                 seasonality_mode: str = "additive") -> pd.DataFrame:
    from prophet import Prophet
    import contextlib, io
    # With >365 days of training data we can enable yearly seasonality safely;
    # Prophet auto-disables it below ~2 cycles, so we set 'auto' and let it decide.
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality="auto",
        daily_seasonality=False,
        seasonality_mode=seasonality_mode,
        holidays=holidays_df,
        changepoint_prior_scale=changepoint_prior_scale,
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model.fit(train)
        return model.predict(pd.DataFrame({"ds": future_dates}))


def _pick_cps_per_branch(train: pd.DataFrame, holidays_df: pd.DataFrame) -> float:
    """Hold out the last 7 training days, pick the cps with the lowest WAPE."""
    if len(train) < 35:
        return 0.05
    cutoff = train["ds"].max() - pd.Timedelta(days=6)
    inner_train = train[train["ds"] < cutoff]
    inner_val = train[train["ds"] >= cutoff]
    best_cps, best_wape = 0.05, float("inf")
    for cps in CHANGEPOINT_GRID:
        try:
            fcst = _fit_predict(inner_train, pd.DatetimeIndex(inner_val["ds"]), cps, holidays_df)
            merged = inner_val[["ds", "y"]].merge(fcst[["ds", "yhat"]], on="ds")
            err = (merged["y"] - merged["yhat"]).abs().sum()
            denom = merged["y"].abs().sum() or 1.0
            wape = err / denom
            if wape < best_wape:
                best_wape, best_cps = wape, cps
        except Exception:
            continue
    return best_cps


def prophet_forecast(panel: pd.DataFrame, test_dates: pd.DatetimeIndex, target: str = "revenue") -> pd.DataFrame:
    """Fit one additive Prophet per branch with per-branch changepoint tuning."""
    holidays_df = _holiday_frame()
    out = []
    for code, g in panel.groupby("branch_code", sort=False):
        train = g[["posting_date", target]].rename(columns={"posting_date": "ds", target: "y"})
        cps = _pick_cps_per_branch(train, holidays_df)
        fcst = _fit_predict(train, test_dates, cps, holidays_df, seasonality_mode="additive")
        for ds, yhat in zip(fcst["ds"], fcst["yhat"]):
            out.append({"branch_code": code, "posting_date": pd.Timestamp(ds), "y_pred": float(yhat)})
    return pd.DataFrame(out)
