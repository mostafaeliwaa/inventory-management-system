"""Weekly aggregation + weekly-horizon forecasts.

Strategy: roll the daily panel up to ISO-week buckets per branch, then run Prophet
and LightGBM on that. With ~20 weekly observations per branch we keep the models
simple — no per-branch tuning, no winsorization (already smoothed by aggregation).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.src.data_loader import get_panel
from forecasting.src.features import EID_FITR_WINDOWS, RAMADAN_WINDOWS
from forecasting.src.metrics import summarize

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
HORIZON_WEEKS = 4


def _week_starts_covering(days: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Map a set of dates to the unique ISO Monday week-starts that contain them."""
    days = pd.DatetimeIndex(days)
    starts = days - pd.to_timedelta(days.dayofweek, unit="D")
    return sorted({pd.Timestamp(s).normalize() for s in starts})


# ---- Ramadan / Eid weeks across every year we have data for ----
RAMADAN_WEEK_STARTS = pd.to_datetime(sorted({
    s for start, end in RAMADAN_WINDOWS
    for s in _week_starts_covering(pd.date_range(start, end, freq="D"))
}))
EID_WEEK_STARTS = pd.to_datetime(sorted({
    s for window in EID_FITR_WINDOWS
    for s in _week_starts_covering(pd.DatetimeIndex(window))
}))

# ---- Payday windows (Egyptian convention: salary lands ~25th–5th of next month) ----
# A day is "payday" if its day-of-month is in {25..31, 1..5}.
def _is_payday(d: pd.Timestamp) -> bool:
    return d.day >= 25 or d.day <= 5

# ---- Egyptian school holidays across the years we have data for ----
# Mid-year break (~last week of Jan to first week of Feb) + spring break (~2nd week of Apr).
SCHOOL_HOLIDAY_DAYS = pd.DatetimeIndex(sorted({
    d for year in (2024, 2025, 2026) for d in (
        pd.date_range(f"{year}-01-23", f"{year}-02-07", freq="D").tolist() +
        pd.date_range(f"{year}-04-11", f"{year}-04-18", freq="D").tolist()
    )
}))
SCHOOL_HOLIDAYS = pd.Series(True, index=SCHOOL_HOLIDAY_DAYS)


def build_weekly_panel(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily (branch, date) -> weekly (branch, week_start) using ISO week-Monday."""
    df = daily.copy()
    df["week_start"] = (df["posting_date"] - pd.to_timedelta(df["posting_date"].dt.dayofweek, unit="D")).dt.normalize()

    # Tag each day with multiple day-level flags before aggregation.
    ramadan_set = {pd.Timestamp(d) for s, e in RAMADAN_WINDOWS
                   for d in pd.date_range(s, e, freq="D")}
    eid_set = {pd.Timestamp(d) for window in EID_FITR_WINDOWS for d in window}
    df["is_ramadan_day"] = df["posting_date"].isin(ramadan_set).astype(int)
    df["is_eid_day"]     = df["posting_date"].isin(eid_set).astype(int)
    df["is_payday_day"]  = df["posting_date"].apply(_is_payday).astype(int)
    df["is_school_holiday_day"] = df["posting_date"].isin(SCHOOL_HOLIDAYS.index).astype(int)

    weekly = (
        df.groupby(["branch_code", "branch_name_en", "week_start"], as_index=False)
          .agg(revenue=("revenue", "sum"),
               tx_count=("tx_count", "sum"),
               days_open=("was_open", "sum"),
               ramadan_days=("is_ramadan_day", "sum"),
               eid_days=("is_eid_day", "sum"),
               payday_days=("is_payday_day", "sum"),
               school_holiday_days=("is_school_holiday_day", "sum"),
               intraweek_std=("revenue", "std"))
    )
    weekly["intraweek_std"] = weekly["intraweek_std"].fillna(0.0)
    # Drop trailing partial week with too few open days.
    max_week = weekly["week_start"].max()
    partial_mask = (weekly["week_start"] == max_week) & (weekly["days_open"] < 5)
    if partial_mask.any():
        weekly = weekly[~partial_mask].copy()
    weekly = weekly.sort_values(["branch_code", "week_start"]).reset_index(drop=True)
    return weekly


# -------------- weekly walk-forward CV --------------
def weekly_folds(weeks: pd.Series, *, horizon: int = 4, first_train_weeks: int = 10, step: int = 1) -> list[tuple]:
    uniq = sorted(pd.to_datetime(weeks.unique()))
    folds: list[tuple] = []
    i = first_train_weeks
    while i + horizon <= len(uniq):
        train_end_excl = uniq[i]
        test_start = uniq[i]
        test_end_excl = uniq[i + horizon - 1] + pd.Timedelta(weeks=1)
        folds.append((train_end_excl, test_start, test_end_excl))
        i += step
    return folds


# -------------- weekly baseline --------------
def weekly_seasonal_naive(panel: pd.DataFrame, test_weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """y_w = average of last 3 same-day-of-month-ish weeks. Use simple lag-1 for weekly."""
    out = []
    for code, g in panel.groupby("branch_code", sort=False):
        g = g.set_index("week_start").sort_index()
        last_obs = g.iloc[-1]["revenue"] if len(g) else 0.0
        for w in test_weeks:
            out.append({"branch_code": code, "week_start": w, "y_pred": float(last_obs)})
    return pd.DataFrame(out)


def weekly_mean_naive(panel: pd.DataFrame, test_weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """y_w = mean of last 4 weeks. A smoother baseline than lag-1."""
    out = []
    for code, g in panel.groupby("branch_code", sort=False):
        g = g.sort_values("week_start")
        ref = g["revenue"].tail(4).mean() if len(g) else 0.0
        for w in test_weeks:
            out.append({"branch_code": code, "week_start": w, "y_pred": float(ref)})
    return pd.DataFrame(out)


# -------------- weekly Prophet --------------
def _weekly_holidays_df() -> pd.DataFrame:
    ramadan = pd.DataFrame({
        "holiday": "ramadan_week", "ds": RAMADAN_WEEK_STARTS,
        "lower_window": 0, "upper_window": 0,
    })
    eid = pd.DataFrame({
        "holiday": "eid_week", "ds": EID_WEEK_STARTS,
        "lower_window": 0, "upper_window": 0,
    })
    # Payday weeks: the week that contains the 1st of each month, across every
    # year in our window (each year's payday signal is then learned once).
    payday_weeks = []
    for year in (2024, 2025, 2026, 2027):
        for month in range(1, 13):
            target = pd.Timestamp(f"{year}-{month:02d}-01")
            mon = target - pd.Timedelta(days=target.dayofweek)
            payday_weeks.append(mon)
    payday = pd.DataFrame({
        "holiday": "payday_week", "ds": pd.to_datetime(sorted(set(payday_weeks))),
        "lower_window": 0, "upper_window": 0,
    })
    # School holiday weeks: any week-start covered by SCHOOL_HOLIDAY_DAYS.
    school_starts = _week_starts_covering(SCHOOL_HOLIDAY_DAYS)
    school = pd.DataFrame({
        "holiday": "school_break", "ds": pd.to_datetime(school_starts),
        "lower_window": 0, "upper_window": 0,
    })
    return pd.concat([ramadan, eid, payday, school], ignore_index=True)


def _fit_predict_weekly_prophet(train: pd.DataFrame, future_weeks: pd.DatetimeIndex,
                                cps: float, mode: str = "additive") -> pd.DataFrame:
    from prophet import Prophet
    import contextlib, io
    # With >=52 weeks of training data we let Prophet fit yearly seasonality.
    model = Prophet(
        weekly_seasonality=False, yearly_seasonality="auto", daily_seasonality=False,
        seasonality_mode=mode, holidays=_weekly_holidays_df(),
        changepoint_prior_scale=cps,
    )
    # Monthly Fourier seasonality (period in weeks ≈ 4.33). 3 Fourier terms is enough.
    model.add_seasonality(name="monthly", period=4.345, fourier_order=3)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model.fit(train)
        return model.predict(pd.DataFrame({"ds": future_weeks}))


def _pick_weekly_cps(train: pd.DataFrame) -> tuple[float, str]:
    """Hold out last 4 training weeks, grid-search cps × {additive, multiplicative}."""
    if len(train) < 12:
        return 0.1, "additive"
    cutoff = train["ds"].max() - pd.Timedelta(weeks=3)
    inner_train = train[train["ds"] < cutoff]
    inner_val = train[train["ds"] >= cutoff]
    best = (0.1, "additive")
    best_wape = float("inf")
    for mode in ("additive", "multiplicative"):
        for cps in (0.05, 0.1, 0.2):
            try:
                if mode == "multiplicative" and (inner_train["y"] <= 0).any():
                    continue
                fcst = _fit_predict_weekly_prophet(inner_train, pd.DatetimeIndex(inner_val["ds"]), cps, mode)
                merged = inner_val[["ds", "y"]].merge(fcst[["ds", "yhat"]], on="ds")
                err = (merged["y"] - merged["yhat"]).abs().sum()
                denom = merged["y"].abs().sum() or 1.0
                w = err / denom
                if w < best_wape:
                    best_wape, best = w, (cps, mode)
            except Exception:
                continue
    return best


def weekly_prophet(panel: pd.DataFrame, test_weeks: pd.DatetimeIndex,
                   *, return_intervals: bool = False) -> pd.DataFrame:
    """Simple additive Prophet per branch. Per-branch tuning over-fit on 21 obs;
    fixed cps=0.1 generalizes better.

    With return_intervals=True the output also carries Prophet's 80% prediction band
    (`y_lower`, `y_upper`). Used downstream as the source of uncertainty bands on the
    ensemble3 forecast — LGBM and naive_mean4 are point models, so we anchor the band
    on Prophet and recenter it on the ensemble3 mean."""
    from prophet import Prophet
    import contextlib, io
    holidays_df = _weekly_holidays_df()
    out = []
    for code, g in panel.groupby("branch_code", sort=False):
        train = g.rename(columns={"week_start": "ds", "revenue": "y"})[["ds", "y"]]
        model = Prophet(
            weekly_seasonality=False, yearly_seasonality="auto", daily_seasonality=False,
            seasonality_mode="additive", holidays=holidays_df,
            changepoint_prior_scale=0.1,
            interval_width=0.80,
        )
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model.fit(train)
            fcst = model.predict(pd.DataFrame({"ds": test_weeks}))
        cols = ["ds", "yhat"] + (["yhat_lower", "yhat_upper"] if return_intervals else [])
        for row in fcst[cols].itertuples(index=False):
            rec = {"branch_code": code, "week_start": pd.Timestamp(row.ds), "y_pred": float(row.yhat)}
            if return_intervals:
                rec["y_lower"] = float(row.yhat_lower)
                rec["y_upper"] = float(row.yhat_upper)
            out.append(rec)
    return pd.DataFrame(out)


# -------------- weekly LightGBM --------------
def _weekly_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["branch_code", "week_start"]).copy()
    df["log_revenue"] = np.log1p(df["revenue"].clip(lower=0))
    df["log_tx"]      = np.log1p(df["tx_count"].clip(lower=0))

    # Revenue lags + rolling stats. lag_52 = same ISO week last year.
    g_rev = df.groupby("branch_code", sort=False)["log_revenue"]
    for k in [1, 2, 3, 4, 52]:
        df[f"lag_{k}"] = g_rev.shift(k)
    df["rmean_4"]  = g_rev.shift(1).groupby(df["branch_code"]).transform(lambda s: s.rolling(4, min_periods=2).mean())
    df["rstd_4"]   = g_rev.shift(1).groupby(df["branch_code"]).transform(lambda s: s.rolling(4, min_periods=2).std())
    df["rmean_8"]  = g_rev.shift(1).groupby(df["branch_code"]).transform(lambda s: s.rolling(8, min_periods=3).mean())
    df["last_to_mean4"] = df["lag_1"] - df["rmean_4"]   # momentum signal
    # Year-over-year delta (current vs same week last year). NaN until we have history.
    df["yoy_log_delta"] = df["lag_1"] - df["lag_52"]

    # Transaction-count lags (independent signal from revenue).
    g_tx = df.groupby("branch_code", sort=False)["log_tx"]
    df["tx_lag_1"] = g_tx.shift(1)
    df["tx_lag_2"] = g_tx.shift(2)
    df["tx_rmean_4"] = g_tx.shift(1).groupby(df["branch_code"]).transform(lambda s: s.rolling(4, min_periods=2).mean())

    # Intra-week std lag (volatility memory).
    df["intraweek_std_lag1"] = df.groupby("branch_code", sort=False)["intraweek_std"].shift(1)

    # Branch-level long-run average (overall scale prior).
    df["branch_mean_logrev"] = df.groupby("branch_code", sort=False)["log_revenue"].transform("mean")

    # Cross-branch median for the same week — peer signal.
    df["peer_median_lag1"] = df.groupby("week_start", sort=False)["log_revenue"].transform("median")
    df["peer_median_lag1"] = df.groupby("branch_code", sort=False)["peer_median_lag1"].shift(1)

    # Calendar / holiday.
    df["is_ramadan_week"] = (df["ramadan_days"] > 0).astype(int)
    df["ramadan_intensity"] = df["ramadan_days"] / 7.0
    df["is_eid_week"]     = (df["eid_days"] > 0).astype(int)
    df["eid_intensity"]   = df["eid_days"] / 7.0
    df["payday_intensity"] = df["payday_days"] / 7.0
    df["is_payday_week"]   = (df["payday_days"] >= 3).astype(int)
    df["school_holiday_intensity"] = df["school_holiday_days"] / 7.0
    df["is_school_holiday_week"]   = (df["school_holiday_days"] >= 3).astype(int)
    df["month"]           = df["week_start"].dt.month
    df["week_of_month"]   = ((df["week_start"].dt.day - 1) // 7 + 1).astype(int)
    return df


WEEKLY_LGBM_PARAMS = {
    "objective": "regression_l1",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 5,
    "feature_fraction": 0.9,
    "verbosity": -1,
    "n_estimators": 400,
}


def weekly_lgbm(panel: pd.DataFrame, test_weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """Direct multi-output per horizon h in 1..H. Cleaner than recursive for weekly."""
    import lightgbm as lgb

    horizon = len(test_weeks)
    feats = _weekly_features(panel)
    feature_cols = [
        "branch_code", "month", "week_of_month",
        "is_ramadan_week", "ramadan_intensity", "is_eid_week", "eid_intensity",
        "is_payday_week", "payday_intensity",
        "is_school_holiday_week", "school_holiday_intensity",
        "lag_1", "lag_2", "lag_3", "lag_4", "lag_52", "yoy_log_delta",
        "rmean_4", "rstd_4", "rmean_8", "last_to_mean4",
        "tx_lag_1", "tx_lag_2", "tx_rmean_4",
        "intraweek_std_lag1", "branch_mean_logrev", "peer_median_lag1",
    ]
    feats["branch_code"] = feats["branch_code"].astype("category")

    # Only require lag_4 to exist; lag_52 / yoy_log_delta are NaN-tolerant in LGBM.
    feats = feats.dropna(subset=["lag_4"]).copy()
    feats = feats.sort_values(["branch_code", "week_start"]).reset_index(drop=True)
    feats["y_log"] = feats["log_revenue"]
    for h in range(1, horizon + 1):
        feats[f"y_h{h}"] = feats.groupby("branch_code", sort=False, observed=True)["y_log"].shift(-h)

    issue_week = test_weeks[0] - pd.Timedelta(weeks=1)
    issue_rows = feats[feats["week_start"] == issue_week].copy()
    if issue_rows.empty:
        issue_rows = feats.groupby("branch_code", as_index=False, observed=True).tail(1)

    out = []
    for h in range(1, horizon + 1):
        train_df = feats.dropna(subset=[f"y_h{h}"]).copy()
        X = train_df[feature_cols]
        y = train_df[f"y_h{h}"]
        if len(y) < 8:
            # Fallback: predict last observed log_revenue.
            for code, last in zip(issue_rows["branch_code"].astype(int), issue_rows["log_revenue"]):
                out.append({"branch_code": int(code), "week_start": issue_week + pd.Timedelta(weeks=h),
                            "y_pred": float(np.expm1(last))})
            continue
        model = lgb.LGBMRegressor(**WEEKLY_LGBM_PARAMS)
        model.fit(X, y, categorical_feature=["branch_code"], callbacks=[lgb.log_evaluation(0)])
        y_pred_log = model.predict(issue_rows[feature_cols].fillna(0))
        pred_week = issue_week + pd.Timedelta(weeks=h)
        for code, p_log in zip(issue_rows["branch_code"].astype(int), y_pred_log):
            out.append({"branch_code": int(code), "week_start": pred_week, "y_pred": float(np.expm1(p_log))})
    return pd.DataFrame(out)


# -------------- evaluation --------------
def _run_fold(model_fn, panel: pd.DataFrame, train_end_excl: pd.Timestamp,
              test_start: pd.Timestamp, test_end_excl: pd.Timestamp) -> pd.DataFrame:
    train = panel[panel["week_start"] < train_end_excl]
    test_weeks = pd.date_range(test_start, test_end_excl - pd.Timedelta(weeks=1), freq="W-MON")
    preds = model_fn(train, test_weeks)
    truth = panel[panel["week_start"].isin(test_weeks)][["branch_code", "branch_name_en", "week_start", "revenue"]]
    truth = truth.rename(columns={"revenue": "y_true"})
    merged = truth.merge(preds, on=["branch_code", "week_start"], how="left")
    merged["fold"] = train_end_excl
    return merged


def evaluate_weekly() -> dict:
    daily = get_panel()
    panel = build_weekly_panel(daily)
    # Need ≥52 weeks of training so lag_52 + Prophet yearly seasonality work.
    folds = weekly_folds(panel["week_start"], horizon=HORIZON_WEEKS, first_train_weeks=78, step=1)
    if not folds:
        raise RuntimeError("not enough weekly history for CV")

    models = {
        "naive_lag1":      weekly_seasonal_naive,
        "naive_mean4":     weekly_mean_naive,
        "prophet_weekly":  weekly_prophet,
        "lgbm_weekly":     weekly_lgbm,
    }
    all_preds: dict[str, pd.DataFrame] = {}
    for name, fn in models.items():
        print(f"[{name}] running {len(folds)} weekly folds...")
        rows = [_run_fold(fn, panel, *f) for f in folds]
        all_preds[name] = pd.concat(rows, ignore_index=True)

    # 2-model ensemble (Prophet + LGBM).
    p, l = all_preds["prophet_weekly"], all_preds["lgbm_weekly"]
    ens2 = p.merge(l[["branch_code", "week_start", "y_pred"]],
                   on=["branch_code", "week_start"], suffixes=("_p", "_l"))
    ens2["y_pred"] = (ens2["y_pred_p"] + ens2["y_pred_l"]) / 2
    all_preds["ensemble_weekly"] = ens2.drop(columns=["y_pred_p", "y_pred_l"])

    # 3-model ensemble (Prophet + LGBM + naive_mean4). naive_mean4 is a strong stable signal.
    m = all_preds["naive_mean4"]
    ens3 = ens2[["branch_code", "branch_name_en", "week_start", "y_true", "fold", "y_pred"]] \
        .rename(columns={"y_pred": "y_pred_ens2"})
    ens3 = ens3.merge(m[["branch_code", "week_start", "y_pred"]].rename(columns={"y_pred": "y_pred_m"}),
                      on=["branch_code", "week_start"])
    ens3["y_pred"] = (ens3["y_pred_ens2"] * 2 + ens3["y_pred_m"]) / 3  # weights: P,L,M = 1,1,1
    all_preds["ensemble3_weekly"] = ens3.drop(columns=["y_pred_ens2", "y_pred_m"])

    report: dict = {
        "config": {
            "horizon_weeks": HORIZON_WEEKS,
            "first_train_weeks": 78,
            "folds": len(folds),
            "first_test_week": str(folds[0][1].date()),
            "last_test_week": str((folds[-1][2] - pd.Timedelta(weeks=1)).date()),
            "weekly_observations_total": int(panel.groupby("branch_code").size().min()),
        },
        "per_branch": {},
        "overall": {},
    }
    for name, df in all_preds.items():
        by_branch = summarize(df, group="branch_name_en")
        overall = summarize(df).iloc[0].to_dict()
        for _, row in by_branch.iterrows():
            b = row["branch_name_en"]
            report["per_branch"].setdefault(b, {})[name] = {
                "mae": round(row["mae"], 2),
                "wape": round(row["wape"], 4),
                "smape": round(row["smape"], 4),
                "n": int(row["n"]),
            }
        report["overall"][name] = {
            "mae": round(overall["mae"], 2),
            "wape": round(overall["wape"], 4),
            "smape": round(overall["smape"], 4),
            "n": int(overall["n"]),
        }
    return report


# -------------- production forecast --------------
def next_4_weeks_forecast() -> dict:
    daily = get_panel()
    panel = build_weekly_panel(daily)
    last_week = panel["week_start"].max()
    test_weeks = pd.date_range(last_week + pd.Timedelta(weeks=1), periods=HORIZON_WEEKS, freq="W-MON")

    p = weekly_prophet(panel, test_weeks, return_intervals=True).rename(
        columns={"y_pred": "prophet", "y_lower": "prophet_lower", "y_upper": "prophet_upper"})
    l = weekly_lgbm(panel, test_weeks).rename(columns={"y_pred": "lgbm"})
    m = weekly_mean_naive(panel, test_weeks).rename(columns={"y_pred": "mean4"})
    merged = p.merge(l, on=["branch_code", "week_start"]) \
              .merge(m, on=["branch_code", "week_start"])
    merged["ensemble"]  = (merged["prophet"] + merged["lgbm"]) / 2
    merged["ensemble3"] = (merged["prophet"] + merged["lgbm"] + merged["mean4"]) / 3
    # Recenter Prophet's 80% band on ensemble3 — keep band half-width, shift center.
    half_width = (merged["prophet_upper"] - merged["prophet_lower"]) / 2.0
    merged["lower"] = (merged["ensemble3"] - half_width).clip(lower=0)
    merged["upper"] = merged["ensemble3"] + half_width
    branches = panel[["branch_code", "branch_name_en"]].drop_duplicates()
    merged = merged.merge(branches, on="branch_code", how="left")

    by_branch: dict[str, dict] = {}
    for b, g in merged.groupby("branch_name_en", sort=False):
        g = g.sort_values("week_start")
        by_branch[b] = {
            "branch_code": int(g["branch_code"].iloc[0]),
            "week_starts": [d.strftime("%Y-%m-%d") for d in g["week_start"]],
            "prophet":  [round(float(v), 2) for v in g["prophet"]],
            "lgbm":     [round(float(v), 2) for v in g["lgbm"]],
            "mean4":    [round(float(v), 2) for v in g["mean4"]],
            "ensemble": [round(float(v), 2) for v in g["ensemble"]],
            "ensemble3":[round(float(v), 2) for v in g["ensemble3"]],
            "lower":    [round(float(v), 2) for v in g["lower"]],
            "upper":    [round(float(v), 2) for v in g["upper"]],
            "total_ensemble_4w":  round(float(g["ensemble"].sum()), 2),
            "total_ensemble3_4w": round(float(g["ensemble3"].sum()), 2),
            "total_lower_4w":     round(float(g["lower"].sum()), 2),
            "total_upper_4w":     round(float(g["upper"].sum()), 2),
        }
    return {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "history_last_week": last_week.strftime("%Y-%m-%d"),
        "forecast_weeks": [d.strftime("%Y-%m-%d") for d in test_weeks],
        "horizon_weeks": HORIZON_WEEKS,
        "target": "weekly_revenue_egp",
        "interval_width": 0.80,
        "interval_method": "prophet_80pct_recentered_on_ensemble3",
        "per_branch": by_branch,
        "company_total_ensemble_4w":  round(sum(v["total_ensemble_4w"]  for v in by_branch.values()), 2),
        "company_total_ensemble3_4w": round(sum(v["total_ensemble3_4w"] for v in by_branch.values()), 2),
        "company_total_lower_4w":     round(sum(v["total_lower_4w"]     for v in by_branch.values()), 2),
        "company_total_upper_4w":     round(sum(v["total_upper_4w"]     for v in by_branch.values()), 2),
    }


if __name__ == "__main__":
    REPORTS_DIR.mkdir(exist_ok=True)

    print("=== Weekly CV ===")
    cv = evaluate_weekly()
    (REPORTS_DIR / "results_weekly.json").write_text(json.dumps(cv, indent=2))
    print(f"folds: {cv['config']['folds']}  (each branch has {cv['config']['weekly_observations_total']} weekly obs)")
    print(f"\n{'model':<18s}  {'WAPE':>6s}  {'MAE (EGP)':>10s}")
    for name, m in cv["overall"].items():
        print(f"{name:<18s}  {m['wape']:>6.3f}  {m['mae']:>10,.0f}")
    print(f"\nPer-branch WAPE (ensemble):")
    for b, mods in cv["per_branch"].items():
        print(f"  {b:<20s} {mods['ensemble_weekly']['wape']:.3f}")

    print("\n=== Next 4 weeks forecast ===")
    fc = next_4_weeks_forecast()
    (REPORTS_DIR / "next_4w.json").write_text(json.dumps(fc, indent=2, ensure_ascii=False))
    print(f"weeks: {fc['forecast_weeks'][0]} to {fc['forecast_weeks'][-1]}")
    print(f"company total (ensemble, 4w): {fc['company_total_ensemble_4w']:,.0f} EGP")
    for b, v in sorted(fc["per_branch"].items(), key=lambda kv: -kv[1]["total_ensemble_4w"]):
        print(f"  {b:<20s} {v['total_ensemble_4w']:>14,.0f}")
