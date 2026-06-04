"""Global LightGBM regressor, one model across all branches. Recursive 7-day prediction.

Trains on log1p(revenue) with winsorized lows to dampen refund spikes, predicts in
log space then back-transforms with expm1. Helps small-revenue branches a lot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.src.features import LAGS, ROLL_MEANS, ROLL_STDS, build_features, feature_columns

LGBM_PARAMS = {
    "objective": "regression_l1",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 10,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbosity": -1,
    "n_estimators": 800,
}
CATEGORICAL = ["branch_code", "dow"]
WINSORIZE_LOWER_Q = 0.01  # floor refund spikes per branch


def _winsorize_per_branch(panel: pd.DataFrame, target: str) -> pd.DataFrame:
    out = panel.copy()
    lows = out.groupby("branch_code")[target].transform(lambda s: s.quantile(WINSORIZE_LOWER_Q))
    out[target] = out[target].clip(lower=lows)
    # log1p requires non-negative; clip any residual negatives to 0.
    out[target] = out[target].clip(lower=0.0)
    return out


def _to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


def lgbm_forecast(panel: pd.DataFrame, test_dates: pd.DatetimeIndex, target: str = "revenue") -> pd.DataFrame:
    """Fit on `panel` history (log1p, winsorized), recursively predict each day in test_dates.

    `panel` must contain the full history up to (but not including) test_dates[0].
    """
    import lightgbm as lgb

    # Train on a winsorized, log-transformed copy. Keep original for the lag history
    # we feed into recursive prediction so the model sees realistic recent values.
    panel_clean = _winsorize_per_branch(panel, target)
    panel_clean[target] = np.log1p(panel_clean[target])

    feats = build_features(panel_clean, target=target)
    cols = feature_columns(target=target)
    train_df = feats.dropna(subset=[f"{target}_lag_28"]).copy()
    train_df = _to_categorical(train_df)

    X_train = train_df[cols]
    y_train = train_df[target]

    cutoff = train_df["posting_date"].max() - pd.Timedelta(days=14)
    val_mask = train_df["posting_date"] > cutoff
    X_fit, y_fit = X_train[~val_mask], y_train[~val_mask]
    X_val, y_val = X_train[val_mask],  y_train[val_mask]

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_fit, y_fit,
        eval_set=[(X_val, y_val)],
        categorical_feature=CATEGORICAL,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    # History for recursive prediction uses winsorized + log-transformed series so lags match training space.
    history = panel_clean[["branch_code", "branch_name_en", "posting_date", target,
                           "tx_count", "cash_paid", "cash_share", "was_open"]].copy()
    out = []
    for d in test_dates:
        branches = history[["branch_code", "branch_name_en"]].drop_duplicates()
        placeholder = branches.assign(
            posting_date=d, **{target: float("nan"), "tx_count": 0, "cash_paid": 0.0,
                                "cash_share": 0.0, "was_open": 1},
        )
        extended = pd.concat([history, placeholder], ignore_index=True)
        feats_day = build_features(extended, target=target)
        row_d = feats_day[feats_day["posting_date"] == d].copy()
        row_d = _to_categorical(row_d)
        X_d = row_d[cols].fillna(0)
        y_pred_log = model.predict(X_d)
        # Back-transform to revenue space for output; keep log for the next-day lag.
        for code, p_log in zip(row_d["branch_code"].astype(int), y_pred_log):
            out.append({"branch_code": int(code), "posting_date": d, "y_pred": float(np.expm1(p_log))})
        pred_log_by_code = {int(c): float(p) for c, p in zip(row_d["branch_code"].astype(int), y_pred_log)}
        new_rows = placeholder.copy()
        new_rows[target] = new_rows["branch_code"].astype(int).map(pred_log_by_code)
        history = pd.concat([history, new_rows], ignore_index=True)

    return pd.DataFrame(out)
