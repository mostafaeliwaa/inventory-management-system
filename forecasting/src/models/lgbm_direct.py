"""Direct multi-output LightGBM: train one model per horizon day h in {1..H}.

For each h, we shift the target by h days so the model learns to predict
revenue h days ahead given features known at time t. Avoids recursive
error compounding at the cost of training H models.

Trains on log1p(revenue) with light winsorization, same as the recursive variant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.src.features import build_features, feature_columns

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
WINSORIZE_LOWER_Q = 0.01


def _winsorize_per_branch(panel: pd.DataFrame, target: str) -> pd.DataFrame:
    out = panel.copy()
    lows = out.groupby("branch_code")[target].transform(lambda s: s.quantile(WINSORIZE_LOWER_Q))
    out[target] = out[target].clip(lower=lows).clip(lower=0.0)
    return out


def _to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


def lgbm_direct_forecast(panel: pd.DataFrame, test_dates: pd.DatetimeIndex, target: str = "revenue") -> pd.DataFrame:
    import lightgbm as lgb

    horizon = len(test_dates)
    panel_clean = _winsorize_per_branch(panel, target)
    panel_clean[target] = np.log1p(panel_clean[target])

    feats = build_features(panel_clean, target=target)
    cols = feature_columns(target=target)
    # Drop rows with NaN in lag_28 (warmup).
    feats = feats.dropna(subset=[f"{target}_lag_28"]).copy()
    feats = _to_categorical(feats)

    # For each branch, the per-horizon target is target shifted by -h.
    feats = feats.sort_values(["branch_code", "posting_date"]).reset_index(drop=True)
    feats["y_log"] = feats[target]
    for h in range(1, horizon + 1):
        feats[f"y_h{h}"] = feats.groupby("branch_code", sort=False)["y_log"].shift(-h)

    out_records: list[dict] = []
    # Predict from the last in-sample timestamp ("issue date"): the row whose features
    # represent everything known up to the day before test_dates[0].
    issue_date = test_dates[0] - pd.Timedelta(days=1)
    issue_rows = feats[feats["posting_date"] == issue_date].copy()
    if issue_rows.empty:
        # Fall back to the latest available row per branch (panel may not include issue_date).
        issue_rows = feats.groupby("branch_code", as_index=False).tail(1)

    for h in range(1, horizon + 1):
        train_df = feats.dropna(subset=[f"y_h{h}"]).copy()
        X_train = train_df[cols]
        y_train = train_df[f"y_h{h}"]

        cutoff = train_df["posting_date"].max() - pd.Timedelta(days=14)
        val_mask = train_df["posting_date"] > cutoff
        X_fit, y_fit = X_train[~val_mask], y_train[~val_mask]
        X_val, y_val = X_train[val_mask],  y_train[val_mask]

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)] if len(X_val) > 0 else None,
            categorical_feature=CATEGORICAL,
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)] if len(X_val) > 0 else None,
        )

        X_pred = issue_rows[cols].fillna(0)
        y_pred_log = model.predict(X_pred)
        pred_date = issue_date + pd.Timedelta(days=h)
        for code, p_log in zip(issue_rows["branch_code"].astype(int), y_pred_log):
            out_records.append({
                "branch_code": int(code),
                "posting_date": pred_date,
                "y_pred": float(np.expm1(p_log)),
            })

    return pd.DataFrame(out_records)
