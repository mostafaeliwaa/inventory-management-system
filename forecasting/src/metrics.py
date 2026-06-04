"""Forecast accuracy metrics. WAPE is the headline number."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error — robust when individual y_true can be small or zero."""
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def summarize(df: pd.DataFrame, group: str | None = None) -> pd.DataFrame:
    """df has columns y_true, y_pred (and optionally a `group` column)."""
    def _row(g: pd.DataFrame) -> pd.Series:
        y, p = g["y_true"].to_numpy(float), g["y_pred"].to_numpy(float)
        return pd.Series({"mae": mae(y, p), "mape": mape(y, p), "wape": wape(y, p), "smape": smape(y, p), "n": len(g)})
    if group is None:
        return _row(df).to_frame().T
    return df.groupby(group, sort=False).apply(_row).reset_index()
