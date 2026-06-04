"""Walk-forward CV: expanding window, fixed-length forecast horizon."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_end: pd.Timestamp     # exclusive
    test_start: pd.Timestamp    # inclusive
    test_end: pd.Timestamp      # exclusive


def walk_forward_folds(
    dates: pd.Series,
    *,
    horizon: int = 7,
    first_train_days: int = 60,
    step: int = 7,
) -> list[Fold]:
    days = pd.Index(sorted(pd.to_datetime(dates.unique())))
    if len(days) < first_train_days + horizon:
        raise ValueError(f"need at least {first_train_days + horizon} days, got {len(days)}")
    folds: list[Fold] = []
    i = first_train_days
    fid = 0
    while i + horizon <= len(days):
        folds.append(Fold(
            fold_id=fid,
            train_end=days[i],
            test_start=days[i],
            test_end=days[i + horizon - 1] + pd.Timedelta(days=1),
        ))
        i += step
        fid += 1
    return folds
