"""Inference helpers loaded once and reused by the API."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from .data_loader import FEATURE_COLUMNS, build_feature_table

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
DEMAND_PATH = ARTIFACTS_DIR / "demand_classifier.joblib"
DEAD_STOCK_PATH = ARTIFACTS_DIR / "dead_stock_classifier.joblib"


class ModelNotTrainedError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_demand():
    if not DEMAND_PATH.exists():
        raise ModelNotTrainedError(
            "Demand model not found. Run: python -m src.train_demand")
    return joblib.load(DEMAND_PATH)


@lru_cache(maxsize=1)
def _load_dead_stock():
    if not DEAD_STOCK_PATH.exists():
        raise ModelNotTrainedError(
            "Dead-stock model not found. Run: python -m src.train_dead_stock")
    return joblib.load(DEAD_STOCK_PATH)


@lru_cache(maxsize=1)
def _feature_table() -> pd.DataFrame:
    return build_feature_table()


def _vector_from_payload(payload: dict) -> pd.DataFrame:
    row = {col: float(payload.get(col, 0.0) or 0.0) for col in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_demand_from_features(payload: dict) -> dict:
    bundle = _load_demand()
    pipe = bundle["pipeline"]
    X = _vector_from_payload(payload)
    label = pipe.predict(X)[0]
    proba = pipe.predict_proba(X)[0]
    return {
        "label": str(label),
        "probabilities": {str(c): float(round(p, 4)) for c, p in zip(pipe.classes_, proba)},
    }


def predict_dead_stock_from_features(payload: dict) -> dict:
    bundle = _load_dead_stock()
    pipe = bundle["pipeline"]
    X = _vector_from_payload(payload)
    proba = float(pipe.predict_proba(X)[0, 1])
    return {
        "is_dead_stock": bool(proba >= 0.5),
        "probability": round(proba, 4),
    }


def _lookup_features(code: str) -> dict | None:
    table = _feature_table()
    row = table[table["الكود"].astype(str) == str(code)]
    if row.empty:
        return None
    record = row.iloc[0].to_dict()
    return record


def predict_by_code(code: str) -> dict:
    record = _lookup_features(code)
    if record is None:
        return {"code": code, "found": False}
    demand = predict_demand_from_features(record)
    dead = predict_dead_stock_from_features(record)
    return {
        "code": code,
        "name": record.get("اسم الصنف"),
        "found": True,
        "features": {k: record.get(k) for k in FEATURE_COLUMNS},
        "demand": demand,
        "dead_stock": dead,
    }


def predict_batch(codes: Iterable[str]) -> list[dict]:
    return [predict_by_code(c) for c in codes]


def list_known_codes(limit: int = 20) -> list[str]:
    return _feature_table()["الكود"].astype(str).head(limit).tolist()
