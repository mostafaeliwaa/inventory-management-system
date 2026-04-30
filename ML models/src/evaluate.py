"""Robustness evaluation: hide the features that define the labels and
re-train, to measure how much the model generalizes beyond the rule.

Both label functions are deterministic over a small set of features
(`total_sales`, `stock_to_sales_ratio`, `sell_through`, `total_stock`).
A model with access to those features trivially scores 1.0. The real
question is: can the *other* features (branch spread, prices, margin,
concentration) recover the same decision? This script answers that.
"""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_loader import FEATURE_COLUMNS, build_feature_table
from .labels import DEMAND_CLASSES, label_dead_stock, label_demand

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
OUT = REPORTS_DIR / "robustness_report.json"

DEMAND_HIDDEN = ["total_sales", "max_branch_sales"]
DEAD_HIDDEN = ["total_sales", "total_stock", "stock_to_sales_ratio", "sell_through"]


def _eval_demand(feats, hidden):
    cols = [c for c in FEATURE_COLUMNS if c not in hidden]
    X = feats[cols].fillna(0.0)
    y = label_demand(feats)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                     random_state=42, n_jobs=-1))])
    pipe.fit(Xtr, ytr)
    yp = pipe.predict(Xte)
    rep = classification_report(yte, yp, output_dict=True, zero_division=0)
    return {
        "hidden": hidden,
        "features_used": cols,
        "macro_f1": round(f1_score(yte, yp, average="macro"), 4),
        "accuracy": round(rep["accuracy"], 4),
        "per_class": {c: rep[c] for c in DEMAND_CLASSES if c in rep},
    }


def _eval_dead(feats, hidden):
    cols = [c for c in FEATURE_COLUMNS if c not in hidden]
    X = feats[cols].fillna(0.0)
    y = label_dead_stock(feats)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", GradientBoostingClassifier(n_estimators=250, max_depth=3,
                                                         learning_rate=0.05, random_state=42))])
    pipe.fit(Xtr, ytr)
    yp = pipe.predict(Xte)
    yproba = pipe.predict_proba(Xte)[:, 1]
    rep = classification_report(yte, yp, output_dict=True, zero_division=0)
    try:
        auc = float(round(roc_auc_score(yte, yproba), 4))
    except ValueError:
        auc = None
    return {
        "hidden": hidden,
        "features_used": cols,
        "f1_dead": round(f1_score(yte, yp), 4),
        "roc_auc": auc,
        "accuracy": round(rep["accuracy"], 4),
        "per_class": {"not_dead": rep.get("0", {}), "dead": rep.get("1", {})},
    }


def run() -> dict:
    feats = build_feature_table()
    out = {
        "note": ("Labels are deterministic over a few features. We retrain "
                 "with those hidden to gauge generalization from the remaining signals."),
        "demand_robust": _eval_demand(feats, DEMAND_HIDDEN),
        "dead_stock_robust": _eval_dead(feats, DEAD_HIDDEN),
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
