"""Train the demand (fast/medium/slow) classifier."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_loader import FEATURE_COLUMNS, build_feature_table
from .labels import DEMAND_CLASSES, label_demand

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "demand_classifier.joblib"
REPORT_PATH = REPORTS_DIR / "demand_classifier_report.json"


def train(random_state: int = 42) -> dict:
    feats = build_feature_table()
    y = label_demand(feats)
    X = feats[FEATURE_COLUMNS].fillna(0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=DEMAND_CLASSES).tolist()
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    importances = dict(zip(FEATURE_COLUMNS,
                           pipe.named_steps["clf"].feature_importances_.round(4).tolist()))

    metrics = {
        "task": "demand_classification",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classes": DEMAND_CLASSES,
        "macro_f1": float(round(macro_f1, 4)),
        "accuracy": float(round(report["accuracy"], 4)),
        "per_class": {c: report[c] for c in DEMAND_CLASSES if c in report},
        "confusion_matrix": cm,
        "feature_importances": importances,
        "feature_columns": FEATURE_COLUMNS,
    }

    joblib.dump({"pipeline": pipe, "feature_columns": FEATURE_COLUMNS,
                 "classes": DEMAND_CLASSES}, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    m = train()
    print(json.dumps({k: v for k, v in m.items() if k != "feature_importances"}, indent=2))
