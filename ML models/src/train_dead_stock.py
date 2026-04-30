"""Train the dead-stock predictor (binary classifier)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_loader import FEATURE_COLUMNS, build_feature_table
from .labels import label_dead_stock

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "dead_stock_classifier.joblib"
REPORT_PATH = REPORTS_DIR / "dead_stock_classifier_report.json"


def train(random_state: int = 42) -> dict:
    feats = build_feature_table()
    y = label_dead_stock(feats)
    X = feats[FEATURE_COLUMNS].fillna(0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            random_state=random_state,
        )),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    f1 = f1_score(y_test, y_pred)
    try:
        auc = float(roc_auc_score(y_test, y_proba))
    except ValueError:
        auc = None

    importances = dict(zip(FEATURE_COLUMNS,
                           pipe.named_steps["clf"].feature_importances_.round(4).tolist()))

    metrics = {
        "task": "dead_stock_prediction",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate_train": float(round(y_train.mean(), 4)),
        "f1_dead_stock": float(round(f1, 4)),
        "roc_auc": auc,
        "accuracy": float(round(report["accuracy"], 4)),
        "per_class": {"not_dead": report.get("0", {}), "dead": report.get("1", {})},
        "confusion_matrix": cm,
        "feature_importances": importances,
        "feature_columns": FEATURE_COLUMNS,
    }

    joblib.dump({"pipeline": pipe, "feature_columns": FEATURE_COLUMNS,
                 "classes": ["not_dead", "dead"]}, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    m = train()
    print(json.dumps({k: v for k, v in m.items() if k != "feature_importances"}, indent=2))
