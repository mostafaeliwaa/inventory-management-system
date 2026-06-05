"""Train and export the production global LightGBM forecaster.

Trains the full model (7 seeds × 2 objectives × 4 horizons = 56 LightGBM models)
on the complete dataset and serializes them to disk for production inference.

Output directory structure::

    forecasting/artifacts/production_model/
    ├── models_h1/
    │   ├── seed_11_l1.txt
    │   ├── seed_11_q60.txt
    │   ├── seed_23_l1.txt
    │   ...
    ├── models_h2/
    │   ...
    ├── models_h3/
    │   ...
    ├── models_h4/
    │   ...
    ├── feature_columns.json
    ├── model_config.json
    └── training_metadata.json

Usage::

    python -m forecasting.src.export_model
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecasting.src.data_loader import get_panel
from forecasting.src.weekly import build_weekly_panel
from forecasting.src.weekly_global import (
    CATEGORICAL,
    FOURIER_K,
    HORIZON_WEEKS,
    L1_PARAMS,
    LAGS,
    Q60_PARAMS,
    ROLL_WINDOWS,
    SEEDS,
    build_global_features,
    feature_columns,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "production_model"


def _train_and_save_models(
    feats: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Train all seed × objective × horizon models and save to disk.

    Args:
        feats: Feature-engineered DataFrame from ``build_global_features``.
        output_dir: Root directory for serialized models.

    Returns:
        Training statistics dict.
    """
    cols = feature_columns()

    # Prepare categorical columns.
    for c in CATEGORICAL:
        feats[c] = feats[c].astype("category")

    # Create target columns for each horizon.
    for h in range(1, HORIZON_WEEKS + 1):
        feats[f"y_h{h}"] = (
            feats.groupby("branch_code", sort=False, observed=True)["log_revenue"]
            .shift(-h)
        )

    stats = {
        "horizons": HORIZON_WEEKS,
        "seeds": SEEDS,
        "objectives": ["l1", "q60"],
        "models_per_horizon": len(SEEDS) * 2,
        "total_models": len(SEEDS) * 2 * HORIZON_WEEKS,
        "per_horizon": {},
    }

    for h in range(1, HORIZON_WEEKS + 1):
        h_dir = output_dir / f"models_h{h}"
        h_dir.mkdir(parents=True, exist_ok=True)

        # Prepare training data for this horizon.
        train_df = feats.dropna(subset=[f"y_h{h}", "lag_4"]).copy()
        train_df = train_df[train_df["weeks_since_open"] >= 2]

        X = train_df[cols]
        y = train_df[f"y_h{h}"]

        # Validation split: last 7 weeks.
        cutoff = train_df["week_start"].max() - pd.Timedelta(weeks=7)
        val_mask = train_df["week_start"] > cutoff
        X_fit, y_fit = X[~val_mask], y[~val_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        if len(X_fit) < 200 or len(X_val) < 16:
            X_fit, y_fit = X, y
            X_val, y_val = X.iloc[-min(32, len(X)):], y.iloc[-min(32, len(X)):]

        logger.info(
            "Horizon h=%d: train=%d, val=%d rows.",
            h, len(X_fit), len(X_val),
        )

        h_stats = {"train_rows": int(len(X_fit)), "val_rows": int(len(X_val)), "models": []}

        for seed in SEEDS:
            for obj_name, base_params in [("l1", L1_PARAMS), ("q60", Q60_PARAMS)]:
                params = {
                    **base_params,
                    "seed": seed,
                    "bagging_seed": seed,
                    "feature_fraction_seed": seed,
                }
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_fit, y_fit,
                    eval_set=[(X_val, y_val)],
                    categorical_feature=CATEGORICAL,
                    callbacks=[
                        lgb.early_stopping(120, verbose=False),
                        lgb.log_evaluation(0),
                    ],
                )

                # Save in LightGBM native text format (portable, fast to load).
                model_path = h_dir / f"seed_{seed}_{obj_name}.txt"
                model.booster_.save_model(str(model_path))

                h_stats["models"].append({
                    "seed": seed,
                    "objective": obj_name,
                    "n_estimators_used": model.best_iteration_ or model.n_estimators,
                    "path": str(model_path.relative_to(output_dir)),
                })

        stats["per_horizon"][f"h{h}"] = h_stats
        logger.info("  Saved %d models for horizon h=%d.", len(h_stats["models"]), h)

    return stats


def export_model(data_path: str | Path | None = None) -> Path:
    """Full export pipeline: load data → train → serialize.

    Args:
        data_path: Optional path to raw data. Defaults to the project's data file.

    Returns:
        Path to the output directory containing all serialized artifacts.
    """
    logger.info("Loading and preparing data...")
    daily = get_panel(data_path)
    weekly = build_weekly_panel(daily)
    feats = build_global_features(weekly)

    logger.info(
        "Data: %d weekly rows, %d branches, weeks %s to %s.",
        len(weekly),
        weekly["branch_code"].nunique(),
        weekly["week_start"].min().date(),
        weekly["week_start"].max().date(),
    )

    # Create output directory.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save feature column ordering.
    cols = feature_columns()
    cols_path = ARTIFACTS_DIR / "feature_columns.json"
    cols_path.write_text(json.dumps(cols, indent=2))
    logger.info("Saved feature_columns.json (%d features).", len(cols))

    # 2. Save model configuration.
    config = {
        "model_name": "global_lgbm_production",
        "model_type": "LightGBM",
        "architecture": "direct_multi_horizon_seed_bag",
        "description": (
            "Global LightGBM forecaster with 7-seed × 2-objective bagging "
            "and direct multi-horizon prediction. Predicts weekly branch revenue "
            "in log1p space."
        ),
        "horizon_weeks": HORIZON_WEEKS,
        "seeds": SEEDS,
        "objectives": {
            "l1": {k: v for k, v in L1_PARAMS.items() if k != "verbosity"},
            "q60": {k: v for k, v in Q60_PARAMS.items() if k != "verbosity"},
        },
        "categorical_features": CATEGORICAL,
        "lags": LAGS,
        "rolling_windows": ROLL_WINDOWS,
        "fourier_yearly_k": FOURIER_K,
        "target_transform": "log1p",
        "inverse_transform": "expm1",
        "required_input_columns": [
            "posting_date", "branch_code", "branch_name_en",
            "total_value", "cash_paid",
        ],
    }
    config_path = ARTIFACTS_DIR / "model_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    logger.info("Saved model_config.json.")

    # 3. Train and save all models.
    logger.info("Training %d models (this may take a few minutes)...", len(SEEDS) * 2 * HORIZON_WEEKS)
    train_stats = _train_and_save_models(feats, ARTIFACTS_DIR)

    # 4. Save training metadata.
    branches = weekly[["branch_code", "branch_name_en"]].drop_duplicates().sort_values("branch_code")
    metadata = {
        "exported_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "data_date_range": {
            "start": str(daily["posting_date"].min().date()),
            "end": str(daily["posting_date"].max().date()),
            "total_days": int((daily["posting_date"].max() - daily["posting_date"].min()).days),
        },
        "weekly_date_range": {
            "start": str(weekly["week_start"].min().date()),
            "end": str(weekly["week_start"].max().date()),
            "total_weeks": int(weekly["week_start"].nunique()),
        },
        "branches": [
            {"code": int(row["branch_code"]), "name": row["branch_name_en"]}
            for _, row in branches.iterrows()
        ],
        "n_branches": int(weekly["branch_code"].nunique()),
        "training_stats": train_stats,
        "accuracy_reference": {
            "source": "reports/accuracy_by_level.json",
            "per_branch_weekly_wape": 0.279,
            "per_branch_monthly_wape": 0.197,
            "company_weekly_wape": 0.188,
            "company_monthly_wape": 0.096,
            "per_branch_monthly_accuracy": 0.803,
            "company_monthly_accuracy": 0.904,
        },
    }
    meta_path = ARTIFACTS_DIR / "training_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    logger.info("Saved training_metadata.json.")

    logger.info("Export complete. Artifacts at: %s", ARTIFACTS_DIR)
    return ARTIFACTS_DIR


if __name__ == "__main__":
    output = export_model()
    print(f"\n{'='*60}")
    print(f"Production model exported to: {output}")
    print(f"{'='*60}")

    # Print summary.
    meta = json.loads((output / "training_metadata.json").read_text())
    config = json.loads((output / "model_config.json").read_text())
    print(f"\nModel: {config['model_name']}")
    print(f"Architecture: {config['architecture']}")
    print(f"Total models: {meta['training_stats']['total_models']}")
    print(f"Data range: {meta['data_date_range']['start']} -> {meta['data_date_range']['end']}")
    print(f"Branches: {meta['n_branches']}")
    for b in meta["branches"]:
        print(f"  [{b['code']}] {b['name']}")
    print(f"\nAccuracy (reference):")
    acc = meta["accuracy_reference"]
    print(f"  Per-branch monthly: {acc['per_branch_monthly_accuracy']*100:.1f}%")
    print(f"  Company monthly:    {acc['company_monthly_accuracy']*100:.1f}%")
