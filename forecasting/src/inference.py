"""Production inference module — load serialized models, preprocess, predict.

This is the end-to-end production pipeline::

    Raw sales data (DataFrame / CSV)
        → preprocess_for_model.preprocess()
        → load serialized LightGBM models
        → predict for each horizon h in 1..4
        → average across 14 models (7 seeds × 2 objectives) in log-space
        → expm1 back to EGP
        → structured forecast dict

Usage::

    # As a library
    from forecasting.src.inference import ProductionForecaster
    forecaster = ProductionForecaster()
    result = forecaster.predict_from_csv("data/merged_new_sales.csv")

    # As a CLI
    python -m forecasting.src.inference [--input data.csv]
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecasting.src.preprocess_for_model import preprocess, preprocess_from_csv
from forecasting.src.weekly_global import CATEGORICAL

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "production_model"


class ProductionForecaster:
    """Load serialized LightGBM models and run production forecasts.

    Attributes:
        model_dir: Path to the production_model artifacts directory.
        config: Model configuration dict (from model_config.json).
        metadata: Training metadata dict (from training_metadata.json).
        feature_cols: Ordered list of feature column names.
        models: Dict mapping ``(horizon, seed, objective)`` → loaded Booster.
    """

    def __init__(self, model_dir: str | Path | None = None) -> None:
        """Initialize the forecaster by loading all serialized models.

        Args:
            model_dir: Path to the production_model directory.
                       Defaults to ``forecasting/artifacts/production_model/``.
        """
        self.model_dir = Path(model_dir) if model_dir else ARTIFACTS_DIR
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load config, metadata, feature columns, and all model files."""
        # Load configuration files.
        config_path = self.model_dir / "model_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Model config not found at {config_path}. "
                "Run `python -m forecasting.src.export_model` first."
            )
        self.config = json.loads(config_path.read_text())

        meta_path = self.model_dir / "training_metadata.json"
        self.metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        cols_path = self.model_dir / "feature_columns.json"
        self.feature_cols = json.loads(cols_path.read_text())

        # Load all LightGBM models.
        self.models: dict[tuple[int, int, str], lgb.Booster] = {}
        horizon_weeks = self.config["horizon_weeks"]
        seeds = self.config["seeds"]
        objectives = ["l1", "q60"]

        n_loaded = 0
        for h in range(1, horizon_weeks + 1):
            for seed in seeds:
                for obj in objectives:
                    model_path = self.model_dir / f"models_h{h}" / f"seed_{seed}_{obj}.txt"
                    if not model_path.exists():
                        logger.warning("Missing model file: %s", model_path)
                        continue
                    booster = lgb.Booster(model_file=str(model_path))
                    self.models[(h, seed, obj)] = booster
                    n_loaded += 1

        expected = horizon_weeks * len(seeds) * len(objectives)
        logger.info("Loaded %d / %d models from %s.", n_loaded, expected, self.model_dir)
        if n_loaded < expected:
            logger.warning(
                "Some models are missing (%d / %d). Predictions will use available models.",
                n_loaded, expected,
            )

    def predict(
        self,
        raw_df: pd.DataFrame,
    ) -> dict:
        """Run the full prediction pipeline on raw transactional data.

        Args:
            raw_df: Raw DataFrame with columns:
                    posting_date, branch_code, branch_name_en, total_value, cash_paid.

        Returns:
            Structured forecast dict with per-branch 4-week predictions.
        """
        # Step 1: Preprocess.
        features_df, weekly_panel = preprocess(raw_df, return_weekly_panel=True)

        # Step 2: Prepare feature matrix for the issue row (last available week per branch).
        for c in CATEGORICAL:
            if c in features_df.columns:
                features_df[c] = features_df[c].astype("category")

        # The issue rows = the last week per branch (the "current" state from which
        # we forecast forward).
        issue_rows = (
            features_df
            .sort_values(["branch_code", "week_start"])
            .groupby("branch_code", as_index=False, observed=True)
            .tail(1)
        )

        if issue_rows.empty:
            raise ValueError("No valid issue rows found. Data may be insufficient.")

        last_week = issue_rows["week_start"].max()

        # Step 3: Predict for each horizon.
        horizon_weeks = self.config["horizon_weeks"]
        seeds = self.config["seeds"]
        objectives = ["l1", "q60"]

        all_preds: list[dict] = []
        for h in range(1, horizon_weeks + 1):
            X_issue = issue_rows[self.feature_cols].copy()

            # Average across all seed × objective models.
            preds_log = np.zeros(len(X_issue), dtype=float)
            n_models = 0
            for seed in seeds:
                for obj in objectives:
                    key = (h, seed, obj)
                    if key not in self.models:
                        continue
                    booster = self.models[key]
                    preds_log += booster.predict(X_issue)
                    n_models += 1

            if n_models == 0:
                logger.error("No models available for horizon h=%d. Skipping.", h)
                continue

            preds_log /= n_models
            pred_week = last_week + pd.Timedelta(weeks=h)

            for code, p_log in zip(
                issue_rows["branch_code"].astype(int), preds_log
            ):
                all_preds.append({
                    "branch_code": int(code),
                    "week_start": pred_week,
                    "y_pred": float(np.expm1(p_log)),
                    "y_pred_log": float(p_log),
                })

        preds_df = pd.DataFrame(all_preds)

        # Step 4: Build structured output.
        branches = weekly_panel[["branch_code", "branch_name_en"]].drop_duplicates()
        preds_df = preds_df.merge(branches, on="branch_code", how="left")

        by_branch: dict[str, dict] = {}
        for branch_name, g in preds_df.groupby("branch_name_en", sort=False):
            g = g.sort_values("week_start")
            by_branch[str(branch_name)] = {
                "branch_code": int(g["branch_code"].iloc[0]),
                "week_starts": [d.strftime("%Y-%m-%d") for d in g["week_start"]],
                "y_pred": [round(float(v), 2) for v in g["y_pred"]],
                "total_4w": round(float(g["y_pred"].sum()), 2),
            }

        forecast_weeks = sorted(preds_df["week_start"].unique())

        return {
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "model": self.config["model_name"],
            "architecture": self.config["architecture"],
            "history_last_week": last_week.strftime("%Y-%m-%d"),
            "forecast_weeks": [d.strftime("%Y-%m-%d") for d in forecast_weeks],
            "horizon_weeks": horizon_weeks,
            "target": "weekly_revenue_egp",
            "target_transform": self.config["target_transform"],
            "n_models_per_horizon": len(seeds) * len(objectives),
            "per_branch": by_branch,
            "company_total_4w": round(
                sum(v["total_4w"] for v in by_branch.values()), 2
            ),
            "accuracy_reference": self.metadata.get("accuracy_reference", {}),
        }

    def predict_from_csv(self, csv_path: str | Path) -> dict:
        """Convenience: read CSV then predict.

        Args:
            csv_path: Path to CSV file with raw transactional data.

        Returns:
            Structured forecast dict.
        """
        logger.info("Reading %s ...", csv_path)
        raw_df = pd.read_csv(csv_path, parse_dates=["posting_date"])
        return self.predict(raw_df)

    def get_model_info(self) -> dict:
        """Return model metadata for API consumers."""
        return {
            "model_name": self.config.get("model_name"),
            "model_type": self.config.get("model_type"),
            "architecture": self.config.get("architecture"),
            "description": self.config.get("description"),
            "horizon_weeks": self.config.get("horizon_weeks"),
            "n_seeds": len(self.config.get("seeds", [])),
            "n_objectives": 2,
            "total_models": len(self.models),
            "required_input_columns": self.config.get("required_input_columns", []),
            "feature_count": len(self.feature_cols),
            "training_metadata": {
                "exported_at": self.metadata.get("exported_at"),
                "data_date_range": self.metadata.get("data_date_range"),
                "n_branches": self.metadata.get("n_branches"),
                "branches": self.metadata.get("branches"),
            },
            "accuracy": self.metadata.get("accuracy_reference", {}),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run production inference with serialized models."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to input CSV. Defaults to data/merged_new_sales.csv.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save forecast JSON. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--model-dir", "-m",
        type=str,
        default=None,
        help="Path to production_model artifacts directory.",
    )
    args = parser.parse_args()

    forecaster = ProductionForecaster(model_dir=args.model_dir)

    input_path = args.input
    if input_path is None:
        input_path = Path(__file__).resolve().parents[2] / "data" / "merged_new_sales.csv"

    result = forecaster.predict_from_csv(input_path)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Forecast saved to {output_path}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

    # Print summary.
    print(f"\n{'='*60}")
    print(f"Model: {result['model']}")
    print(f"Forecast weeks: {result['forecast_weeks'][0]} -> {result['forecast_weeks'][-1]}")
    print(f"Company total (4w): {result['company_total_4w']:,.0f} EGP")
    print(f"\n{'branch':<20s} {'4-week total (EGP)':>20s}")
    print("-" * 42)
    for b, v in sorted(
        result["per_branch"].items(), key=lambda kv: -kv[1]["total_4w"]
    ):
        print(f"{b:<20s} {v['total_4w']:>20,.0f}")
