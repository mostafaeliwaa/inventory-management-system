"""FastAPI service for branch-revenue forecasts.

Endpoints:
    GET  /health                 – service health check
    GET  /forecast/weekly        – pre-generated 4-week forecast (all branches)
    GET  /forecast/weekly/{b}    – pre-generated 4-week forecast (one branch)
    GET  /metrics/weekly         – CV results for the weekly models
    GET  /forecast               – pre-generated 7-day daily forecast (all branches)
    GET  /forecast/{branch}      – pre-generated 7-day daily forecast (one branch)
    GET  /metrics                – CV results for the daily models
    POST /forecast/predict       – PRODUCTION: accept raw data JSON, return 4-week forecast
    POST /forecast/predict-csv   – PRODUCTION: accept CSV upload, return 4-week forecast
    GET  /forecast/model-info    – PRODUCTION: model metadata and accuracy info

Run: uvicorn forecasting.api:app --port 8004 --reload
"""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
FORECAST_FILE = REPORTS_DIR / "next_7d.json"
CV_FILE = REPORTS_DIR / "results.json"
WEEKLY_FORECAST_FILE = REPORTS_DIR / "next_4w.json"
WEEKLY_CV_FILE = REPORTS_DIR / "results_weekly.json"

app = FastAPI(
    title="Branch Revenue Forecasting API",
    description=(
        "Weekly and daily revenue forecasts per branch. "
        "Includes production endpoints for real-time prediction from raw POS data."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Lazy-loaded production forecaster (avoids loading 56 models at import time).
# ---------------------------------------------------------------------------
_forecaster = None


def _get_forecaster():
    """Lazy-load the ProductionForecaster on first use."""
    global _forecaster
    if _forecaster is None:
        try:
            from forecasting.src.inference import ProductionForecaster
            _forecaster = ProductionForecaster()
            logger.info("Production forecaster loaded successfully.")
        except FileNotFoundError as exc:
            logger.warning("Production model not available: %s", exc)
            raise HTTPException(
                status_code=503,
                detail=(
                    "Production model not exported yet. "
                    "Run `python -m forecasting.src.export_model` first."
                ),
            ) from exc
    return _forecaster


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> dict:
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"{path.name} not found. Run the appropriate forecasting script first.",
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _annotate_weekly(data: dict) -> dict:
    """Promote ensemble3 to the headline ``forecast`` field (3-model is production)."""
    out = {k: v for k, v in data.items() if k != "per_branch"}
    out["primary_model"] = "ensemble3"
    out["primary_model_description"] = (
        "Mean of Prophet, LightGBM and naive_mean4. Production model. WAPE 0.229 in CV."
    )
    new_per_branch = {}
    for b, v in data["per_branch"].items():
        v = dict(v)
        v["forecast"] = v.get("ensemble3", v.get("ensemble"))
        v["forecast_total_4w"] = v.get("total_ensemble3_4w", v.get("total_ensemble_4w"))
        new_per_branch[b] = v
    out["per_branch"] = new_per_branch
    out["company_total_forecast_4w"] = data.get(
        "company_total_ensemble3_4w", data.get("company_total_ensemble_4w")
    )
    return out


# ---------------------------------------------------------------------------
# Pydantic models for the production prediction endpoint.
# ---------------------------------------------------------------------------

class SalesRecord(BaseModel):
    """Single raw sales transaction from the POS system.

    Accepts either ``branch_code`` or ``branch_id`` — both map to the same thing.
    """
    posting_date: str
    branch_code: Optional[int] = None
    branch_id: Optional[int] = None
    branch_name_en: str
    total_value: float
    cash_paid: float = 0.0


class PredictRequest(BaseModel):
    """Request body for the /forecast/predict endpoint."""
    records: list[SalesRecord]


# ---------------------------------------------------------------------------
# Existing endpoints (pre-generated forecasts from JSON files)
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Service health check."""
    production_available = (
        Path(__file__).resolve().parent / "artifacts" / "production_model" / "model_config.json"
    ).exists()
    return {
        "status": "ok",
        "daily_forecast_available": FORECAST_FILE.exists(),
        "daily_cv_available": CV_FILE.exists(),
        "weekly_forecast_available": WEEKLY_FORECAST_FILE.exists(),
        "weekly_cv_available": WEEKLY_CV_FILE.exists(),
        "production_model_available": production_available,
    }


@app.get("/forecast/weekly")
def all_weekly() -> dict:
    """Pre-generated 4-week ensemble forecast for all branches."""
    return _annotate_weekly(_load(WEEKLY_FORECAST_FILE))


@app.get("/forecast/weekly/{branch}")
def branch_weekly(branch: str) -> dict:
    """Pre-generated 4-week ensemble forecast for a single branch."""
    data = _annotate_weekly(_load(WEEKLY_FORECAST_FILE))
    key_map = {k.lower(): k for k in data["per_branch"]}
    real_key = key_map.get(branch.lower())
    if real_key is None:
        raise HTTPException(
            status_code=404,
            detail=f"branch '{branch}' not found. Available: {sorted(data['per_branch'])}",
        )
    return {
        "branch": real_key,
        "generated_at": data["generated_at"],
        "history_last_week": data["history_last_week"],
        "horizon_weeks": data["horizon_weeks"],
        "primary_model": data["primary_model"],
        **data["per_branch"][real_key],
    }


@app.get("/metrics/weekly")
def metrics_weekly() -> dict:
    """Cross-validation results for the weekly models."""
    return _load(WEEKLY_CV_FILE)


# ------ Daily endpoints (declared LAST so weekly path-prefix wins routing) ------

@app.get("/forecast")
def all_forecasts() -> dict:
    """Pre-generated 7-day daily forecast for all branches."""
    return _load(FORECAST_FILE)


@app.get("/forecast/{branch}")
def branch_forecast(branch: str) -> dict:
    """Pre-generated 7-day daily forecast for a single branch."""
    data = _load(FORECAST_FILE)
    key_map = {k.lower(): k for k in data["per_branch"]}
    real_key = key_map.get(branch.lower())
    if real_key is None:
        raise HTTPException(
            status_code=404,
            detail=f"branch '{branch}' not found. Available: {sorted(data['per_branch'])}",
        )
    return {
        "branch": real_key,
        "generated_at": data["generated_at"],
        "history_last_date": data["history_last_date"],
        "horizon_days": data["horizon_days"],
        **data["per_branch"][real_key],
    }


@app.get("/metrics")
def metrics() -> dict:
    """Cross-validation results for the daily models."""
    return _load(CV_FILE)


# ---------------------------------------------------------------------------
# PRODUCTION ENDPOINTS — Real-time prediction from raw data
# ---------------------------------------------------------------------------

@app.post("/forecast/predict")
def predict_from_json(request: PredictRequest) -> dict:
    """Accept raw POS data as JSON and return a 4-week revenue forecast.

    The input should be a list of sales records with at minimum:
    ``posting_date``, ``branch_code``, ``branch_name_en``, ``total_value``, ``cash_paid``.

    Requires the production model to be exported first via
    ``python -m forecasting.src.export_model``.
    """
    forecaster = _get_forecaster()

    records = [r.model_dump() for r in request.records]
    if not records:
        raise HTTPException(status_code=400, detail="No records provided.")

    raw_df = pd.DataFrame(records)

    # Normalize: branch_id -> branch_code (DB uses branch_id).
    if "branch_id" in raw_df.columns:
        raw_df["branch_code"] = raw_df["branch_code"].fillna(raw_df["branch_id"])
        raw_df = raw_df.drop(columns=["branch_id"])
    if raw_df["branch_code"].isna().any():
        raise HTTPException(
            status_code=400,
            detail="Each record must have either 'branch_code' or 'branch_id'.",
        )

    raw_df["posting_date"] = pd.to_datetime(raw_df["posting_date"])

    try:
        result = forecaster.predict(raw_df)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return result


@app.post("/forecast/predict-csv")
async def predict_from_csv(file: UploadFile = File(...)) -> dict:
    """Accept a CSV file upload and return a 4-week revenue forecast.

    The CSV must contain columns:
    ``posting_date``, ``branch_code``, ``branch_name_en``, ``total_value``, ``cash_paid``.

    Requires the production model to be exported first via
    ``python -m forecasting.src.export_model``.
    """
    forecaster = _get_forecaster()

    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv file.")

    content = await file.read()
    try:
        raw_df = pd.read_csv(io.BytesIO(content), parse_dates=["posting_date"])
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to parse CSV: {exc}"
        ) from exc

    try:
        result = forecaster.predict(raw_df)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return result


@app.get("/forecast/model-info")
def model_info() -> dict:
    """Return production model metadata, feature list, and accuracy metrics."""
    forecaster = _get_forecaster()
    return forecaster.get_model_info()
