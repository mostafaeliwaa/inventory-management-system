"""FastAPI service for daily branch-revenue forecasts.

Run: uvicorn forecasting.api:app --port 8004 --reload
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
FORECAST_FILE = REPORTS_DIR / "next_7d.json"
CV_FILE = REPORTS_DIR / "results.json"
WEEKLY_FORECAST_FILE = REPORTS_DIR / "next_4w.json"
WEEKLY_CV_FILE = REPORTS_DIR / "results_weekly.json"

app = FastAPI(
    title="Branch Revenue Forecasting API",
    description="7-day daily revenue forecast per branch. Built from merged_new_sales transactions.",
    version="0.1.0",
)


def _load(path: Path) -> dict:
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"{path.name} not found. Run forecasting.src.forecast first.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "daily_forecast_available": FORECAST_FILE.exists(),
        "daily_cv_available": CV_FILE.exists(),
        "weekly_forecast_available": WEEKLY_FORECAST_FILE.exists(),
        "weekly_cv_available": WEEKLY_CV_FILE.exists(),
    }


def _annotate_weekly(data: dict) -> dict:
    """Promote ensemble3 to the headline `forecast` field (3-model is production)."""
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


@app.get("/forecast/weekly")
def all_weekly() -> dict:
    return _annotate_weekly(_load(WEEKLY_FORECAST_FILE))


@app.get("/forecast/weekly/{branch}")
def branch_weekly(branch: str) -> dict:
    data = _annotate_weekly(_load(WEEKLY_FORECAST_FILE))
    key_map = {k.lower(): k for k in data["per_branch"]}
    real_key = key_map.get(branch.lower())
    if real_key is None:
        raise HTTPException(status_code=404, detail=f"branch '{branch}' not found. Available: {sorted(data['per_branch'])}")
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
    return _load(WEEKLY_CV_FILE)


# ------ Daily endpoints (declared LAST so weekly path-prefix wins routing) ------
@app.get("/forecast")
def all_forecasts() -> dict:
    return _load(FORECAST_FILE)


@app.get("/forecast/{branch}")
def branch_forecast(branch: str) -> dict:
    data = _load(FORECAST_FILE)
    key_map = {k.lower(): k for k in data["per_branch"]}
    real_key = key_map.get(branch.lower())
    if real_key is None:
        raise HTTPException(status_code=404, detail=f"branch '{branch}' not found. Available: {sorted(data['per_branch'])}")
    return {
        "branch": real_key,
        "generated_at": data["generated_at"],
        "history_last_date": data["history_last_date"],
        "horizon_days": data["horizon_days"],
        **data["per_branch"][real_key],
    }


@app.get("/metrics")
def metrics() -> dict:
    return _load(CV_FILE)
