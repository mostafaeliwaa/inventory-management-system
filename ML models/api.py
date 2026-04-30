"""FastAPI service exposing demand classification and dead-stock prediction.

Endpoints
---------
GET  /health                     liveness probe
GET  /metrics                    training metrics for both models
POST /predict/demand             predict from raw features
POST /predict/dead-stock         predict from raw features
GET  /predict/by-code/{code}     predict both for a known product code
POST /predict/batch              batch predict by list of codes
GET  /codes/sample               sample known product codes (for testing)

Run:
    uvicorn api:app --host 0.0.0.0 --port 8003 --reload
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import (ModelNotTrainedError, list_known_codes, predict_batch,
                         predict_by_code, predict_dead_stock_from_features,
                         predict_demand_from_features)

REPORTS_DIR = Path(__file__).parent / "reports"

app = FastAPI(
    title="Inventory ML API",
    description="Demand classification (fast/medium/slow) and dead-stock prediction.",
    version="1.0.0",
)


class ProductFeatures(BaseModel):
    total_sales: float = Field(0, ge=0)
    total_stock: float = Field(0, ge=0)
    branches_with_sales: float = Field(0, ge=0)
    branches_with_stock: float = Field(0, ge=0)
    max_branch_sales: float = Field(0, ge=0)
    max_branch_stock: float = Field(0, ge=0)
    sales_concentration: float = Field(0, ge=0, le=1)
    stock_to_sales_ratio: float = Field(0, ge=0)
    sell_through: float = Field(0, ge=0, le=1)
    cost_price: float = Field(0, ge=0)
    retail_price: float = Field(0, ge=0)
    margin_pct: float = 0.0


class BatchRequest(BaseModel):
    codes: list[str]


class DemandResponse(BaseModel):
    label: str
    probabilities: dict


class DeadStockResponse(BaseModel):
    is_dead_stock: bool
    probability: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    out: dict = {}
    for key, fname in [
        ("demand", "demand_classifier_report.json"),
        ("dead_stock", "dead_stock_classifier_report.json"),
        ("robustness", "robustness_report.json"),
    ]:
        path = REPORTS_DIR / fname
        out[key] = json.loads(path.read_text()) if path.exists() else None
    return out


@app.post("/predict/demand", response_model=DemandResponse)
def predict_demand(payload: ProductFeatures):
    try:
        return predict_demand_from_features(payload.model_dump())
    except ModelNotTrainedError as e:
        raise HTTPException(503, str(e))


@app.post("/predict/dead-stock", response_model=DeadStockResponse)
def predict_dead_stock(payload: ProductFeatures):
    try:
        return predict_dead_stock_from_features(payload.model_dump())
    except ModelNotTrainedError as e:
        raise HTTPException(503, str(e))


@app.get("/predict/by-code/{code}")
def predict_code(code: str):
    try:
        result = predict_by_code(code)
    except ModelNotTrainedError as e:
        raise HTTPException(503, str(e))
    if not result.get("found"):
        raise HTTPException(404, f"Product code {code} not found")
    return result


@app.post("/predict/batch")
def predict_batch_endpoint(req: BatchRequest):
    if not req.codes:
        raise HTTPException(400, "codes list cannot be empty")
    if len(req.codes) > 500:
        raise HTTPException(400, "max 500 codes per request")
    try:
        return {"results": predict_batch(req.codes)}
    except ModelNotTrainedError as e:
        raise HTTPException(503, str(e))


@app.get("/codes/sample")
def sample_codes(limit: int = 20):
    try:
        return {"codes": list_known_codes(limit=limit)}
    except ModelNotTrainedError as e:
        raise HTTPException(503, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
