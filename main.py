"""Unified API server for the Inventory Management System.

Mounts all sub-services under one FastAPI app:

    /ocr           — Invoice OCR + product matching
    /chatbot       — AI inventory assistant (LangGraph)
    /forecasting   — Revenue forecasting (LightGBM production model)
    /              — Root health check

Run:
    uvicorn main:app --port 8000 --reload
"""
from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# OCR dependencies
# ---------------------------------------------------------------------------
from routers.ocr import router as ocr_router, initialize_router
from routers.chatbot import router as chatbot_router
from ocr1.Models.complete_ocr_pipeline import setup_gemini
from ocr1.Models.product_matcher_system import ProductMatcher
from ocr1.Scripts.database_matching_utils import (
    load_product_database_from_supabase,
    load_product_database_from_excel,
)

# ---------------------------------------------------------------------------
# Forecasting API (imported as a sub-application)
# ---------------------------------------------------------------------------
from forecasting.api import app as forecasting_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: initialize heavy resources once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize OCR product matcher and Gemini model at startup."""
    logger.info("Starting Inventory Management System...")

    # --- Load product database (Supabase -> CSV fallback) ---
    supabase_success = False
    product_db = None

    try:
        logger.info("Attempting to load products from Supabase...")
        product_db, _ = load_product_database_from_supabase(table_name="products")
        if product_db:
            supabase_success = True
            logger.info("Loaded %d products from Supabase.", len(product_db))
        else:
            logger.warning("Supabase returned empty product list.")
    except Exception as exc:
        logger.warning("Supabase connection failed: %s", exc)

    if not supabase_success:
        logger.info("Falling back to CSV file...")
        try:
            csv_path = "ocr1/data/supabaseData.csv"
            if os.path.exists(csv_path):
                product_db, _ = load_product_database_from_excel(
                    file_path=csv_path,
                    product_column="product_name",
                )
                logger.info("Loaded %d products from CSV fallback.", len(product_db))
            else:
                raise FileNotFoundError(f"CSV file not found at {csv_path}")
        except Exception as exc:
            logger.error("Failed to load product database: %s", exc)
            raise RuntimeError("Cannot start without product database") from exc

    if not product_db or len(product_db) == 0:
        raise RuntimeError("Product database is empty")

    # --- Initialize OCR components ---
    product_matcher = ProductMatcher(product_database=product_db, threshold=0.6)
    logger.info("Product matcher initialized with %d products.", len(product_db))

    gemini_model = setup_gemini()
    if gemini_model is None:
        logger.warning("Gemini model unavailable - OCR endpoint will fail.")
    else:
        logger.info("Gemini model ready.")

    initialize_router(product_matcher, gemini_model)

    app.state.product_matcher = product_matcher
    app.state.gemini_model = gemini_model

    # --- Check forecasting model availability ---
    model_config = (
        Path(__file__).resolve().parent
        / "forecasting" / "artifacts" / "production_model" / "model_config.json"
    )
    if model_config.exists():
        logger.info("Forecasting production model available.")
    else:
        logger.warning(
            "Forecasting production model not exported. "
            "Run `python -m forecasting.src.export_model` to enable /forecasting/forecast/predict."
        )

    logger.info("System fully ready - %d products loaded.", len(product_db))

    yield

    logger.info("Shutting down...")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Inventory Management System",
    description=(
        "Unified API for inventory management: OCR invoice processing, "
        "AI chatbot assistant, and revenue forecasting."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Mount routers and sub-applications
# ---------------------------------------------------------------------------

# OCR endpoints: /ocr/process-invoice, /ocr/health
app.include_router(ocr_router, prefix="/ocr", tags=["OCR"])

# Chatbot endpoints: /chatbot/chat, /chatbot/health
app.include_router(chatbot_router, prefix="/chatbot", tags=["Chatbot"])

# Forecasting endpoints (mounted as a sub-application):
#   /forecasting/health
#   /forecasting/forecast/weekly
#   /forecasting/forecast/weekly/{branch}
#   /forecasting/forecast/predict        (POST - production)
#   /forecasting/forecast/predict-csv    (POST - production)
#   /forecasting/forecast/model-info
#   /forecasting/metrics/weekly
#   /forecasting/forecast               (daily, pre-generated)
#   /forecasting/forecast/{branch}      (daily, pre-generated)
#   /forecasting/metrics                (daily)
app.mount("/forecasting", forecasting_app)


# ---------------------------------------------------------------------------
# Root endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["System"])
async def root():
    """System overview and available services."""
    return {
        "service": "Inventory Management System",
        "status": "running",
        "endpoints": {
            "ocr": "/ocr  (POST /ocr/process-invoice)",
            "chatbot": "/chatbot  (POST /chatbot/chat)",
            "forecasting": "/forecasting  (GET /forecasting/forecast/weekly, POST /forecasting/forecast/predict)",
        },
        "docs": "/docs",
    }


@app.get("/health", tags=["System"])
async def health():
    """Unified health check across all services."""
    forecasting_model_available = (
        Path(__file__).resolve().parent
        / "forecasting" / "artifacts" / "production_model" / "model_config.json"
    ).exists()

    return {
        "status": "ok",
        "services": {
            "ocr": {
                "status": "ok" if hasattr(app.state, "product_matcher") else "unavailable",
                "products_loaded": (
                    len(app.state.product_matcher.products)
                    if hasattr(app.state, "product_matcher") and app.state.product_matcher.products
                    else 0
                ),
                "gemini_ready": (
                    app.state.gemini_model is not None
                    if hasattr(app.state, "gemini_model")
                    else False
                ),
            },
            "chatbot": {"status": "ok"},
            "forecasting": {
                "status": "ok",
                "production_model_available": forecasting_model_available,
            },
        },
    }