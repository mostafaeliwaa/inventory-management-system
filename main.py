import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.ocr import router as ocr_router, initialize_router
from routers.chatbot import router as chatbot_router
from ocr1.Models.complete_ocr_pipeline import setup_gemini
from ocr1.Models.product_matcher_system import ProductMatcher
from ocr1.Scripts.database_matching_utils import (
    load_product_database_from_supabase,
    load_product_database_from_excel
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting app…")

    supabase_success = False
    product_db = None
    
    try:
        logger.info("📡 Attempting to load products from Supabase...")
        product_db, _ = load_product_database_from_supabase(table_name="products")
        supabase_success = True
        logger.info(f"✅ Loaded {len(product_db)} products from Supabase")
    except Exception as exc:
        logger.warning(f"⚠️ Supabase connection failed: {exc}")
        logger.info("🔄 Falling back to CSV file...")

    if not supabase_success:
        try:
            csv_path = "ocr1/data/supabaseData.csv"
            if os.path.exists(csv_path):
                product_db, _ = load_product_database_from_excel(
                    file_path=csv_path,
                    product_column="product_name"
                )
                logger.info(f"✅ Loaded {len(product_db)} products from CSV fallback")
            else:
                raise FileNotFoundError(f"CSV file not found at {csv_path}")
        except Exception as exc:
            logger.error(f"❌ Failed to load product database: {exc}")
            raise RuntimeError("Cannot start without product database") from exc

    if not product_db or len(product_db) == 0:
        raise RuntimeError("Product database is empty")

    product_matcher = ProductMatcher(product_database=product_db, threshold=0.6)
    logger.info(f"✅ Product matcher initialized with {len(product_db)} products")

    gemini_model = setup_gemini()
    if gemini_model is None:
        logger.warning("⚠️ Gemini model unavailable — OCR endpoint will fail.")
    else:
        logger.info("✅ Gemini model ready")

    initialize_router(product_matcher, gemini_model)

    app.state.product_matcher = product_matcher
    app.state.gemini_model = gemini_model

    logger.info(f"✅ System fully ready — {len(product_db)} products loaded.")
    
    yield
    
    logger.info("🛑 Shutting down…")

app = FastAPI(title="Inventory Management System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ocr_router, prefix="/ocr", tags=["OCR"])
app.include_router(chatbot_router, prefix="/chatbot", tags=["Chatbot"])

@app.get("/")
async def root():
    return {"message": "Inventory Management System API", "status": "running"}