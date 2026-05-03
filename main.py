from asyncio.log import logger
import os
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from ocr.Scripts.database_matching_utils import load_product_database_from_supabase
from routers.chatbot import router as chatbot_router
from routers.ocr import router as ocr_router, logger
import routers.ocr as ocr_module
from ocr.Models.complete_ocr_pipeline import setup_gemini
from ocr.Models.product_matcher_system import ProductMatcher
import pandas as pd
from ocr.Scripts.database_matching_utils import load_product_database_from_supabase, get_top10_matches

# Load environment variables from .env file
load_dotenv()

FUZZY_ONLY = os.getenv("FUZZY_ONLY_MATCHING", "false").lower() == "true"

# Global variables that will be initialized in lifespan
PRODUCT_MATCHER = None
PRODUCT_DB = None
GEMINI_MODEL = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global PRODUCT_MATCHER, PRODUCT_DB, GEMINI_MODEL

    logger.info("🚀 Starting app…")

    try:
        PRODUCT_DB, _ = load_product_database_from_supabase(table_name="products")
    except Exception as exc:
        logger.error(f"❌ Failed to load product database: {exc}")
        raise RuntimeError("Cannot start without product database") from exc

    PRODUCT_MATCHER = ProductMatcher(product_database=PRODUCT_DB, threshold=0.6)

    GEMINI_MODEL = setup_gemini()
    if GEMINI_MODEL is None:
        logger.warning("⚠️ Gemini model unavailable — OCR endpoint will fail.")

    # Also set the variables in the ocr_module so the router can access them
    ocr_module.PRODUCT_MATCHER = PRODUCT_MATCHER
    ocr_module.PRODUCT_DB = PRODUCT_DB
    ocr_module.GEMINI_MODEL = GEMINI_MODEL

    logger.info(f"✅ System ready — {len(PRODUCT_DB)} products loaded.")
    yield
    logger.info("🛑 Shutting down…")

    logger.info("🚀 Starting app…")

    try:
        PRODUCT_DB, _ = load_product_database_from_supabase(table_name="products")
    except Exception as exc:
        logger.error(f"❌ Failed to load product database: {exc}")
        raise RuntimeError("Cannot start without product database") from exc

    PRODUCT_MATCHER = ProductMatcher(product_database=PRODUCT_DB, threshold=0.6)


    GEMINI_MODEL = setup_gemini()
    if GEMINI_MODEL is None:
        logger.warning("⚠️ Gemini model unavailable — OCR endpoint will fail.")

    logger.info(f"✅ System ready — {len(PRODUCT_DB)} products loaded.")
    yield
    logger.info("🛑 Shutting down…")
app = FastAPI(title="Inventory Management System", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    chatbot_router,
    prefix="/chatbot",
    tags=["Chatbot"]
)

app.include_router(
    ocr_router,
    prefix="/ocr",
    tags=["OCR"]
)