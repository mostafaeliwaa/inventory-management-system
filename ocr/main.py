import os
import shutil
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image

from Models.complete_ocr_pipeline import preprocess_image, setup_gemini, extract_invoice_data_gemini
from Models.product_matcher_system import ProductMatcher
from Scripts.database_matching_utils import load_product_database_from_supabase, get_top10_matches

logger = logging.getLogger(__name__)

# ── Globals ───────────────────────────────────────────────────────────────────

PRODUCT_MATCHER: ProductMatcher = None
PRODUCT_DB: list = None
GEMINI_MODEL = None

# ── Startup / shutdown ────────────────────────────────────────────────────────

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

    logger.info(f"✅ System ready — {len(PRODUCT_DB)} products loaded.")
    yield
    logger.info("🛑 Shutting down…")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Invoice Processing API", lifespan=lifespan)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "products_loaded": len(PRODUCT_DB) if PRODUCT_DB else 0,
        "gemini_ready": GEMINI_MODEL is not None,
    }


@app.post("/process-invoice")
async def process_invoice(file: UploadFile = File(...)):
    temp_path = f"/tmp/invoice_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        image = Image.open(temp_path)
        processed_image = preprocess_image(image)
        extracted_data = extract_invoice_data_gemini(GEMINI_MODEL, processed_image)

        items = extracted_data.get("items", []) if extracted_data else []

        summary = []
        for item in items:
            query = (item.get("product_name") or "").strip()
            if not query:
                continue

            matches = get_top10_matches(PRODUCT_MATCHER, query, min_score=0.0, top_k=10)
            summary.append({
                "query": query,
                "matched": [{"name": m["name"], "score": m["score"]} for m in matches],
                "top_score": matches[0]["score"] if matches else 0.0,
            })

        return JSONResponse(content={
            "extracted": extracted_data,
            "generated_at": datetime.utcnow().isoformat(),
            "summary": summary,
        })

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)