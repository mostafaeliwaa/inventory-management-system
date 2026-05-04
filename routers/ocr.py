import os
import shutil
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from ocr1.Models.complete_ocr_pipeline import preprocess_image, extract_invoice_data_gemini
from ocr1.Models.product_matcher_system import ProductMatcher
from ocr1.Scripts.database_matching_utils import get_top3_matches

logger = logging.getLogger(__name__)

router = APIRouter()

# Store instances in router state
_router_state = {
    "product_matcher": None,
    "gemini_model": None
}

def initialize_router( product_matcher: ProductMatcher, gemini_model):
    """Call this from main.py to set dependencies"""
    _router_state["product_matcher"] = product_matcher
    _router_state["gemini_model"] = gemini_model

def get_product_matcher():
    """Dependency to get product matcher"""
    if _router_state["product_matcher"] is None:
        raise HTTPException(status_code=503, detail="Product matcher not initialized")
    return _router_state["product_matcher"]

def get_gemini_model():
    """Dependency to get Gemini model"""
    if _router_state["gemini_model"] is None:
        raise HTTPException(status_code=503, detail="Gemini model not initialized")
    return _router_state["gemini_model"]

@router.get("/health")
async def health(
    matcher: ProductMatcher = Depends(get_product_matcher),
    model = Depends(get_gemini_model)
):
    return {
        "status": "ok",
        "products_loaded": len(matcher.products) if matcher.products else 0,
        "gemini_ready": model is not None,
    }

@router.post("/process-invoice")
async def process_invoice(
    file: UploadFile = File(...),
    matcher: ProductMatcher = Depends(get_product_matcher),
    model = Depends(get_gemini_model)
):
    temp_path = f"/tmp/invoice_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        image = Image.open(temp_path)
        processed_image = preprocess_image(image)
        extracted_data = extract_invoice_data_gemini(model, processed_image)

        items = extracted_data.get("items", []) if extracted_data else []

        summary = []
        for item in items:
            query = (item.get("product_name") or "").strip()
            if not query:
                continue

            matches = get_top3_matches(matcher, query, min_score=0.0, top_k=3)
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

    except Exception as e:
        logger.error(f"Error processing invoice: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)