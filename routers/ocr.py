import logging
import os
import sys
import tempfile
import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel


REPO_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ocr"))


from ocr.Models.complete_ocr_pipeline import setup_gemini
from ocr.Models.product_matcher_system import ProductMatcher
from ocr.Scripts.process_uploaded_image import process_image
from ocr.Scripts.supabase_client import get_supabase_client


def normalize_product_name(text: str) -> str:
    """Simple normalization for product names."""
    if not text:
        return ""

    return text.strip().lower()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("invoice_api")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
PRODUCT_DB_PATH = REPO_ROOT / "cleaning" / "output" / "master_dataframe.csv"
# Use the system tempdir on hosted environments (Render, etc.) where the
# repo path is read-only. Fall back to <repo>/results locally if explicitly set.
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", tempfile.gettempdir())) / "ims_results"

_state: dict[str, Any] = {
    "gemini_model": None,
    "product_matcher": None,
    "ready": False,
}


router = APIRouter()

class CorrectionItem(BaseModel):
    wrong_text: str
    correct_text: str


class ConfirmItemsRequest(BaseModel):
    corrections: list[CorrectionItem]


# Helper functions
def _assert_services() -> None:
    """Raise 503 if critical services are not available."""
    if _state["gemini_model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini model is not available.",
        )
    if _state["product_matcher"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ProductMatcher is not available.",
        )


def _validate_extension(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )


# Endpoints
@router.get("/health", tags=["OCR"])
def health() -> dict:
    """Return live status of the API and its dependencies."""
    return {
        "status": "ok" if _state["ready"] else "degraded",
        "gemini": _state["gemini_model"] is not None,
        "matcher": _state["product_matcher"] is not None,
    }


@router.get("/stats", tags=["OCR"])
def stats() -> dict:
    """Return ProductMatcher statistics."""
    if _state["product_matcher"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ProductMatcher not initialised.",
        )
    return _state["product_matcher"].get_stats()


@router.get("/debug/startup", tags=["OCR"])
def debug_startup() -> dict:
    """Return simple startup diagnostics for the frontend debug panel."""
    return {
        "product_db_path": str(PRODUCT_DB_PATH),
        "product_db_exists": PRODUCT_DB_PATH.exists(),
        "gemini_loaded": _state.get("gemini_model") is not None,
        "matcher_loaded": _state.get("product_matcher") is not None,
        "ready": _state.get("ready", False),
    }


@router.post("/process-invoice", tags=["OCR"])
async def process_invoice(file: UploadFile = File(...)) -> JSONResponse:
    """Process uploaded invoice image and return OCR results with product matches."""
    _validate_extension(file.filename or "")
    _assert_services()

    suffix = Path(file.filename or "upload").suffix.lower()
    tmp_path = Path(tempfile.mktemp(suffix=suffix, dir=str(RESULTS_DIR)))

    try:
        contents = await file.read()
        tmp_path.write_bytes(contents)
        log.info("Saved upload to %s", tmp_path)

        result = process_image(
            tmp_path,
            matcher=_state["product_matcher"],
            model=_state["gemini_model"]
        )

        sample_path = result.get("sample")
        matches_path = result.get("matches")
        output_dir = result.get("output_dir")

        sample = {}
        if sample_path and Path(sample_path).exists():
            try:
                with open(sample_path, "r", encoding="utf-8") as f:
                    sample = json.load(f)
            except Exception as e:
                log.error(f"Error reading sample file: {e}")

        matches = {}
        if matches_path and Path(matches_path).exists():
            try:
                with open(matches_path, "r", encoding="utf-8") as f:
                    matches = json.load(f)
            except Exception as e:
                log.error(f"Error reading matches file: {e}")

        invoice_data = sample.get("extracted", {})
        items_from_invoice = invoice_data.get("items", [])
        
        items_dict = {}
        for item in items_from_invoice:
            product_name = item.get("product_name", "")
            if product_name:
                items_dict[product_name] = {
                    "quantity": item.get("quantity"),
                    "unit_price": item.get("unit_price"),
                    "item_total": item.get("item_total")
                }
            normalized_name = normalize_product_name(product_name) if product_name else ""
            if normalized_name:
                items_dict[normalized_name] = {
                    "quantity": item.get("quantity"),
                    "unit_price": item.get("unit_price"),
                    "item_total": item.get("item_total"),
                    "original_name": product_name
                }
        
        products_list = []
        matches_summary = matches.get("summary", []) if isinstance(matches, dict) else []
        
        for match_item in matches_summary:
            query = match_item.get("query", "")
            matched_list = match_item.get("matched", [])
            
            best_match = matched_list[0] if matched_list else None
            quantity = None
            unit_price = None
            item_total = None
            
            if query in items_dict:
                quantity = items_dict[query].get("quantity")
                unit_price = items_dict[query].get("unit_price")
                item_total = items_dict[query].get("item_total")
            else:
                normalized_query = normalize_product_name(query) if query else ""
                if normalized_query in items_dict:
                    quantity = items_dict[normalized_query].get("quantity")
                    unit_price = items_dict[normalized_query].get("unit_price")
                    item_total = items_dict[normalized_query].get("item_total")
                else:
                    for prod_name, details in items_dict.items():
                        if query and prod_name and (query in prod_name or prod_name in query):
                            quantity = details.get("quantity")
                            unit_price = details.get("unit_price")
                            item_total = details.get("item_total")
                            break
            
            products_list.append({
                "name": query,
                "product_name": query,
                "match": {
                    "matched_product": best_match.get("name") if best_match else None,
                    "score": best_match.get("score") if best_match else None,
                    "similarity": best_match.get("score") if best_match else None,
                },
                "all_matches": matched_list,
                "total_matches": match_item.get("total_matches", len(matched_list)),
                "top_matches": matched_list[:10] if len(matched_list) > 10 else matched_list,
                "quantity": quantity,
                "unit_price": unit_price,
                "item_total": item_total,
            })
        
        if not products_list and items_from_invoice:
            for item in items_from_invoice:
                products_list.append({
                    "name": item.get("product_name", ""),
                    "product_name": item.get("product_name", ""),
                    "quantity": item.get("quantity"),
                    "unit_price": item.get("unit_price"),
                    "item_total": item.get("item_total"),
                    "match": {},
                    "all_matches": [],
                    "top_matches": [],
                })

        processed_files = []
        if output_dir and Path(output_dir).exists():
            processed_files = [p.name for p in Path(output_dir).iterdir()]

        return JSONResponse(
            content={
                "filename": file.filename,
                "invoice": invoice_data,
                "products": products_list,
                "matches_summary": matches,
                "processed_files": processed_files
            }
        )

    except Exception as e:
        log.error(f"Error processing invoice: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing invoice: {str(e)}"
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@router.post("/confirm-items", tags=["OCR"])
def confirm_items(body: ConfirmItemsRequest) -> dict:
    """
    Accept user corrections and teach the ProductMatcher.
    """
    if _state["product_matcher"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ProductMatcher not initialised.",
        )

    applied: list[dict] = []
    failed: list[dict] = []

    for corr in body.corrections:
        try:
            _state["product_matcher"].add_correction(corr.wrong_text, corr.correct_text)
            applied.append({"wrong_text": corr.wrong_text, "correct_text": corr.correct_text})
        except Exception as exc:
            log.error("Failed to add correction %s → %s: %s", corr.wrong_text, corr.correct_text, exc)
            failed.append({"wrong_text": corr.wrong_text, "error": str(exc)})

    return {
        "status": "ok",
        "applied": len(applied),
        "failed": len(failed),
        "details": {"applied": applied, "failed": failed},
    }