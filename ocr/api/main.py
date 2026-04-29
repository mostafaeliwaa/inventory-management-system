import logging
import sys
import tempfile
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
OCR_DIR = REPO_ROOT / "ocr"
sys.path.insert(0, str(OCR_DIR))
sys.path.insert(0, str(REPO_ROOT))

from complete_ocr_pipeline import setup_gemini
from product_matcher_system import ProductMatcher
from ocr.process_uploaded_image import process_image

# إضافة دالة تطبيع بسيطة للمنتجات
def normalize_product_name(text: str) -> str:
    """Simple normalization for product names."""
    if not text:
        return ""
    # إزالة المسافات الزائدة وتحويل إلى lowercase
    return text.strip().lower()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("invoice_api")

# Constants
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
PRODUCT_DB_PATH = REPO_ROOT / "cleaning" / "output" / "master_dataframe.csv"
RESULTS_DIR = REPO_ROOT / "results"

_state: dict[str, Any] = {
    "gemini_model": None,
    "product_matcher": None,
    "ready": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup ----
    log.info("Starting up Invoice OCR API...")

    # Gemini
    try:
        _state["gemini_model"] = setup_gemini()
        log.info("Gemini model loaded successfully.")
    except Exception as exc:
        log.error("Failed to load Gemini model: %s", exc)

    # Product database + matcher
    try:
        if PRODUCT_DB_PATH.exists():
            df = pd.read_csv(PRODUCT_DB_PATH)
            
            product_names = []
            
            # Find the correct column for product names
            possible_name_cols = ['اسم الصنف', 'product_name', 'name', 'Product', 'الصنف', 'اسم المنتج']
            name_col = None
            
            for col in possible_name_cols:
                if col in df.columns:
                    name_col = col
                    break
            
            if name_col is None and len(df.columns) > 0:
                # Use first column as fallback, but skip if it's purely numeric IDs
                first_col = df.columns[0]
                sample_vals = df[first_col].dropna().head(10).astype(str)
                # If most values are not just numbers, use this column
                if not sample_vals.str.match(r'^\d+$').all():
                    name_col = first_col
                elif len(df.columns) > 1:
                    name_col = df.columns[1]  # Try second column
            
            if name_col:
                product_names = df[name_col].dropna().astype(str).str.strip().tolist()
                # Filter out empty strings and 'nan'
                product_names = [p for p in product_names if p and p.lower() != 'nan']
            else:
                # Fallback: collect all non-numeric values from all columns
                for col in df.columns:
                    for val in df[col].dropna():
                        str_val = str(val).strip()
                        if str_val and not str_val.isdigit() and str_val.lower() != 'nan':
                            product_names.append(str_val)
            
            if not product_names:
                log.warning("No product names found, using IDs as fallback")
                # Last resort: use first column even if numeric
                first_col = df.columns[0]
                product_names = df[first_col].dropna().astype(str).str.strip().tolist()
                product_names = [p for p in product_names if p and p.lower() != 'nan']
            
            _state["product_matcher"] = ProductMatcher(product_database=product_names)
            _state["product_matcher"].compute_embeddings()
            log.info("ProductMatcher ready with %d products.", len(product_names))
            log.info("Sample products: %s", product_names[:5])
        else:
            log.warning("Product database not found at %s", PRODUCT_DB_PATH)
    except Exception as exc:
        log.error("Failed to initialise ProductMatcher: %s", exc)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _state["ready"] = _state["gemini_model"] is not None and _state["product_matcher"] is not None
    log.info("API startup complete. Ready=%s", _state["ready"])

    yield
    
    # ---- shutdown ----
    log.info("Shutting down Invoice OCR API.")


# FastAPI app
app = FastAPI(
    title="Invoice OCR & Product Matching API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic schemas
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
@app.get("/health", tags=["meta"])
def health() -> dict:
    """Return live status of the API and its dependencies."""
    return {
        "status": "ok" if _state["ready"] else "degraded",
        "gemini": _state["gemini_model"] is not None,
        "matcher": _state["product_matcher"] is not None,
    }


@app.get("/stats", tags=["meta"])
def stats() -> dict:
    """Return ProductMatcher statistics."""
    if _state["product_matcher"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ProductMatcher not initialised.",
        )
    return _state["product_matcher"].get_stats()


@app.get("/debug/startup", tags=["meta"])
def debug_startup() -> dict:
    """Return simple startup diagnostics for the frontend debug panel."""
    return {
        "product_db_path": str(PRODUCT_DB_PATH),
        "product_db_exists": PRODUCT_DB_PATH.exists(),
        "gemini_loaded": _state.get("gemini_model") is not None,
        "matcher_loaded": _state.get("product_matcher") is not None,
        "ready": _state.get("ready", False),
    }


@app.post("/process-invoice", tags=["ocr"])
async def process_invoice(file: UploadFile = File(...)) -> JSONResponse:
    """Process uploaded invoice image and return OCR results with product matches."""
    _validate_extension(file.filename or "")
    _assert_services()

    suffix = Path(file.filename or "upload").suffix.lower()
    tmp_path = Path(tempfile.mktemp(suffix=suffix, dir=str(RESULTS_DIR)))

    try:
        # Read and save uploaded file
        contents = await file.read()
        tmp_path.write_bytes(contents)
        log.info("Saved upload to %s", tmp_path)

        # Process image with OCR pipeline
        result = process_image(
            tmp_path,
            matcher=_state["product_matcher"],
            model=_state["gemini_model"]
        )

        # Get paths from result
        sample_path = result.get("sample")
        matches_path = result.get("matches")
        output_dir = result.get("output_dir")

        # Load sample data (OCR results)
        sample = {}
        if sample_path and Path(sample_path).exists():
            try:
                with open(sample_path, "r", encoding="utf-8") as f:
                    sample = json.load(f)
            except Exception as e:
                log.error(f"Error reading sample file: {e}")

        # Load matches data
        matches = {}
        if matches_path and Path(matches_path).exists():
            try:
                with open(matches_path, "r", encoding="utf-8") as f:
                    matches = json.load(f)
            except Exception as e:
                log.error(f"Error reading matches file: {e}")

        # Extract invoice data
        invoice_data = sample.get("extracted", {})
        items_from_invoice = invoice_data.get("items", [])
        
        # 🔧 إنشاء قاموس للبحث عن المنتجات بالاسم (لربط الكميات والأسعار)
        # تطبيع الأسماء لتسهيل المطابقة
        items_dict = {}
        for item in items_from_invoice:
            product_name = item.get("product_name", "")
            # تخزين البيانات حسب الاسم الأصلي
            if product_name:
                items_dict[product_name] = {
                    "quantity": item.get("quantity"),
                    "unit_price": item.get("unit_price"),
                    "item_total": item.get("item_total")
                }
            # أيضاً تخزين نسخة مطبعة للبحث
            normalized_name = normalize_product_name(product_name) if product_name else ""
            if normalized_name:
                items_dict[normalized_name] = {
                    "quantity": item.get("quantity"),
                    "unit_price": item.get("unit_price"),
                    "item_total": item.get("item_total"),
                    "original_name": product_name
                }
        
            # Build products list from matches
            products_list = []
            matches_summary = matches.get("summary", []) if isinstance(matches, dict) else []

            for match_item in matches_summary:
                query = match_item.get("query", "")
                matched_list = match_item.get("matched", [])  # Now contains ALL matches
    
                # Best match (first one in the list)
                best_match = matched_list[0] if matched_list else None
    
                # Search for quantity and unit price from invoice data
                quantity = None
                unit_price = None
                item_total = None
    
                # Search by exact match
                if query in items_dict:
                    quantity = items_dict[query].get("quantity")
                    unit_price = items_dict[query].get("unit_price")
                    item_total = items_dict[query].get("item_total")
                else:
                    # Search by normalized name
                    normalized_query = normalize_product_name(query) if query else ""
                    if normalized_query in items_dict:
                        quantity = items_dict[normalized_query].get("quantity")
                        unit_price = items_dict[normalized_query].get("unit_price")
                        item_total = items_dict[normalized_query].get("item_total")
                    else:
                        # Partial match search
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
                    "all_matches": matched_list,  # ALL matches now included here
                    "total_matches": match_item.get("total_matches", len(matched_list)),
                    "top_matches": matched_list[:10] if len(matched_list) > 10 else matched_list,  # Keep top 10 for UI
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "item_total": item_total,
                })
                
        # Get processed files list
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
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.post("/confirm-items", tags=["matching"])
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

    # Load master CSV
    try:
        if PRODUCT_DB_PATH.exists():
            master_df = pd.read_csv(PRODUCT_DB_PATH)
        else:
            master_df = pd.DataFrame()
    except Exception as e:
        log.exception("Failed to read product DB CSV, continuing with empty DataFrame: %s", e)
        master_df = pd.DataFrame()

    for corr in body.corrections:
        try:
            # Add correction to matcher
            _state["product_matcher"].add_correction(corr.wrong_text, corr.correct_text)
            applied.append({"wrong_text": corr.wrong_text, "correct_text": corr.correct_text})

            # Append to master CSV if product not present
            try:
                col_candidates = [c for c in master_df.columns if "اسم" in c or "صنف" in c or "name" in c.lower()]
                target_col = col_candidates[0] if col_candidates else None
                
                if target_col is None:
                    target_col = "اسم الصنف"
                    if target_col not in master_df.columns:
                        master_df[target_col] = []

                # Check if product already exists
                exists = False
                if not master_df.empty:
                    exists = master_df[target_col].astype(str).str.strip().eq(str(corr.correct_text).strip()).any()

                # Add new product if it doesn't exist
                if not exists:
                    if master_df.empty and len(master_df.columns) == 0:
                        master_df = pd.DataFrame([{target_col: corr.correct_text}])
                    else:
                        new_row = {c: "" for c in master_df.columns}
                        new_row[target_col] = corr.correct_text
                        master_df = pd.concat([master_df, pd.DataFrame([new_row])], ignore_index=True)
            except Exception as e:
                log.warning("Failed to persist correction to CSV: %s", e)
                
        except Exception as exc:
            log.error("Failed to add correction %s → %s: %s", corr.wrong_text, corr.correct_text, exc)
            failed.append({"wrong_text": corr.wrong_text, "error": str(exc)})

    # Save updated master CSV
    try:
        if not master_df.empty:
            master_df.to_csv(PRODUCT_DB_PATH, index=False)
            log.info(f"Saved {len(applied)} corrections to CSV")
    except Exception as e:
        log.warning("Failed to save updated master product CSV: %s", e)

    return {
        "status": "ok",
        "applied": len(applied),
        "failed": len(failed),
        "details": {"applied": applied, "failed": failed},
    }


# Entrypoint
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)