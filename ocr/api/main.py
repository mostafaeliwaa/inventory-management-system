"""
Invoice OCR & Product Matching API
FastAPI backend — imports from existing pipeline modules, does not rewrite them.
"""

import logging
import os
import sys
import tempfile
import uuid
import subprocess
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

# ---------------------------------------------------------------------------
# Path setup — ensure sibling packages are importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
OCR_DIR = REPO_ROOT / "ocr"
sys.path.insert(0, str(OCR_DIR))
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Imports from existing modules (DO NOT rewrite these)
# ---------------------------------------------------------------------------
from complete_ocr_pipeline import (  # noqa: E402
    extract_invoice_data_gemini,
    load_and_preprocess_image,
    preprocess_full_pipeline,
    setup_gemini,
)
from product_matcher_system import ProductMatcher  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("invoice_api")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
PRODUCT_DB_PATH = REPO_ROOT / "cleaning" / "output" / "master_dataframe.csv"
RESULTS_DIR = REPO_ROOT / "results"

# ---------------------------------------------------------------------------
# Application state (populated on startup)
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {
    "gemini_model": None,
    "product_matcher": None,
    "ready": False,
}

# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------


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
            # Ensure values are strings and strip whitespace; filter out empty/nan
            try:
                first_col = df.columns[0]
                raw_vals = df[first_col].dropna().astype(str).str.strip()
                product_list = [v for v in raw_vals.tolist() if v and v.lower() != "nan"]
            except Exception:
                # Fallback: coerce entire frame to strings and take unique non-empty values
                product_list = [str(x).strip() for x in df.values.flatten() if pd.notna(x) and str(x).strip()]

            _state["product_matcher"] = ProductMatcher(product_database=product_list)
            _state["product_matcher"].compute_embeddings()
            log.info("ProductMatcher ready with %d products.", len(product_list))
        else:
            log.warning("Product database not found at %s", PRODUCT_DB_PATH)
    except Exception as exc:
        log.error("Failed to initialise ProductMatcher: %s", exc)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _state["ready"] = _state["gemini_model"] is not None and _state["product_matcher"] is not None
    log.info("API startup complete. Ready=%s", _state["ready"])

    yield  # app runs here

    # ---- shutdown ----
    log.info("Shutting down Invoice OCR API.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class CorrectionItem(BaseModel):
    wrong_text: str
    correct_text: str


class ConfirmItemsRequest(BaseModel):
    corrections: list[CorrectionItem]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


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
    """
    Upload an invoice image → OCR extraction → product matching.
    Returns combined structured results.
    """
    _validate_extension(file.filename or "")
    _assert_services()

    suffix = Path(file.filename or "upload").suffix.lower()
    tmp_path = Path(tempfile.mktemp(suffix=suffix, dir=str(RESULTS_DIR)))

    try:

        # Save upload to temp file
        contents = await file.read()
        tmp_path.write_bytes(contents)
        log.info("Saved upload to %s (%d bytes)", tmp_path, len(contents))

        # Generate a result id early so any produced files can be collected there
        result_id = str(uuid.uuid4())

        # First try running the existing helper script which performs preprocess, extraction, and fallback matching
        try:
            proc = subprocess.run([sys.executable, str(REPO_ROOT / "ocr" / "process_uploaded_image.py"), str(tmp_path)], cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
            log.info("process_uploaded_image.py stdout: %s", proc.stdout.strip())
            if proc.returncode != 0:
                log.warning("process_uploaded_image.py returned %s: %s", proc.returncode, proc.stderr.strip())
        except Exception as e:
            log.exception("Failed to run process_uploaded_image.py: %s", e)

        # If the helper created its output folder, ingest those files and return them
        ext_folder = REPO_ROOT / "ocr" / "results" / (tmp_path.stem + "_processed")
        result_folder = REPO_ROOT / "ocr" / "results" / result_id
        result_folder.mkdir(parents=True, exist_ok=True)

        if ext_folder.exists():
            try:
                # copy all produced files into our per-result folder
                import shutil

                for p in ext_folder.iterdir():
                    if p.is_file():
                        shutil.copy2(p, result_folder / p.name)
                # load sample and matches if present
                sample_path = result_folder / "sample_invoice.json"
                matches_path = result_folder / "matches_summary_fallback.json"
                sample = None
                matches = None
                if sample_path.exists():
                    with open(sample_path, "r", encoding="utf-8") as sf:
                        sample = json.load(sf)
                if matches_path.exists():
                    with open(matches_path, "r", encoding="utf-8") as mf:
                        matches = json.load(mf)

                # find processed image file if any
                processed_files = [str(p.name) for p in result_folder.iterdir() if p.suffix.lower() in ALLOWED_EXTENSIONS]

                # Build response compatible with the Streamlit UI: provide `invoice` and `products`.
                invoice_payload = None
                if sample and isinstance(sample, dict):
                    invoice_payload = sample.get("extracted") or sample

                matched_products = []
                if invoice_payload and isinstance(invoice_payload, dict):
                    prod_items = invoice_payload.get("products") or invoice_payload.get("items") or []
                elif isinstance(invoice_payload, list):
                    prod_items = invoice_payload
                else:
                    prod_items = []

                matcher = _state.get("product_matcher")
                for it in prod_items:
                    if isinstance(it, dict):
                        raw_name = it.get("name") or it.get("product_name") or it.get("extracted_text") or ""
                    else:
                        raw_name = str(it)
                    match_result = None
                    try:
                        if matcher and raw_name:
                            match_result = matcher.hybrid_match(raw_name)
                            # Convert MatchResult objects to plain dicts for JSON serialization
                            if hasattr(match_result, "to_dict") and callable(match_result.to_dict):
                                match_result = match_result.to_dict()
                    except Exception:
                        match_result = None
                    # include any external top matches alongside the matcher result
                    external_top = None
                    if matches and isinstance(matches, dict):
                        # matches['summary'] is a list of {query, matched, top_score}
                        for s in matches.get("summary", []):
                            if s.get("query") == raw_name:
                                external_top = s.get("matched")
                                break

                    entry = {**(it if isinstance(it, dict) else {"name": raw_name})}
                    entry["match"] = match_result
                    if external_top is not None:
                        entry["top_matches"] = external_top
                    matched_products.append(entry)

                return JSONResponse(
                    content={
                        "result_id": result_id,
                        "filename": file.filename,
                        "invoice": {k: v for k, v in (invoice_payload or {}).items() if k != "products"},
                        "products": matched_products,
                        "matches_summary": matches,
                        "processed_files": processed_files,
                    }
                )
            except Exception:
                log.exception("Failed to collect outputs from process_uploaded_image.py; falling back to inline pipeline")

        # Preprocess (fallback inline)
        # Preprocess using the user's pipeline function
        from PIL import Image

        with Image.open(tmp_path) as img:
            pil_image = preprocess_full_pipeline(img, max_width=600)

        # Extract invoice data with Gemini
        extraction = extract_invoice_data_gemini(_state["gemini_model"], pil_image, max_retries=3)
        if extraction is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Gemini extraction returned no data.",
            )

        # Match products (single best match)
        products = extraction.get("products", [])
        matched_products = []
        for item in products:
            raw_name = item.get("name") or item.get("product_name", "")
            if not raw_name:
                matched_products.append({**item, "match": None})
                continue
            match_result = _state["product_matcher"].hybrid_match(raw_name)
            if hasattr(match_result, "to_dict") and callable(match_result.to_dict):
                match_result = match_result.to_dict()
            matched_products.append({**item, "match": match_result})

        # Persist result files: processed image, sample_invoice.json, matches_summary_fallback.json
        try:
            # create per-result folder under ocr/results
            result_folder = REPO_ROOT / "ocr" / "results" / result_id
            result_folder.mkdir(parents=True, exist_ok=True)

            # Save processed image (pil_image is PIL.Image)
            proc_name = f"{result_id}{Path(file.filename or 'upload').suffix or '.jpg'}"
            proc_path = result_folder / proc_name
            try:
                pil_image.save(proc_path)
                log.info("Saved processed image to %s", proc_path)
            except Exception as e:
                log.warning("Failed to save processed image: %s", e)

            # Save sample_invoice.json
            sample = {"source_filename": file.filename, "extracted": extraction}
            sample_path = result_folder / "sample_invoice.json"
            try:
                with open(sample_path, "w", encoding="utf-8") as sf:
                    import json as _json

                    _json.dump(sample, sf, ensure_ascii=False, indent=2)
                log.info("Saved sample invoice JSON to %s", sample_path)
            except Exception as e:
                log.warning("Failed to save sample JSON: %s", e)

            # Prepare product database for top-10 matching
            product_db = []
            try:
                if PRODUCT_DB_PATH.exists():
                    df_master = pd.read_csv(PRODUCT_DB_PATH)
                    if not df_master.empty:
                        # take first meaningful column similar to startup logic
                        col0 = df_master.columns[0]
                        product_db = df_master[col0].dropna().astype(str).str.strip().tolist()
            except Exception:
                product_db = []

            # Compute top-10 matches per extracted product using rapidfuzz if available, else difflib
            matches_summary = []
            for item in products:
                qname = item.get("name") or item.get("product_name") or item.get("extracted_text") or ""
                if not qname:
                    continue
                top_matches = []
                try:
                    from rapidfuzz import process as _rf_process  # type: ignore

                    raw_top = _rf_process.extract(qname, product_db, limit=10, score_cutoff=0)
                    # rapidfuzz returns tuples (match, score, idx)
                    for m, score, _ in raw_top:
                        top_matches.append({"name": m, "score": float(score / 100.0)})
                except Exception:
                    # fallback to difflib
                    from difflib import SequenceMatcher

                    scores = [(p, SequenceMatcher(None, qname, p).ratio()) for p in product_db]
                    scores.sort(key=lambda x: x[1], reverse=True)
                    for p, s in scores[:10]:
                        top_matches.append({"name": p, "score": float(s)})

                matches_summary.append({"query": qname, "matched": top_matches, "top_score": top_matches[0]["score"] if top_matches else 0.0})

            matches_path = result_folder / "matches_summary_fallback.json"
            try:
                with open(matches_path, "w", encoding="utf-8") as mf:
                    import json as _json

                    _json.dump({"generated_at": __import__("datetime").datetime.now().isoformat(), "summary": matches_summary}, mf, ensure_ascii=False, indent=2)
                log.info("Saved matches summary to %s", matches_path)
            except Exception as e:
                log.warning("Failed to save matches summary: %s", e)
        except Exception as e:
            log.warning("Failed to persist result files: %s", e)

        # if we reach here we either ran inline processing or failed to collect external outputs
        return JSONResponse(
            content={
                "result_id": result_id,
                "filename": file.filename,
                "invoice": {k: v for k, v in extraction.items() if k != "products"},
                "products": matched_products,
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Error processing invoice: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {exc}",
        )
    finally:
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

    # Also persist confirmed products to the master CSV (if available)
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

                exists = False
                if not master_df.empty:
                    exists = master_df[target_col].astype(str).str.strip().eq(str(corr.correct_text).strip()).any()

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

    # Persist master_df back to CSV if modified
    try:
        if not master_df.empty:
            master_df.to_csv(PRODUCT_DB_PATH, index=False)
    except Exception as e:
        log.warning("Failed to save updated master product CSV: %s", e)

    return {
        "status": "ok",
        "applied": len(applied),
        "failed": len(failed),
        "details": {"applied": applied, "failed": failed},
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)