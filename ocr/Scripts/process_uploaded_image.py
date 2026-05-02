from pathlib import Path
import sys
import json
import logging
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ocr"))
sys.path.insert(0, str(REPO_ROOT))

from ocr.Models.complete_ocr_pipeline import (
    preprocess_full_pipeline,
    extract_invoice_data_gemini
)
from ocr.Scripts.run_matching_fallback import run_fallback_matching

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("process_uploaded_image")


def process_image(img_path: Path, matcher, model) -> dict:
 
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    if matcher is None:
        raise ValueError("Matcher is not initialized")

    if model is None:
        raise ValueError("Gemini model is not initialized")

    log.info("Processing image: %s", img_path)

    out_dir = REPO_ROOT / "ocr" / "results" / (img_path.stem + "_processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(img_path) as im:
        processed = preprocess_full_pipeline(im, max_width=600)
        processed_path = out_dir / (img_path.stem + img_path.suffix)
        processed.save(processed_path)

    try:
        extraction = extract_invoice_data_gemini(
            model,
            processed,
            max_retries=3
        )
    except Exception as e:
        log.error("Gemini error: %s", e)
        extraction = None

    if extraction is None:
        log.warning("Gemini failed → using empty extraction")
        extraction = {"items": []}

    sample_path = out_dir / "sample_invoice.json"
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_image": str(img_path),
                "extracted": extraction
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    try:
        matches_result, matches_path = run_fallback_matching(
            extraction=extraction,
            matcher=matcher,
            output_dir=out_dir
        )
    except Exception as e:
        log.error("Matching failed: %s", e)
        matches_result = {}
        matches_path = None

    log.info("Pipeline done → %s", out_dir)

    return {
        "output_dir": out_dir,
        "sample": sample_path,
        "matches": matches_path,
        "image": processed_path,
        "matches_result": matches_result
    }
