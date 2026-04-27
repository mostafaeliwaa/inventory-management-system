"""Process a single uploaded image: preprocess, run Gemini extraction, and produce
matching fallback summary. Writes three files into a new folder under `ocr/results/`:
 - processed image (same extension)
 - sample_invoice.json (extracted data)
 - matches_summary_fallback.json (top-10 matches for each item)

Usage:
    python ocr/process_uploaded_image.py [path/to/image.jpg]
If no path provided, the script selects the newest file in the repository `results/` folder.
"""
from pathlib import Path
import sys
import json
from datetime import datetime
import subprocess
import logging

from PIL import Image

# ensure repo imports work
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ocr"))
sys.path.insert(0, str(REPO_ROOT))

from complete_ocr_pipeline import preprocess_full_pipeline, setup_gemini, extract_invoice_data_gemini  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("process_uploaded_image")


def pick_latest_in_results() -> Path | None:
    results_dir = REPO_ROOT / "results"
    if not results_dir.exists():
        return None
    candidates = [p for p in results_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# Matching logic is delegated to ocr/run_matching_fallback.py; helpers removed.


def main():
    img_path = None
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
    else:
        img_path = pick_latest_in_results()

    if img_path is None or not img_path.exists():
        log.error("No image provided and no recent image found in results/")
        sys.exit(1)

    log.info("Processing image: %s", img_path)

    # Prepare output folder
    out_dir = REPO_ROOT / "ocr" / "results" / (img_path.stem + "_processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess
    try:
        with Image.open(img_path) as im:
            processed = preprocess_full_pipeline(im, max_width=600)
            processed_path = out_dir / (img_path.stem + img_path.suffix)
            processed.save(processed_path)
            log.info("Saved processed image to %s", processed_path)
    except Exception as e:
        log.exception("Preprocessing failed: %s", e)
        sys.exit(1)

    # Setup Gemini
    try:
        model = setup_gemini()
    except Exception as e:
        log.exception("Failed to setup Gemini: %s", e)
        model = None

    # Extract invoice data
    try:
        extraction = extract_invoice_data_gemini(model, processed, max_retries=3)
        sample_json = {"source_image": str(img_path), "extracted": extraction}
        sample_path = out_dir / "sample_invoice.json"
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(sample_json, f, ensure_ascii=False, indent=2)
        log.info("Saved extraction to %s", sample_path)
    except Exception as e:
        log.exception("Extraction failed: %s", e)
        sample_path = out_dir / "sample_invoice.json"
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump({"source_image": str(img_path), "error": str(e)}, f, ensure_ascii=False, indent=2)

    # Matching fallback: delegate to ocr/run_matching_fallback.py
    # Write the sample JSON to the location the fallback script expects
    results_model_dir = REPO_ROOT / "ocr" / "resultsOfModel"
    results_model_dir.mkdir(parents=True, exist_ok=True)
    external_sample = results_model_dir / "sample_invoice.json"
    try:
        with open(external_sample, "w", encoding="utf-8") as f:
            json.dump(sample_json, f, ensure_ascii=False, indent=2)
        log.info("Wrote sample invoice for fallback matcher to %s", external_sample)
    except Exception:
        log.exception("Failed to write external sample invoice for fallback matcher")

    # Run the fallback matcher script; it's intentionally a separate script
    try:
        proc = subprocess.run([sys.executable, str(REPO_ROOT / "ocr" / "run_matching_fallback.py")], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
        log.info("run_matching_fallback stdout: %s", proc.stdout.strip())
        if proc.returncode != 0:
            log.warning("run_matching_fallback returned %s: %s", proc.returncode, proc.stderr.strip())
    except Exception as e:
        log.exception("Failed to execute run_matching_fallback.py: %s", e)

    external_matches = results_model_dir / "matches_summary_fallback.json"
    out_path = out_dir / "matches_summary_fallback.json"
    if external_matches.exists():
        try:
            with open(external_matches, "r", encoding="utf-8") as f:
                ext = json.load(f)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(ext, f, ensure_ascii=False, indent=2)
            log.info("Copied fallback matches to %s", out_path)
        except Exception:
            log.exception("Failed to copy external matches to output folder")
    else:
        # Fallback: produce an empty summary file so downstream consumers have expected file
        out = {"generated_at": datetime.now().isoformat(), "summary": []}
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            log.info("Saved empty matches summary to %s", out_path)
        except Exception:
            log.exception("Failed to write empty matches summary to %s", out_path)

    print("Done. Output directory:", out_dir)


if __name__ == "__main__":
    main()
