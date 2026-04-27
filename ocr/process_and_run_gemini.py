"""Preprocess images from original_data/invoices1, save to processing_images/invoices1
with unique names (image_001, ...), run Gemini extraction on each processed
image, and save per-image results into ocr/resultsOfModel as JSON.

Run from repository root:
    python ocr/process_and_run_gemini.py
"""
from pathlib import Path
import json
import sys
from PIL import Image
import importlib.util


def load_module_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    repo_root = Path(__file__).resolve().parents[1]
    pipeline_path = repo_root / "ocr" / "complete_ocr_pipeline.py"
    if not pipeline_path.exists():
        print(f"Pipeline not found at {pipeline_path}")
        sys.exit(1)

    pipeline = load_module_from_path(pipeline_path, "complete_ocr_pipeline")

    src_dir = repo_root / "ocr" / "original_data" / "invoices1"
    dst_dir = repo_root / "ocr" / "processing_images" / "invoices1"
    results_dir = repo_root / "ocr" / "resultsOfModel"

    dst_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp", "*.webp"):
        images.extend(sorted(src_dir.rglob(ext)))

    if not images:
        print(f"No images found in {src_dir}")
        return

    # Preprocess and save with unique names
    processed_paths = []
    for i, img_path in enumerate(images, start=1):
        try:
            pil = Image.open(img_path)
            processed = pipeline.preprocess_full_pipeline(pil)
            name = f"image_{i:03d}{img_path.suffix.lower()}"
            out_path = dst_dir / name
            processed.save(out_path)
            processed_paths.append(out_path)
            print(f"Saved processed image: {out_path}")
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

    # Setup Gemini model (uses GEMINI_API_KEY env var if not provided)
    model = pipeline.setup_gemini()

    # Run extraction and save per-image JSON
    for p in processed_paths:
        try:
            pil = Image.open(p)
            extracted = pipeline.extract_invoice_data_gemini(model, pil)
            out_json = results_dir / (p.stem + ".json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump({
                    "source_image": str(p),
                    "extracted": extracted
                }, f, ensure_ascii=False, indent=2)
            print(f"Saved result: {out_json}")
        except Exception as e:
            print(f"Failed to extract for {p}: {e}")


if __name__ == "__main__":
    main()
