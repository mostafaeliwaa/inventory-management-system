"""
Complete Image Processing and OCR Extraction Pipeline
=====================================================

A production-ready pipeline for:
1. Recursively scanning folders for images (handles Windows paths)
2. Preprocessing invoice images (PDF pages or image files)
3. Sending processed images to Gemini Vision model
4. Extracting structured invoice data using optimized prompts
5. Outputting clean structured JSON ready for product matching system
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import glob
import json
import logging
try:
    import json_repair
except Exception:
    json_repair = json
    # Optional dependency missing; warn via print (logging configured later)
    print("Optional dependency 'json_repair' not found — falling back to 'json' (no repair features)")
from tqdm import tqdm
import google.generativeai as genai
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# IMAGE PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_full_pipeline(pil_image: Image.Image, max_width: int = 600) -> Image.Image:
    """
    Full preprocessing pipeline for invoice images.
    """
    # PIL → numpy (RGB)
    image = np.array(pil_image)
    
    # RGB → BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # HSV enhancement
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Adjust HSV channels for better text visibility
    hsv[:, :, 0] *= 0.1    # Hue - reduce color variation
    hsv[:, :, 1] *= 4.0    # Saturation - increase color intensity
    hsv[:, :, 2] *= 0.5    # Value - reduce brightness for better contrast
    
    # Clip values to valid ranges
    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    
    hsv = hsv.astype(np.uint8)
    
    # Convert back to BGR → RGB
    image_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    pil_img = Image.fromarray(image_rgb)
    
    # Convert to grayscale
    gray_image = pil_img.convert('L')
    
    # Resize if too large
    if gray_image.width > max_width:
        ratio = max_width / gray_image.width
        new_height = int(gray_image.height * ratio)
        gray_image = gray_image.resize((max_width, new_height), Image.LANCZOS)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(gray_image)
    enhanced_image = enhancer.enhance(1.5)
    
    return enhanced_image

def load_and_preprocess_image(image_path: str, max_width: int = 600) -> Optional[Image.Image]:
    """
    Load an image from file and apply preprocessing.
    """
    try:
        image_path = str(Path(image_path))
        pil_image = Image.open(image_path)
        processed_image = preprocess_full_pipeline(pil_image, max_width)
        logger.debug(f"Successfully preprocessed: {image_path}")
        return processed_image
    except Exception as e:
        logger.error(f"Error preprocessing {image_path}: {e}")
        return None

# ============================================================================
# FOLDER SCANNING AND IMAGE DISCOVERY
# ============================================================================

def scan_images_recursively(root_dir: str, extensions: List[str] = None) -> List[str]:
    """
    Recursively scan folders for images with given extensions.
    Handles Windows paths with spaces and special characters.
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    
    root_path = Path(root_dir)
    
    if not root_path.exists():
        logger.warning(f"Directory does not exist: {root_dir}")
        return []
    
    image_paths = []
    
    for ext in extensions:
        pattern = str(root_path / '**' / f'*{ext}')
        matches = glob.glob(pattern, recursive=True)
        pattern_upper = str(root_path / '**' / f'*{ext.upper()}')
        matches_upper = glob.glob(pattern_upper, recursive=True)
        image_paths.extend(matches)
        image_paths.extend(matches_upper)
    
    image_paths = sorted(list(set(image_paths)))
    logger.info(f"Found {len(image_paths)} images in {root_dir}")
    return image_paths

def get_all_images_from_folders(folders: List[str] = None) -> List[str]:
    """
    Get all images from specified folders.
    """
    if folders is None:
        folders = [
            "ocr/processing_images",
            "ocr/original_data"
        ]
    
    all_images = []
    for folder in folders:
        images = scan_images_recursively(folder)
        all_images.extend(images)
        logger.info(f"Found {len(images)} images in {folder}")
    
    return sorted(list(set(all_images)))

# ============================================================================
# GEMINI API INTEGRATION
# ============================================================================

def setup_gemini(api_key: str = None):
    """
    Setup Gemini API with optional API key.
    """
    try:
        if api_key:
            genai.configure(api_key=api_key)
        elif "GEMINI_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        else:
            logger.warning("No Gemini API key provided. Set GEMINI_API_KEY environment variable.")
            return None
        #models/gemini-3.1-flash-lite-preview    
        model = genai.GenerativeModel("models/gemini-flash-latest")
        logger.info("Gemini API configured successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to setup Gemini API: {e}")
        return None

def get_invoice_extraction_prompt() -> str:
    """
    Returns the optimized prompt for invoice data extraction.
    """
    return """أنت خبير في معالجة الفواتير والوثائق التجارية في السوق المصري.

قم بتحليل صورة الفاتورة واستخراج البيانات التالية بدقة:

1. **المعلومات الأساسية**:
   - اسم المتجر/الشركة (إن وجد)
   - تاريخ الفاتورة
   - رقم الفاتورة
   - إجمالي المبلغ

2. **المنتجات/السلع** (الأهم):
   لكل عنصر في الفاتورة، استخرج:
   - اسم المنتج (بالعربية أو الإنجليزية)
   - الكمية
   - السعر للوحدة
   - الإجمالي للعنصر (إن وجد)

3. **ملاحظات مهمة**:
   - الفواتير قد تكون باللغة العربية أو الإنجليزية أو مزيج منهما
   - قد تحتوي على أرقام عربية (٠١٢٣٤٥٦٧٨٩) أو أرقام لاتينية
   - قد تكون هناك اختصارات أو أخطاء إملائية
   - التركيز على دقة أسماء المنتجات أكثر من أي شيء آخر

4. **تنسيق الإخراج**:
أرجع النتيجة بتنسيق JSON فقط، بدون أي نص إضافي، بالشكل التالي:
{
  "store_name": "اسم المتجر إن وجد",
  "invoice_date": "التاريخ إن وجد",
  "invoice_number": "رقم الفاتورة إن وجد",
  "total_amount": "الإجمالي إن وجد",
  "items": [
    {
      "product_name": "اسم المنتج",
      "quantity": "الكمية (رقم)",
      "unit_price": "سعر الوحدة (رقم)",
      "item_total": "إجمالي العنصر (رقم، إن وجد)"
    }
  ],
  "currency": "العملة (مثل جنيه، EGP، ريال، إلخ)",
  "notes": "أي ملاحظات إضافية"
}

إذا لم تتمكن من استخراج بيانات معينة، اترك القيمة فارغة "" أو null.
ركز على استخراج أسماء المنتجات بدقة لأنها ستستخدم لمطابقة المنتجات في قاعدة بيانات."""

def extract_invoice_data_gemini(model, image: Image.Image, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """
    Send image to Gemini Vision API and extract structured invoice data.
    """
    if model is None:
        logger.error("Gemini model not initialized")
        return None
    
    prompt = get_invoice_extraction_prompt()
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Sending image to Gemini API (attempt {attempt + 1}/{max_retries})")
            
            response = model.generate_content([prompt, image])
            raw_response = response.text
            
            # Clean up the response
            cleaned_response = raw_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            # Try to parse as JSON
            try:
                parsed_data = json_repair.loads(cleaned_response)
                logger.info("Successfully extracted invoice data from Gemini")
                return parsed_data
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed, attempting repair: {e}")
                try:
                    parsed_data = json_repair.loads(cleaned_response)
                    logger.info("Successfully repaired and parsed JSON")
                    return parsed_data
                except Exception as repair_error:
                    logger.error(f"JSON repair also failed: {repair_error}")
                    if attempt == max_retries - 1:
                        return {
                            "error": "Failed to parse JSON response",
                            "raw_response": raw_response[:500] + "..." if len(raw_response) > 500 else raw_response
                        }
            
        except Exception as e:
            logger.error(f"Gemini API error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return {"error": str(e)}
    
    return None

# ============================================================================
# PIPELINE PROCESSING
# ============================================================================

def process_single_image(image_path: str, model, output_dir: str = "ocr/results") -> Dict[str, Any]:
    """
    Process a single image through the complete pipeline.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = Path(image_path).stem
    processing_id = f"{image_name}_{timestamp}"
    
    result = {
        "processing_id": processing_id,
        "image_path": str(image_path),
        "image_name": image_name,
        "timestamp": timestamp,
        "status": "pending",
        "preprocessed": False,
        "gemini_processed": False,
        "error": None
    }
    
    try:
        # Step 1: Load and preprocess image
        logger.info(f"Processing image: {image_path}")
        processed_image = load_and_preprocess_image(image_path)
        
        if processed_image is None:
            result["status"] = "failed"
            result["error"] = "Failed to preprocess image"
            return result
        
        result["preprocessed"] = True
        
        # Step 2: Extract data with Gemini
        if model:
            extracted_data = extract_invoice_data_gemini(model, processed_image)
            
            if extracted_data:
                result["gemini_processed"] = True
                result["extracted_data"] = extracted_data
                result["status"] = "completed"
            else:
                result["status"] = "partial"
                result["error"] = "Gemini extraction returned no data"
        else:
            result["status"] = "skipped"
            result["error"] = "Gemini API not available"
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        result["status"] = "failed"
        result["error"] = str(e)
    
    return result

def process_image_batch(image_paths: List[str], model, output_dir: str = "ocr/results") -> List[Dict[str, Any]]:
    """
    Process a batch of images through the pipeline.
    """
    results = []
    
    logger.info(f"Starting batch processing of {len(image_paths)} images")
    
    for i, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
        logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
        
        result = process_single_image(
            image_path=image_path,
            model=model,
            output_dir=output_dir
        )
        
        results.append(result)
        
        # Save intermediate results every 10 images
        if (i + 1) % 10 == 0:
            batch_results_path = Path(output_dir) / f"batch_results_{i+1}.json"
            with open(batch_results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved intermediate results to {batch_results_path}")
    
    return results

# ============================================================================
# OUTPUT AND EXPORT
# ============================================================================

def save_results_to_json(results: List[Dict[str, Any]], output_file: str = "ocr/results/ocr_results.json") -> str:
    """
    Save processing results to a JSON file.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create summary statistics
    total = len(results)
    completed = sum(1 for r in results if r.get("status") == "completed")
    failed = sum(1 for r in results if r.get("status") == "failed")
    partial = sum(1 for r in results if r.get("status") == "partial")
    
    summary = {
        "total_images": total,
        "completed": completed,
        "failed": failed,
        "partial": partial,
        "success_rate": (completed / total * 100) if total > 0 else 0
    }
    
    # Prepare the complete output data
    output_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_results": total,
            "pipeline_version": "1.0.0"
        },
        "summary": summary,
        "results": results
    }
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(results)} results to {output_path}")
    logger.info(f"Summary: {completed} completed, {failed} failed, {partial} partial")
    
    return str(output_path)

def export_for_product_matching(results: List[Dict[str, Any]], output_file: str = "ocr/results/products_for_matching.json") -> str:
    """
    Extract product data from results for use with product matching system.
    """
    products = []
    
    for result in results:
        if result.get("status") == "completed" and "extracted_data" in result:
            extracted = result["extracted_data"]
            
            # Extract items from the invoice
            if "items" in extracted and isinstance(extracted["items"], list):
                for item in extracted["items"]:
                    if isinstance(item, dict) and "product_name" in item:
                        product_entry = {
                            "source_image": result["image_path"],
                            "processing_id": result["processing_id"],
                            "extracted_text": item.get("product_name", ""),
                            "quantity": item.get("quantity"),
                            "unit_price": item.get("unit_price"),
                            "item_total": item.get("item_total"),
                            "invoice_metadata": {
                                "store_name": extracted.get("store_name"),
                                "invoice_date": extracted.get("invoice_date"),
                                "invoice_number": extracted.get("invoice_number"),
                                "currency": extracted.get("currency")
                            }
                        }
                        products.append(product_entry)
    
    # Prepare output
    output_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_products": len(products),
            "source_images": len(set(p["source_image"] for p in products))
        },
        "products": products
    }
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Exported {len(products)} products for matching to {output_path}")
    return str(output_path)

# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def run_complete_pipeline(
    api_key: str = None,
    input_folders: List[str] = None,
    output_dir: str = "ocr/results",
    max_images: int = None,
    test_mode: bool = False
) -> Dict[str, Any]:
    """
    Run the complete OCR pipeline.
    
    Args:
        api_key: Gemini API key (optional, can use GEMINI_API_KEY env var)
        input_folders: List of folders to scan for images
        output_dir: Directory to save results
        max_images: Maximum number of images to process (for testing)
        test_mode: If True, only process a few images
    
    Returns:
        Dictionary with pipeline results
    """
    logger.info("=" * 40)
    logger.info("Starting complete OCR pipeline")

    # Prepare model
    model = setup_gemini(api_key)

    # Discover images
    if input_folders:
        images = []
        for f in input_folders:
            images.extend(scan_images_recursively(f))
        images = sorted(list(set(images)))
    else:
        images = get_all_images_from_folders()

    if max_images and max_images > 0:
        images = images[:max_images]

    if test_mode:
        images = images[:5]

    logger.info(f"Pipeline will process {len(images)} images")

    if not images:
        logger.warning("No images found; exiting pipeline")
        return {"error": "no_images_found", "processed": 0}

    # Ensure output dir exists
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Run processing
    results = process_image_batch(images, model, output_dir=output_dir)

    # Save results and export products
    results_file = Path(output_dir) / "ocr_results.json"
    saved_results = save_results_to_json(results, str(results_file))

    products_file = Path(output_dir) / "products_for_matching.json"
    exported_products = export_for_product_matching(results, str(products_file))

    # Prepare summary
    total = len(results)
    completed = sum(1 for r in results if r.get("status") == "completed")
    failed = sum(1 for r in results if r.get("status") == "failed")
    partial = sum(1 for r in results if r.get("status") == "partial")

    summary = {
        "total": total,
        "completed": completed,
        "failed": failed,
        "partial": partial,
        "results_file": str(saved_results),
        "products_file": str(exported_products)
    }

    logger.info(f"Pipeline finished: {completed} completed, {failed} failed, {partial} partial")
    return {"summary": summary, "results": results}


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run complete OCR pipeline")
    parser.add_argument("--api-key", type=str, default=None, help="Gemini API key")
    parser.add_argument("--input-folders", type=str, nargs="*", default=None,
                        help="One or more folders to scan for images")
    parser.add_argument("--output-dir", type=str, default="ocr/results", help="Directory to write outputs")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images processed (0 = no limit)")
    parser.add_argument("--test", action="store_true", help="Test mode: process only a few images")
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    max_images = args.max_images if args.max_images and args.max_images > 0 else None

    result = run_complete_pipeline(
        api_key=args.api_key,
        input_folders=args.input_folders,
        output_dir=args.output_dir,
        max_images=max_images,
        test_mode=args.test
    )

    # Print concise summary
    if isinstance(result, dict) and "summary" in result:
        s = result["summary"]
        print(f"Processed: {s['total']} images — {s['completed']} completed, {s['failed']} failed, {s['partial']} partial")
        print(f"Results JSON: {s['results_file']}")
        print(f"Products JSON: {s['products_file']}")
    else:
        print("Pipeline did not produce a summary. See logs for details.")


if __name__ == "__main__":
    main()