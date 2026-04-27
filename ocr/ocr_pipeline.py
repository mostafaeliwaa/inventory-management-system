"""
Complete Image Processing and OCR Extraction Pipeline
=====================================================

A production-ready pipeline for:
1. Recursively scanning folders for images (handles Windows paths)
2. Preprocessing invoice images (PDF pages or image files)
3. Sending processed images to Gemini Vision model
4. Extracting structured invoice data using optimized prompts
5. Outputting clean structured JSON ready for product matching system

Windows file paths: D:\AI 4\Graduation Project\inventory-management-system\ocr\...
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import glob
import json
import json_repair
from tqdm import tqdm
import google.generativeai as genai
from pathlib import Path, PureWindowsPath
import sys
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Gemini API setup
try:
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel("models/gemini-3.1-flash-lite-preview")
    GEMINI_AVAILABLE = True
except Exception as e:
    logger.warning(f"Gemini API not configured: {e}")
    GEMINI_AVAILABLE = False
    model = None

# ============================================================================
# IMAGE PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_full_pipeline(pil_image: Image.Image, max_width: int = 600) -> Image.Image:
    """
    Full preprocessing pipeline for invoice images:
    - HSV enhancement for better color separation
    - Convert to grayscale
    - Resize image for consistent processing
    - Contrast enhancement for better OCR accuracy
    
    Args:
        pil_image: PIL Image object
        max_width: Maximum width for resizing (default: 600px)
    
    Returns:
        Preprocessed PIL Image
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
    
    Args:
        image_path: Path to image file
        max_width: Maximum width for resizing
    
    Returns:
        Preprocessed PIL Image or None if error
    """
    try:
        # Handle Windows paths with spaces
        image_path = str(Path(image_path))
        
        # Load image
        pil_image = Image.open(image_path)
        
        # Preprocess
        processed_image = preprocess_full_pipeline(pil_image, max_width)
        
        logger.debug(f"Successfully preprocessed: {image_path}")
        return processed_image
        
    except Exception as e:
        logger.error(f"Error preprocessing {image_path}: {e}")
        return None


# ============================================================================
# FOLDER SCANNING AND IMAGE DISCOVERY
# ============================================================================

def scan_images_recursively(root_dir: str, 
                           extensions: List[str] = None) -> List[str]:
    """
    Recursively scan folders for images with given extensions.
    Handles Windows paths with spaces and special characters.
    
    Args:
        root_dir: Root directory to scan
        extensions: List of file extensions to include (default: common image formats)
    
    Returns:
        List of absolute paths to image files
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    
    # Convert to Path object for cross-platform compatibility
    root_path = Path(root_dir)
    
    if not root_path.exists():
        logger.warning(f"Directory does not exist: {root_dir}")
        return []
    
    image_paths = []
    
    # Walk through directory tree
    for ext in extensions:
        # Use glob with recursive pattern
        pattern = str(root_path / '**' / f'*{ext}')
        matches = glob.glob(pattern, recursive=True)
        
        # Also check uppercase extensions
        pattern_upper = str(root_path / '**' / f'*{ext.upper()}')
        matches_upper = glob.glob(pattern_upper, recursive=True)
        
        image_paths.extend(matches)
        image_paths.extend(matches_upper)
    
    # Remove duplicates and sort
    image_paths = sorted(list(set(image_paths)))
    
    logger.info(f"Found {len(image_paths)} images in {root_dir}")
    return image_paths


def get_image_folders(base_dirs: List[str] = None) -> Dict[str, List[str]]:
    """
    Get images from multiple base directories.
    
    Args:
        base_dirs: List of base directories to scan
    
    Returns:
        Dictionary with folder name as key and list of image paths as value
    """
    if base_dirs is None:
        # Use the default folders from the project
        base_dirs = [
            "ocr/processing_images",
            "ocr/original_data"
        ]
    
    folder_images = {}
    
    for base_dir in base_dirs:
        # Convert to absolute path
        abs_dir = str(Path(base_dir).absolute())
        
        if not Path(base_dir).exists():
            logger.warning(f"Directory does not exist: {base_dir}")
            continue
        
        # Get all subdirectories
        for subdir in Path(base_dir).iterdir():
            if subdir.is_dir():
                folder_name = subdir.name
                images = scan_images_recursively(str(subdir))
                
                if images:
                    folder_images[folder_name] = images
                    logger.info(f"Found {len(images)} images in {folder_name}")
    
    # Also include the base directories themselves if they contain images
    for base_dir in base_dirs:
        base_path = Path(base_dir)
        if base_path.exists():
            images = scan_images_recursively(str(base_path))
            if images:
                folder_name = base_path.name
                if folder_name not in folder_images:
                    folder_images[folder_name] = images
                else:
                    folder_images[folder_name].extend(images)
    
    return folder_images


# ============================================================================
# GEMINI OCR EXTRACTION
# ============================================================================

def get_invoice_extraction_prompt() -> str:
    """
    Returns the optimized prompt for invoice data extraction.
    This prompt is designed for Egyptian/Arabic invoices with mixed languages.
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


def extract_invoice_data_gemini(image: Image.Image, 
                               max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """
    Send image to Gemini Vision API and extract structured invoice data.
    
    Args:
        image: PIL Image object (preprocessed)
        max_retries: Maximum number of retry attempts
    
    Returns:
        Dictionary with extracted invoice data or None if error
    """
    if not GEMINI_AVAILABLE or model is None:
        logger.error("Gemini API is not available. Please configure GEMINI_API_KEY.")
        return None
    
    prompt = get_invoice_extraction_prompt()
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Sending image to Gemini API (attempt {attempt + 1}/{max_retries})")
            
            response = model.generate_content([
                prompt,
                image
            ])
            
            raw_response = response.text
            
            # Clean up the response - remove markdown code blocks if present
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
                # Try json_repair
                try:
                    parsed_data = json_repair.loads(cleaned_response)
                    logger.info("Successfully repaired and parsed JSON")
                    return parsed_data
                except Exception as repair_error:
                    logger.error(f"JSON repair also failed: {repair_error}")
                    # If this is the last attempt, return the raw text
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
# PIPELINE ORCHESTRATION
# ============================================================================

def process_single_image(image_path: str, 
                        output_dir: str = "ocr/results",
                        save_preprocessed: bool = False) -> Dict[str, Any]:
    """
    Process a single image through the complete pipeline.
    
    Args:
        image_path: Path to the image file
        output_dir: Directory to save results and preprocessed images
        save_preprocessed: Whether to save the preprocessed image
    
    Returns:
        Dictionary with processing results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate unique ID for this processing run
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
        
        # Save preprocessed image if requested
        if save_preprocessed:
            preprocessed_path = output_path / f"{processing_id}_preprocessed.jpg"
            processed_image.save(str(preprocessed_path), 'JPEG', quality=90)
            result["preprocessed_path"] = str(preprocessed_path)
        
        # Step 2: Extract data with Gemini
        if GEMINI_AVAILABLE:
            extracted_data = extract_invoice_data_gemini(processed_image)
            
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


def process_image_batch(image_paths: List[str],
                       output_dir: str = "ocr/results",
                       batch_size: int = 10,
                       save_preprocessed: bool = False) -> List[Dict[str, Any]]:
    """
    Process a batch of images through the pipeline.
    
    Args:
        image_paths: List of image paths to process
        output_dir: Directory to save results
        batch_size: Number of images to process in parallel (not implemented yet)
        save_preprocessed: Whether to save preprocessed images
    
    Returns:
        List of processing results
    """
    results = []
    
    logger.info(f"Starting batch processing of {len(image_paths)} images")
    
    for i, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
        logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
        
        result = process_single_image(
            image_path=image_path,
            output_dir=output_dir,
            save_preprocessed=save_preprocessed
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

def save_results_to_json(results: List[Dict[str, Any]], 
                        output_file: str = "ocr/results/ocr_results.json") -> str:
    """
    Save processing results to a JSON file.
    
    Args:
        results: List of processing results
        output_file: Path to output JSON file
    
    Returns:
        Path to the saved file
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
        "completed_ids": [r.get("processing_id") for r in results if r.get("status") == "completed"],
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }

    payload = {
        "summary": summary,
        "results": results
    }

    # Ensure parent directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved OCR results to {output_path}")
    return str(output_path)


# Simple CLI entrypoint so the pipeline can be run from the command line.
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR pipeline on a folder of images")
    parser.add_argument("--scan-dir", type=str, default="ocr/processing_images",
                        help="Directory to scan for images")
    parser.add_argument("--output-dir", type=str, default="ocr/results",
                        help="Directory to write results to")
    parser.add_argument("--save-preprocessed", action="store_true",
                        help="Save preprocessed images to output directory")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (not implemented as parallelism)")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Limit to first N images (0 = no limit)")

    args = parser.parse_args()

    # Discover images
    image_paths = scan_images_recursively(args.scan_dir)
    if args.max_images and args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    if not image_paths:
        logger.warning(f"No images found in {args.scan_dir}. Exiting.")
        return

    # Process images
    results = process_image_batch(
        image_paths=image_paths,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        save_preprocessed=args.save_preprocessed
    )

    # Save results
    out_file = Path(args.output_dir) / "ocr_results.json"
    saved = save_results_to_json(results, str(out_file))
    print(f"Results saved to: {saved}")


if __name__ == "__main__":
    main()