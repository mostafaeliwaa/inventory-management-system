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

# IMAGE PREPROCESSING FUNCTIONS

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
    
    hsv[:, :, 0] *= 0.1    
    hsv[:, :, 1] *= 4.0    
    hsv[:, :, 2] *= 0.5    
    
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

# FOLDER SCANNING AND IMAGE DISCOVERY

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

# GEMINI API INTEGRATION
def resolve_gemini_model(preferred: Optional[str] = None) -> Optional[str]:
    """
    Pick an available Gemini model that supports generateContent.

    Order:
      1. `preferred` arg, if given and supported.
      2. $GEMINI_MODEL env var, if set and supported.
      3. First match from a prioritized candidate list.
      4. Any listed model whose name contains 'flash' and supports generateContent.
    """
    candidates = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-flash-latest",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]

    try:
        available = []
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                # API returns names like "models/gemini-2.5-flash"
                available.append(m.name.removeprefix("models/"))
    except Exception as e:
        logger.error(f"Failed to list Gemini models: {e}")
        return preferred or os.environ.get("GEMINI_MODEL")

    def _pick(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        bare = name.removeprefix("models/")
        return bare if bare in available else None

    chosen = (
        _pick(preferred)
        or _pick(os.environ.get("GEMINI_MODEL"))
        or next((c for c in candidates if c in available), None)
        or next((n for n in available if "flash" in n), None)
        or (available[0] if available else None)
    )

    if chosen:
        logger.info(f"Resolved Gemini model: {chosen}")
    else:
        logger.error("No Gemini model with generateContent support found")
    return chosen


def setup_gemini(api_key: str = None, model_name: Optional[str] = None):
    """
    Setup Gemini API. Dynamically resolves an available model.
    """
    try:
        if api_key:
            genai.configure(api_key=api_key)
        elif "GEMINI_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        else:
            logger.warning("No Gemini API key provided. Set GEMINI_API_KEY environment variable.")
            return None

        resolved = resolve_gemini_model(model_name)
        if not resolved:
            return None

        model = genai.GenerativeModel(resolved)
        logger.info(f"Gemini API configured successfully with model: {resolved}")
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

