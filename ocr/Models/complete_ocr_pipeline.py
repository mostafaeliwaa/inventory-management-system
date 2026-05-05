import os
import sys
import json
import glob
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import google.generativeai as genai
from tqdm import tqdm
try:
    import json_repair
except ImportError:
    json_repair = json
    logging.warning("json_repair not installed — fallback to json (no auto-repair)")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)

def preprocess_image(
    image: Image.Image,
    max_width: int = 600,
    contrast: float = 1.5
) -> Image.Image:
    """
    Preprocess invoice image to improve OCR accuracy.
    
    Steps:
    1. Convert to OpenCV format
    2. Apply HSV enhancement
    3. Convert to grayscale
    4. Resize (if needed)
    5. Enhance contrast
    """

    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    H_FACTOR = 0.1
    S_FACTOR = 4.0
    V_FACTOR = 0.5

    hsv[:, :, 0] *= H_FACTOR
    hsv[:, :, 1] *= S_FACTOR
    hsv[:, :, 2] *= V_FACTOR

    hsv = np.clip(hsv, [0, 0, 0], [179, 255, 255]).astype(np.uint8)

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(img).convert("L")

    if img.width > max_width:
        ratio = max_width / img.width
        new_size = (max_width, int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    img = ImageEnhance.Contrast(img).enhance(contrast)

    return img

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
    Initialize Gemini model.
    - Uses provided API key OR environment variable.
    - Returns ready-to-use model or None if failed.
    """
    try:
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            logger.warning("No Gemini API key provided. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
            return None
        genai.configure(api_key=key)

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

def extract_invoice_data_gemini(
    model,
    image: Image.Image,
    max_retries: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Send image to Gemini and return structured invoice data (JSON).
    """

    if model is None:
        logger.error("❌ Gemini model not initialized")
        return None

    prompt = get_invoice_extraction_prompt()

    for attempt in range(max_retries):
        try:
            logger.info(f"📤 Sending request to Gemini (attempt {attempt + 1})")

            response = model.generate_content([prompt, image])
            raw_text = (response.text or "").strip()

            cleaned = raw_text.replace("```json", "").replace("```", "").strip()

            try:
                data = json_repair.loads(cleaned)
                logger.info("✅ Invoice extracted successfully")
                return data

            except Exception as parse_error:
                logger.warning(f"⚠️ JSON parsing failed: {parse_error}")
                if attempt == max_retries - 1:
                    return {
                        "error": "Invalid JSON from model",
                        "raw_response": raw_text[:500]
                    }

        except Exception as api_error:
            logger.error(f"❌ Gemini API error: {api_error}")

            if attempt == max_retries - 1:
                return {
                    "error": str(api_error)
                }

    return None

