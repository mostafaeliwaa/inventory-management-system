# ocr/__init__.py
"""OCR Package for Invoice Processing"""

from .supabase_client import SupabaseProductDB, get_supabase_client
from .product_matcher import normalize_product_name
from .product_matcher_system import ProductMatcher
from .complete_ocr_pipeline import setup_gemini, extract_invoice_data_gemini, preprocess_full_pipeline
from .process_uploaded_image import process_image
from .run_matching_fallback import run_fallback_matching, build_matcher

__all__ = [
    'SupabaseProductDB',
    'get_supabase_client',
    'normalize_product_name',
    'ProductMatcher',
    'setup_gemini',
    'extract_invoice_data_gemini',
    'preprocess_full_pipeline',
    'process_image',
    'run_fallback_matching',
    'build_matcher'
]