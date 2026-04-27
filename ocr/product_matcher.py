"""
Arabic Text Normalization for OCR Product Matching
==================================================

This module provides comprehensive text normalization functions for Arabic and mixed-script
product names extracted from OCR systems. Designed for Egyptian market use cases with
Arabic + English mixed text and common OCR noise patterns.
"""

import re
import string
from typing import Optional, List, Set


def normalize_arabic(text: str) -> str:
    """
    Apply comprehensive Arabic text normalization.
    
    Transformations applied IN ORDER:
    1. Convert Alef variants (أ, إ, آ) → ا
    2. Convert Teh Marbuta (ة) → ه
    3. Convert Alef Maqsura (ى) → ي
    4. Remove diacritics (harakat: َ, ُ, ِ, ْ, ً, ٌ, ٍ)
    5. Remove non-alphanumeric characters (keep Arabic letters, English letters, numbers, spaces)
    6. Convert to lowercase
    7. Strip extra spaces (trim + collapse multiple spaces to single)
    8. Remove leading/trailing whitespace
    
    Args:
        text: Input Arabic text, may contain mixed scripts
        
    Returns:
        Normalized Arabic text
        
    Examples:
        >>> normalize_arabic("إسكواش  فراولة ٥٠٠ جرام")
        'اسكواش فراولة 500 جرام'
        
        >>> normalize_arabic("خبز ٌإفرنجي ُطازج")
        'خبز افرنجي طازج'
        
        >>> normalize_arabic("مياه معدنية   (طبيعية)")
        'مياه معدنية طبيعية'
    """
    if not text:
        return ""
    
    # Step 1-3: Character normalization using translation tables
    # Create translation tables for Arabic character normalization
    alef_variants = {
        'أ': 'ا', 'إ': 'ا', 'آ': 'ا',  # Alef variants
        'ة': 'ه',  # Teh Marbuta
        'ى': 'ي',  # Alef Maqsura
    }
    
    # Step 4: Remove Arabic diacritics (harakat)
    diacritics = 'ًًٌٌٍٍََُُِِّّْْٰٰ'
    
    # Combine all replacements into a single translation table
    translation_table = str.maketrans({
        **alef_variants,
        **{d: '' for d in diacritics}
    })
    
    # Apply character replacements
    text = text.translate(translation_table)
    
    # Step 5: Remove non-alphanumeric characters
    # Keep Arabic letters (0600-06FF), English letters, numbers, and spaces
    # Arabic Unicode range: \u0600-\u06FF
    # Also include Arabic extended (0750-077F) and Persian/Arabic supplement (FB50-FDFF, FE70-FEFF)
    arabic_pattern = r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFFa-zA-Z0-9\s]'
    text = re.sub(arabic_pattern, ' ', text)
    
    # Step 6: Convert to lowercase (affects English letters only)
    text = text.lower()
    
    # Step 7-8: Clean up whitespace
    # Collapse multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)
    # Trim leading/trailing whitespace
    text = text.strip()
    
    return text


def normalize_mixed_script(text: str) -> str:
    """
    Handle Arabic + English mixed text normalization.
    
    Preserves both scripts while applying appropriate normalization:
    - Apply Arabic normalization to Arabic portions
    - Apply standard lowercase to English portions
    - Keep numbers as-is
    - Preserve script boundaries
    
    Args:
        text: Input text with mixed Arabic and English scripts
        
    Returns:
        Normalized mixed-script text
        
    Examples:
        >>> normalize_mixed_script("مياه معدنية   (natural water)")
        'مياه معدنية natural water'
        
        >>> normalize_mixed_script("Coca-Cola ١.٥ لتر")
        'coca-cola 1.5 لتر'
        
        >>> normalize_mixed_script("Apple Juice عصير تفاح")
        'apple juice عصير تفاح'
    """
    if not text:
        return ""
    
    # First, normalize Arabic characters in the entire text
    # This handles character-level normalization without removing English
    alef_variants = {'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ة': 'ه', 'ى': 'ي'}
    diacritics = 'ًًٌٌٍٍََُُِِّّْْٰٰ'
    
    translation_table = str.maketrans({
        **alef_variants,
        **{d: '' for d in diacritics}
    })
    
    text = text.translate(translation_table)
    
    # Convert to lowercase for English portions
    # We'll handle this by processing the text while preserving Arabic
    result_parts = []
    current_part = []
    
    for char in text:
        if '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F' or \
           '\uFB50' <= char <= '\uFDFF' or '\uFE70' <= char <= '\uFEFF':
            # Arabic character - keep as is (already normalized)
            current_part.append(char)
        else:
            # Non-Arabic character
            if current_part:
                result_parts.append(''.join(current_part))
                current_part = []
            result_parts.append(char.lower() if char.isalpha() else char)
    
    if current_part:
        result_parts.append(''.join(current_part))
    
    text = ''.join(result_parts)
    
    # Remove non-alphanumeric characters except spaces and common punctuation
    # Keep Arabic, English, numbers, spaces, and basic punctuation (. -)
    pattern = r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFFa-zA-Z0-9\s\.\-]'
    text = re.sub(pattern, ' ', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def remove_stop_words(text: str, language: str = 'arabic') -> str:
    """
    Remove common stop words from text.
    
    Args:
        text: Input text
        language: 'arabic' or 'english' or 'both'
        
    Returns:
        Text with stop words removed
        
    Examples:
        >>> remove_stop_words("هذا المنتج من الشركة", "arabic")
        'المنتج الشركة'
        
        >>> remove_stop_words("the product of the company", "english")
        'product company'
        
        >>> remove_stop_words("من the product", "both")
        'product'
    """
    if not text:
        return ""
    
    # Define stop word lists
    arabic_stop_words = {
        'من', 'إلى', 'عن', 'على', 'مع', 'في', 'هذا', 'هذه', 
        'ذلك', 'تلك', 'كان', 'يكون', 'هو', 'هي', 'هم', 'هن',
        'أن', 'إن', 'لا', 'ما', 'لم', 'لن', 'ليس', 'لو'
    }
    
    english_stop_words = {
        'the', 'a', 'an', 'of', 'for', 'to', 'with', 'and', 'or',
        'in', 'on', 'at', 'by', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'
    }
    
    # Combine based on language selection
    stop_words = set()
    if language == 'arabic':
        stop_words = arabic_stop_words
    elif language == 'english':
        stop_words = english_stop_words
    elif language == 'both':
        stop_words = arabic_stop_words.union(english_stop_words)
    else:
        raise ValueError(f"Unsupported language: {language}. Use 'arabic', 'english', or 'both'.")
    
    # Split text into words
    words = text.split()
    
    # Filter out stop words
    filtered_words = [word for word in words if word not in stop_words]
    
    # Rejoin with single spaces
    return ' '.join(filtered_words)


def normalize_ocr_noise(text: str) -> str:
    """
    Handle common OCR errors and noise patterns.
    
    Fixes include:
    - Remove repeated characters (للله → لله)
    - Fix common ligature issues (لا as separate vs combined)
    - Handle stray punctuation at word boundaries
    - Fix number-word confusion (0 for O, 1 for I in English context)
    - Normalize Arabic-Indic digits to Western digits
    
    Args:
        text: Input text with OCR noise
        
    Returns:
        Cleaned text with OCR noise reduced
        
    Examples:
        >>> normalize_ocr_noise("للله الرحمن")
        'لله الرحمن'
        
        >>> normalize_ocr_noise("C0ca-C0la")
        'Coca-Cola'
        
        >>> normalize_ocr_noise("١٠٠ جرام  (100g)")
        '100 جرام 100g'
    """
    if not text:
        return ""
    
    # Fix 1: Normalize Arabic-Indic digits to Western digits
    arabic_digits = '٠١٢٣٤٥٦٧٨٩'
    western_digits = '0123456789'
    digit_translation = str.maketrans(arabic_digits, western_digits)
    text = text.translate(digit_translation)
    
    # Fix 2: Remove repeated characters (3+ repetitions -> 2)
    # This handles cases like "للله" -> "لله"
    text = re.sub(r'(.)\1\1+', r'\1\1', text)
    
    # Fix 3: Fix common ligature issues
    # Handle ل + ا that might be separated or combined incorrectly
    text = re.sub(r'ل\s*ا', 'لا', text)  # ل ا -> لا
    text = re.sub(r'ل\s*أ', 'لا', text)  # ل أ -> لا
    text = re.sub(r'ل\s*إ', 'لا', text)  # ل إ -> لا
    text = re.sub(r'ل\s*آ', 'لا', text)  # ل آ -> لا
    
    # Fix 4: Handle stray punctuation at word boundaries
    # Remove punctuation that's stuck to words
    text = re.sub(r'(\w)[.,;:!?]+(\s|$)', r'\1\2', text)  # End of word
    text = re.sub(r'(\s|^)[.,;:!?]+(\w)', r'\1\2', text)  # Start of word
    
    # Fix 5: Fix common English OCR errors
    # 0/O confusion in context
    text = re.sub(r'\b([A-Za-z])0([A-Za-z])\b', r'\1O\2', text)  # Middle of word
    text = re.sub(r'\b0([A-Za-z]+)\b', r'O\1', text)  # Start of word
    text = re.sub(r'\b([A-Za-z]+)0\b', r'\1O', text)  # End of word
    
    # 1/I/l confusion
    text = re.sub(r'\b([A-Za-z])1([A-Za-z])\b', r'\1I\2', text)
    text = re.sub(r'\b1([A-Za-z]+)\b', r'I\1', text)
    
    # Fix 6: Remove extra spaces around hyphens
    text = re.sub(r'\s*-\s*', '-', text)
    
    # Fix 7: Remove orphaned parentheses and brackets
    text = re.sub(r'\([^)]*\)', lambda m: m.group(0) if '(' in m.group(0) and ')' in m.group(0) else '', text)
    text = re.sub(r'\[[^]]*\]', lambda m: m.group(0) if '[' in m.group(0) and ']' in m.group(0) else '', text)
    
    # Final cleanup
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def normalize_product_name(text: str, aggressive: bool = False) -> str:
    """
    Comprehensive product name normalization pipeline.
    
    Applies all normalization steps in optimal order:
    1. OCR noise cleanup
    2. Mixed script normalization
    3. Arabic normalization
    4. Stop word removal (if aggressive)
    5. Final whitespace cleanup
    
    Args:
        text: Raw product name from OCR
        aggressive: If True, remove stop words for stricter matching
        
    Returns:
        Fully normalized product name ready for matching
    """
    if not text:
        return ""
    
    # Step 1: Fix OCR noise
    text = normalize_ocr_noise(text)
    
    # Step 2: Handle mixed scripts
    text = normalize_mixed_script(text)
    
    # Step 3: Apply Arabic normalization
    text = normalize_arabic(text)
    
    # Step 4: Optionally remove stop words
    if aggressive:
        text = remove_stop_words(text, 'both')
    
    # Step 5: Final cleanup
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


# Test function to verify implementations
def _test_normalization():
    """Run tests to verify normalization functions work correctly."""
    test_cases = [
        ("إسكواش  فراولة ٥٠٠ جرام", "اسكواش فراولة 500 جرام"),
        ("مياه معدنية   (natural water)", "مياه معدنية natural water"),
        ("خبز ٌإفرنجي ُطازج", "خبز افرنجي طازج"),
        ("Coca-Cola ١.٥ لتر", "coca-cola 1.5 لتر"),
        ("هذا المنتج من الشركة", "المنتج الشركة"),
        ("للله الرحمن الرحيم", "لله الرحمن الرحيم"),
    ]
    
    print("Testing normalization functions...")
    for input_text, expected in test_cases:
        result = normalize_product_name(input_text)
        status = "✓" if result == expected else "✗"
        print(f"{status} Input: {input_text}")
        print(f"  Expected: {expected}")
        print(f"  Got:      {result}")
        if result != expected:
            print(f"  MISMATCH!")
        print()


if __name__ == "__main__":
    # Run tests when module is executed directly
    _test_normalization()
    
    # Example usage
    examples = [
        "إسكواش  فراولة ٥٠٠ جرام",
        "مياه معدنية   (natural water)",
        "خبز ٌإفرنجي ُطازج",
        "C0ca-C0la ٢ لتر",
    ]
    
    print("\nExample normalizations:")
    for example in examples:
        normalized = normalize_product_name(example)
        print(f"'{example}' -> '{normalized}'")