import re
from functools import lru_cache

#  Shared constants (defined once) 

_ARABIC_CHAR_MAP = {'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ة': 'ه', 'ى': 'ي'}
_DIACRITICS = 'ًٌٍَُِّْٰ'
_TRANSLATION_TABLE = str.maketrans({**_ARABIC_CHAR_MAP, **{d: '' for d in _DIACRITICS}})

_ARABIC_STOP_WORDS = {
    'من', 'إلى', 'عن', 'على', 'مع', 'في', 'هذا', 'هذه',
    'ذلك', 'تلك', 'كان', 'يكون', 'هو', 'هي', 'هم', 'هن',
    'أن', 'إن', 'لا', 'ما', 'لم', 'لن', 'ليس', 'لو'
}
_ENGLISH_STOP_WORDS = {
    'the', 'a', 'an', 'of', 'for', 'to', 'with', 'and', 'or',
    'in', 'on', 'at', 'by', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'
}
_ALL_STOP_WORDS = _ARABIC_STOP_WORDS | _ENGLISH_STOP_WORDS

#  Core helpers

def _is_arabic(char: str) -> bool:
    return '\u0600' <= char <= '\u06FF'


def normalize_ocr_noise(text: str) -> str:
    if not text:
        return ""
    text = text.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789'))
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'ل\s*[اأإآ]', 'لا', text)
    text = re.sub(r'(\w)[.,;:!?]+(\s|$)', r'\1\2', text)
    text = re.sub(r'(\s|^)[.,;:!?]+(\w)', r'\1\2', text)
    text = re.sub(r'\b([A-Za-z])0([A-Za-z])\b', r'\1O\2', text)
    text = re.sub(r'\b([A-Za-z])1([A-Za-z])\b', r'\1I\2', text)
    text = re.sub(r'\s*-\s*', '-', text)
    return re.sub(r'\s+', ' ', text).strip()


def normalize_mixed_script(text: str) -> str:
    """Normalize Arabic variants and lowercase Latin — single shared implementation."""
    if not text:
        return ""
    text = text.translate(_TRANSLATION_TABLE)
    result, arabic_buffer = [], []
    for char in text:
        if _is_arabic(char):
            arabic_buffer.append(char)
        else:
            if arabic_buffer:
                result.append(''.join(arabic_buffer))
                arabic_buffer = []
            result.append(char.lower() if char.isalpha() else char)
    if arabic_buffer:
        result.append(''.join(arabic_buffer))
    text = ''.join(result)
    text = re.sub(r'[^\u0600-\u06FFa-zA-Z0-9\s.\-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def remove_stop_words(text: str, language: str = "both") -> str:
    if not text:
        return ""
    stop_words = {
        "arabic": _ARABIC_STOP_WORDS,
        "english": _ENGLISH_STOP_WORDS,
        "both": _ALL_STOP_WORDS,
    }.get(language)
    if stop_words is None:
        raise ValueError("language must be: arabic / english / both")
    return " ".join(w for w in text.split() if w not in stop_words)


#Public API

@lru_cache(maxsize=10_000)
def normalize_product_name_cached(text: str, aggressive: bool = False) -> str:
    if not text:
        return ""
    text = normalize_ocr_noise(text)
    text = normalize_mixed_script(text)
    if aggressive:
        text = remove_stop_words(text, language="both")
    return " ".join(text.split())


def normalize_product_name(text: str) -> str:
    """Alias for normalize_product_name_cached with default parameters."""
    return normalize_product_name_cached(text, aggressive=False)