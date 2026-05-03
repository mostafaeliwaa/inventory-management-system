import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from .supabase_client import get_supabase_client
from ocr.Models.product_matcher_system import ProductMatcher
from ocr.Models.product_matcher import normalize_product_name_cached
from rapidfuzz import fuzz

def load_product_database_from_supabase(
    table_name: str = "products", 
    product_column: str = "product_name"
) -> Tuple[List[str], Dict[int, str]]:

    supabase = get_supabase_client()
    
    product_names = supabase.get_all_products(
        table_name=table_name,
        product_name_column=product_column
    )
    
    product_lookup = {idx: name for idx, name in enumerate(product_names)}
    
    print(f"[SUPABASE] Loaded {len(product_names)} products from Supabase table '{table_name}'")
    if product_names:
        print(f"[SUPABASE] Sample products: {product_names[:5]}")
    
    return product_names, product_lookup


def get_top10_matches(
    matcher: ProductMatcher,
    query: str,
    min_score: float = 0.0,
    top_k: int = 5
) -> List[Dict[str, Any]]:

    if not query or not query.strip():
        return []
    
    normalized_query = normalize_product_name_cached(query, aggressive=True)
    
    # Check if matcher has the new get_top_matches method
    if hasattr(matcher, 'get_top_matches'):
        # Use the new simplified API
        matches = matcher.get_top_matches(query, top_k=top_k, min_score=min_score)
        
        results = []
        for i, match in enumerate(matches):
            results.append({
                "name": match["name"],
                "score": float(match["score"]),
                "semantic": 0.0,  # Not available in rapidfuzz-only version
                "fuzzy": float(match["score"]),  # All score is fuzzy now
                "rank": i + 1,
                "id": match.get("index", i)
            })
        return results
    
    # Fallback for old matcher (should not happen with our new implementation)
    
    
    candidate_indices = list(range(len(matcher.products)))
    fuzzy_scores = []
    
    for idx in candidate_indices:
        score = fuzz.token_sort_ratio(normalized_query, matcher.normalized_products[idx]) / 100.0
        if score >= min_score:
            fuzzy_scores.append((idx, score))
    
    fuzzy_scores.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for i, (idx, score) in enumerate(fuzzy_scores[:top_k]):
        if idx < len(matcher.products):
            product_name = matcher.products[idx]
        else:
            product_name = f"unknown_{idx}"
        
        results.append({
            "name": product_name,
            "score": float(score),
            "semantic": 0.0,
            "fuzzy": float(score),
            "rank": i + 1,
            "id": idx
        })
    
    return results