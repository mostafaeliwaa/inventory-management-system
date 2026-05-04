import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from ocr1.Scripts.supabase_client import get_supabase_client
from ocr1.Models.product_matcher_system import ProductMatcher
from ocr1.Models.product_matcher import normalize_product_name_cached
import pandas as pd

def load_product_database_from_excel(
    file_path: str,
    product_column: str = "product_name"
) -> Tuple[List[str], Dict[int, str]]:
    
    try:
        df = pd.read_csv(file_path)
        if product_column not in df.columns:
            raise ValueError(f"Column '{product_column}' not found in CSV file")
        product_names = (
            df[product_column]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )
        product_lookup = {idx: name for idx, name in enumerate(product_names)}

        print(f"✅ Loaded {len(product_names)} products from CSV file '{file_path}'")
        if product_names:
            print(f"📋 Sample products: {product_names[:5]}")

        return product_names, product_lookup

    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return [], {}

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
    
    print(f"✅ Loaded {len(product_names)} products from Supabase table '{table_name}'")
    if product_names:
        print(f"📋 Sample products: {product_names[:5]}")
    
    return product_names, product_lookup


def get_top3_matches(
    matcher: ProductMatcher, 
    query: str, 
    min_score: float = 0.0,
    top_k: int = 3
) -> List[Dict[str, Any]]:

    if not query or not query.strip():
        return []
    
    normalized_query = normalize_product_name_cached(query, aggressive=True)

    if hasattr(matcher, 'filter_candidates'):
        candidates = matcher.filter_candidates(
            normalized_query,
            matcher.normalized_products,
            max_candidates=len(matcher.products)
        )
        candidate_indices = [idx for idx, _ in candidates]
    else:
        candidate_indices = list(range(len(matcher.products)))
    
    if not candidate_indices:
        candidate_indices = list(range(len(matcher.products)))
    
    if hasattr(matcher, 'semantic_score'):
        semantic = matcher.semantic_score(normalized_query, candidate_indices)
    else:
        semantic = []
        query_embedding = matcher.encode_query([normalized_query], normalize_embeddings=True)[0]
        for idx in candidate_indices:
            score = float(np.dot(query_embedding, matcher.embeddings[idx]))
            semantic.append((idx, score))

    if hasattr(matcher, 'fuzzy_score'):
        fuzzy = matcher.fuzzy_score(normalized_query, candidate_indices)
    else:
        from rapidfuzz import fuzz
        fuzzy = []
        for idx in candidate_indices:
            score = fuzz.token_sort_ratio(normalized_query, matcher.normalized_products[idx]) / 100.0
            fuzzy.append((idx, score))
    
    semantic_dict = dict(semantic)
    fuzzy_dict = dict(fuzzy)
    
    combined = []
    
    for idx in candidate_indices:
        s = semantic_dict.get(idx, 0.0)
        f = fuzzy_dict.get(idx, 0.0)
        score = (matcher.semantic_weight * s) + (matcher.fuzzy_weight * f)
        
        if score >= min_score:
            combined.append((idx, score, s, f))
    
    combined.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    
    for idx, score, s, f in combined[:top_k]:
        if idx < len(matcher.products):
            product_name = matcher.products[idx]
        else:
            product_name = f"unknown_{idx}"
        
        results.append({
            "name": product_name,
            "score": float(score),
            "semantic": float(s),
            "fuzzy": float(f),
            "rank": len(results) + 1,
            "id": idx
        })
    
    return results