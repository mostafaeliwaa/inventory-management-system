import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from ocr1.Scripts.supabase_client import get_supabase_client
from ocr1.Models.product_matcher_system import ProductMatcher
from ocr1.Models.product_matcher import normalize_product_name_cached
import pandas as pd
import logging

logger = logging.getLogger(__name__)


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
    """
    Return the top-k candidate matches for *query* against the matcher's database.

    Output dict keys are always:
        name     – matched product name
        score    – combined score (or fuzzy-only when semantic is disabled)
        semantic – semantic component (0.0 when semantic is disabled)
        fuzzy    – fuzzy component
        rank     – 1-based rank
        id       – index in matcher.products

    When ``matcher.use_semantic`` is False the function skips all embedding
    logic entirely and scores solely on fuzzy similarity.
    """
    if not query or not query.strip():
        return []

    normalized_query = normalize_product_name_cached(query, aggressive=True)

    # ------------------------------------------------------------------
    # Candidate filtering (optional helper on matcher)
    # ------------------------------------------------------------------
    if hasattr(matcher, "filter_candidates"):
        candidates = matcher.filter_candidates(
            normalized_query,
            matcher.normalized_products,
            max_candidates=len(matcher.products),
        )
        candidate_indices = [idx for idx, _ in candidates]
    else:
        candidate_indices = list(range(len(matcher.products)))

    if not candidate_indices:
        candidate_indices = list(range(len(matcher.products)))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    use_semantic = getattr(matcher, "use_semantic", False) and matcher.embeddings is not None

    if use_semantic:
        logger.debug("get_top3_matches: semantic + fuzzy scoring.")

        # --- Semantic scores ---
        if hasattr(matcher, "semantic_score"):
            semantic = matcher.semantic_score(normalized_query, candidate_indices)
        else:
            query_embedding = matcher.encode_query(
                [normalized_query], normalize_embeddings=True
            )
            # encode_query returns None when semantic is disabled, but we already
            # checked use_semantic above, so this is safe.
            semantic = [
                (idx, float(np.dot(query_embedding[0], matcher.embeddings[idx])))
                for idx in candidate_indices
            ]

        # --- Fuzzy scores ---
        if hasattr(matcher, "fuzzy_score"):
            fuzzy = matcher.fuzzy_score(normalized_query, candidate_indices)
        else:
            from rapidfuzz import fuzz
            fuzzy = [
                (idx, fuzz.token_sort_ratio(normalized_query, matcher.normalized_products[idx]) / 100.0)
                for idx in candidate_indices
            ]

        semantic_dict = dict(semantic)
        fuzzy_dict = dict(fuzzy)

        combined = []
        for idx in candidate_indices:
            s = semantic_dict.get(idx, 0.0)
            f = fuzzy_dict.get(idx, 0.0)
            score = (matcher.semantic_weight * s) + (matcher.fuzzy_weight * f)
            if score >= min_score:
                combined.append((idx, score, s, f))

    else:
        # --- Fuzzy only ---
        logger.debug(
            "get_top3_matches: semantic disabled (use_semantic=%s, embeddings=%s) — "
            "fuzzy-only scoring.",
            getattr(matcher, "use_semantic", False),
            matcher.embeddings is not None,
        )
        from rapidfuzz import fuzz

        combined = []
        for idx in candidate_indices:
            f = fuzz.token_sort_ratio(normalized_query, matcher.normalized_products[idx]) / 100.0
            score = f  # semantic contribution = 0
            if score >= min_score:
                combined.append((idx, score, 0.0, f))

    # ------------------------------------------------------------------
    # Sort and build output — API shape is always the same
    # ------------------------------------------------------------------
    combined.sort(key=lambda x: x[1], reverse=True)

    results: List[Dict[str, Any]] = []
    for idx, score, s, f in combined[:top_k]:
        product_name = matcher.products[idx] if idx < len(matcher.products) else f"unknown_{idx}"
        results.append({
            "name": product_name,
            "score": float(score),
            "semantic": float(s),   # 0.0 when semantic is disabled
            "fuzzy": float(f),
            "rank": len(results) + 1,
            "id": idx,
        })

    return results