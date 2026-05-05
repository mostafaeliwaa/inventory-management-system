import logging
import warnings
from typing import List, Dict
from dataclasses import dataclass

from rapidfuzz import fuzz, process
from ocr.Models.product_matcher import normalize_product_name_cached

# Optional imports
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_AVAILABLE = True
except ImportError:
    SENTENCE_AVAILABLE = False
    warnings.warn("sentence-transformers not installed → semantic disabled")

logger = logging.getLogger(__name__)


# ======================
# Data Class
# ======================
@dataclass
class MatchResult:
    original: str
    matched: str
    score: float


# ======================
# Product Matcher
# ======================
class ProductMatcher:

    def __init__(
        self,
        product_database: List[str],
        threshold: float = 0.6,
        semantic_weight: float = 0.7,
        fuzzy_weight: float = 0.3,
    ):
        self.products = product_database
        self.normalized_products = [
            normalize_product_name_cached(p, aggressive=True)
            for p in product_database
        ]

        self.threshold = threshold
        self.semantic_weight = semantic_weight
        self.fuzzy_weight = fuzzy_weight

        # Semantic setup (optional)
        self.use_semantic = SENTENCE_AVAILABLE

        if self.use_semantic:
            try:
                self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                self.embeddings = self.model.encode(
                    self.normalized_products,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                logger.info("✅ Semantic matching enabled")
            except Exception as e:
                logger.warning(f"❌ Failed to load semantic model: {e}")
                self.use_semantic = False
        else:
            logger.info("⚠️ Running in fuzzy-only mode")

        logger.info(f"Matcher initialized with {len(self.products)} products")

    # ======================
    # Main Matching
    # ======================
    def match_batch(self, queries: List[str]) -> List[MatchResult]:
        if not queries:
            return []

        results: List[MatchResult] = []

        for query in queries:
            norm_q = normalize_product_name_cached(query, aggressive=True)

            best_score = 0.0
            best_idx = -1

            # Semantic vector
            if self.use_semantic:
                query_vec = self.model.encode(
                    [norm_q],
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )[0]
                semantic_scores = np.dot(self.embeddings, query_vec)
            else:
                semantic_scores = [0.0] * len(self.products)

            # Loop products
            for idx in range(len(self.products)):
                fuzzy_score = (
                    fuzz.token_sort_ratio(norm_q, self.normalized_products[idx]) / 100.0
                )

                semantic_score = semantic_scores[idx] if self.use_semantic else 0.0

                combined = (
                    self.semantic_weight * semantic_score
                    + self.fuzzy_weight * fuzzy_score
                )

                if combined > best_score:
                    best_score = combined
                    best_idx = idx

            # Save result
            if best_score >= self.threshold and best_idx >= 0:
                results.append(MatchResult(query, self.products[best_idx], best_score))
            else:
                results.append(MatchResult(query, query, 0.0))

        return results

    # ======================
    # Single Match
    # ======================
    def match_single(self, query: str) -> MatchResult:
        result = self.match_batch([query])
        return result[0] if result else MatchResult(query, query, 0.0)

    # ======================
    # Top Matches
    # ======================
    def get_top_matches(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict]:

        norm_q = normalize_product_name_cached(query, aggressive=True)

        # Fast fuzzy top-k
        matches = process.extract(
            norm_q,
            self.normalized_products,
            scorer=fuzz.token_sort_ratio,
            limit=top_k
        )

        results = []
        for matched_text, score, idx in matches:
            score = score / 100.0

            if score >= min_score:
                results.append({
                    "name": self.products[idx],
                    "score": score,
                    "index": idx
                })

        return results

    # ======================
    # Stats
    # ======================
    def get_stats(self) -> Dict:
        return {
            "database_size": len(self.products),
            "semantic_enabled": self.use_semantic,
            "threshold": self.threshold,
        }