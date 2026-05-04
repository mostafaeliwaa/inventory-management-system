import io
import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz

from ocr1.Models.product_matcher import normalize_product_name_cached

logger = logging.getLogger(__name__)

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Data classes
@dataclass
class MatchResult:
    original: str
    matched: str
    score: float


# Matcher
class ProductMatcher:

    def __init__(
        self,
        product_database: List[str],
        threshold: float = 0.6,
        semantic_weight: float = 0.6,
        fuzzy_weight: float = 0.4,
    ):
        self.products = product_database
        self.normalized_products = [
            normalize_product_name_cached(p, aggressive=True)
            for p in product_database
        ]
        self.threshold = threshold
        self.semantic_weight = semantic_weight
        self.fuzzy_weight = fuzzy_weight

        # Load model once
        logger.info(f"Loading SentenceTransformer: {_MODEL_NAME}")
        self.model = SentenceTransformer(_MODEL_NAME)

        self.embeddings: Optional[np.ndarray] = None
        self._load_or_compute_embeddings()

    # Embeddings
    def _cache_path(self) -> Optional[Path]:
        """
        Return a writable cache path.
        On Render (read-only filesystem outside /tmp), use /tmp.
        Returns None if caching should be skipped.
        """
        cache_dir = Path(os.getenv("EMBEDDINGS_CACHE_DIR", "/tmp/.product_matcher_cache"))
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            return cache_dir / "embeddings.pkl"
        except OSError:
            logger.warning("Cannot create cache directory — embeddings will not be persisted.")
            return None

    def _load_or_compute_embeddings(self) -> None:
        """Load embeddings from cache or compute them"""
        cache_path = self._cache_path()

        # 1. Try to load from cache
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)

                self.embeddings = data["embeddings"]
                self.normalized_products = data.get(
                    "normalized_products",
                    self.normalized_products
                )

                logger.info(f"Loaded embeddings: {self.embeddings.shape[0]} products")
                return

            except Exception as exc:
                logger.warning(f"Cache corrupted → recomputing: {exc}")

        # 2. Compute embeddings
        logger.info(f"Computing embeddings for {len(self.normalized_products)} products…")

        # Adaptive batch size
        batch_size = min(64, max(16, len(self.normalized_products) // 50))

        embeddings_list = []

        for i in range(0, len(self.normalized_products), batch_size):
            batch = self.normalized_products[i:i + batch_size]

            batch_embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            embeddings_list.append(batch_embeddings)

        self.embeddings = np.vstack(embeddings_list)

        # 3. Save to cache
        if cache_path:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump({
                        "embeddings": self.embeddings,
                        "normalized_products": self.normalized_products
                    }, f)

                logger.info(f"Embeddings cached: {self.embeddings.shape}")

            except Exception as exc:
                logger.warning(f"Cache write failed: {exc}")

    # Scoring helpers 
    def encode_query(self, queries: List[str], normalize_embeddings: bool = True) -> np.ndarray:
        return self.model.encode(queries, normalize_embeddings=normalize_embeddings)

    # Main API 
    def match_batch(self, queries: List[str]) -> List[MatchResult]:
        if not queries:
            return []

        normalized_queries = [
            normalize_product_name_cached(q, aggressive=True) for q in queries
        ]
        query_embeddings = self.encode_query(normalized_queries)
        semantic_scores = np.dot(query_embeddings, self.embeddings.T)  # (Q, P)

        results: List[MatchResult] = []
        top_k = 20

        for orig, norm_q, sem_scores in zip(queries, normalized_queries, semantic_scores):
            top_indices = np.argsort(sem_scores)[-top_k:][::-1]

            best_score, best_idx = 0.0, -1
            for idx in top_indices:
                fuzzy_score = (
                    fuzz.token_sort_ratio(norm_q, self.normalized_products[idx]) / 100.0
                )
                combined = (self.semantic_weight * float(sem_scores[idx])) + (
                    self.fuzzy_weight * fuzzy_score
                )
                if combined > best_score:
                    best_score, best_idx = combined, idx

            if best_score >= self.threshold and best_idx >= 0:
                results.append(MatchResult(orig, self.products[best_idx], best_score))
            else:
                results.append(MatchResult(orig, orig, 0.0))

        return results

    def get_stats(self) -> Dict:
        return {
            "database_size": len(self.products),
            "embeddings_loaded": self.embeddings is not None,
            "semantic_weight": self.semantic_weight,
            "fuzzy_weight": self.fuzzy_weight,
        }