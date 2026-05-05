import io
import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz

from ocr1.Models.product_matcher import normalize_product_name_cached

logger = logging.getLogger(__name__)

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# ---------------------------------------------------------------------------
# Optional SentenceTransformer import — graceful fallback if not installed
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("SentenceTransformers loaded successfully — semantic matching enabled.")
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "SentenceTransformers is NOT available — "
        "ProductMatcher will fall back to fuzzy-only matching."
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class MatchResult:
    original: str
    matched: str
    score: float


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Decide whether semantic matching is available
        # ------------------------------------------------------------------
        if _SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.info(f"Loading SentenceTransformer: {_MODEL_NAME}")
            self.model = SentenceTransformer(_MODEL_NAME)
            self.use_semantic = True
            logger.info("Semantic matching is ENABLED.")
        else:
            self.model = None
            self.embeddings = None
            self.use_semantic = False
            logger.warning(
                "Semantic model unavailable → using fuzzy matching only. "
                "Install sentence-transformers to enable semantic matching."
            )

        self.embeddings: Optional[np.ndarray] = None
        self._load_or_compute_embeddings()

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
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
        """Load embeddings from cache or compute them. Skipped entirely when semantic is disabled."""
        if not self.use_semantic:
            logger.info("Semantic matching disabled — skipping embedding computation.")
            return

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

                logger.info(f"Loaded embeddings from cache: {self.embeddings.shape[0]} products")
                return

            except Exception as exc:
                logger.warning(f"Cache corrupted → recomputing: {exc}")

        # 2. Compute embeddings
        logger.info(f"Computing embeddings for {len(self.normalized_products)} products…")

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

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------
    def encode_query(self, queries: List[str], normalize_embeddings: bool = True) -> Optional[np.ndarray]:
        """
        Encode queries into dense vectors.
        Returns None (controlled skip) when semantic matching is disabled.
        """
        if not self.use_semantic or self.model is None:
            logger.debug("encode_query called but semantic is disabled — returning None.")
            return None
        return self.model.encode(queries, normalize_embeddings=normalize_embeddings)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def match_batch(self, queries: List[str]) -> List[MatchResult]:
        if not queries:
            return []

        normalized_queries = [
            normalize_product_name_cached(q, aggressive=True) for q in queries
        ]

        # Pre-compute semantic scores only when available
        if self.use_semantic and self.embeddings is not None:
            query_embeddings = self.encode_query(normalized_queries)
            semantic_scores_matrix = np.dot(query_embeddings, self.embeddings.T)  # (Q, P)
            logger.debug("match_batch: using combined semantic + fuzzy scoring.")
        else:
            semantic_scores_matrix = None
            logger.debug("match_batch: semantic unavailable — using fuzzy-only scoring.")

        results: List[MatchResult] = []
        top_k = 20

        for i, (orig, norm_q) in enumerate(zip(queries, normalized_queries)):

            best_score, best_idx = 0.0, -1

            if semantic_scores_matrix is not None:
                # --- Semantic + Fuzzy ---
                sem_scores = semantic_scores_matrix[i]
                top_indices = np.argsort(sem_scores)[-top_k:][::-1]

                for idx in top_indices:
                    fuzzy_score = (
                        fuzz.token_sort_ratio(norm_q, self.normalized_products[idx]) / 100.0
                    )
                    combined = (
                        self.semantic_weight * float(sem_scores[idx])
                        + self.fuzzy_weight * fuzzy_score
                    )
                    if combined > best_score:
                        best_score, best_idx = combined, idx
            else:
                # --- Fuzzy only ---
                for idx, norm_p in enumerate(self.normalized_products):
                    fuzzy_score = fuzz.token_sort_ratio(norm_q, norm_p) / 100.0
                    if fuzzy_score > best_score:
                        best_score, best_idx = fuzzy_score, idx

            if best_score >= self.threshold and best_idx >= 0:
                results.append(MatchResult(orig, self.products[best_idx], best_score))
            else:
                results.append(MatchResult(orig, orig, 0.0))

        return results

    def get_stats(self) -> Dict:
        return {
            "database_size": len(self.products),
            "embeddings_loaded": self.embeddings is not None,
            "semantic_matching_enabled": self.use_semantic,
            "semantic_weight": self.semantic_weight,
            "fuzzy_weight": self.fuzzy_weight,
        }