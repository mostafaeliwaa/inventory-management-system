import re
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import time
from collections import defaultdict
import warnings

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn("sentence-transformers not installed. Install with: pip install sentence-transformers")

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    warnings.warn("rapidfuzz not installed. Install with: pip install rapidfuzz")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn("FAISS not installed. Install with: pip install faiss-cpu")

try:
    from .product_matcher import normalize_product_name, normalize_arabic, normalize_mixed_script
except ImportError:
    def normalize_product_name(text: str, aggressive: bool = False) -> str:
        """Fallback normalization if product_matcher not available."""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text.strip())
        return text.lower()
    
    def normalize_arabic(text: str) -> str:
        """Fallback Arabic normalization."""
        return text.strip().lower() if text else ""
    
    def normalize_mixed_script(text: str) -> str:
        """Fallback mixed script normalization."""
        return text.strip().lower() if text else ""


@dataclass
class MatchResult:
    """Result of product matching operation."""
    original: str
    matched: str
    score: float
    method_breakdown: Dict[str, float]
    candidate_count: int = 0
    normalized_query: str = ""
    is_correction: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['score'] = float(result['score'])
        for key in result['method_breakdown']:
            result['method_breakdown'][key] = float(result['method_breakdown'][key])
        return result


class ProductMatcher:
    """
    High-accuracy product matching system for OCR-extracted invoice items.
    
    Features:
    1. Text normalization (Arabic-aware)
    2. Candidate filtering (fast preselection)
    3. Semantic similarity (embeddings)
    4. Fuzzy matching
    5. Score fusion
    6. Learning from user corrections
    """
    
    def __init__(
        self,
        product_database: List[str],
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        semantic_weight: float = 0.7,
        fuzzy_weight: float = 0.3,
        match_threshold: float = 0.6,
        cache_dir: Optional[str] = None,
        use_faiss: bool = False
    ):
        """
        Initialize the product matcher.
        
        Args:
            product_database: List of product names in the database
            model_name: Sentence transformer model name
            semantic_weight: Weight for semantic similarity score (0.0-1.0)
            fuzzy_weight: Weight for fuzzy matching score (0.0-1.0)
            match_threshold: Minimum score to return a match (otherwise return original)
            cache_dir: Directory to cache embeddings and models
            use_faiss: Use FAISS for fast similarity search (for large databases)
        """
        self.product_database = product_database
        self.normalized_database = [normalize_product_name(p, aggressive=True) for p in product_database]
        self.model_name = model_name
        self.semantic_weight = semantic_weight
        self.fuzzy_weight = fuzzy_weight
        self.match_threshold = match_threshold
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".") / ".product_matcher_cache"
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        
        self.cache_dir.mkdir(exist_ok=True)
        
        self.correction_memory: Dict[str, str] = {}
        self.correction_file = self.cache_dir / "corrections.json"
        self._load_corrections()
        
        self.embeddings_cache_file = self.cache_dir / f"embeddings_{self._get_db_hash()}.pkl"
        self.embeddings = None
        self.faiss_index = None
        self.model = None
        
        self.stats = {
            "total_queries": 0,
            "matches_found": 0,
            "corrections_applied": 0,
            "avg_processing_time": 0.0
        }
    
    def _get_db_hash(self) -> str:
        """Generate hash of the database for caching."""
        db_str = "||".join(sorted(self.normalized_database))
        return hashlib.md5(db_str.encode()).hexdigest()[:16]
    
    def _load_corrections(self) -> None:
        """Load user corrections from file."""
        if self.correction_file.exists():
            try:
                with open(self.correction_file, 'r', encoding='utf-8') as f:
                    self.correction_memory = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.correction_memory = {}
    
    def _save_corrections(self) -> None:
        """Save user corrections to file."""
        try:
            with open(self.correction_file, 'w', encoding='utf-8') as f:
                json.dump(self.correction_memory, f, ensure_ascii=False, indent=2)
        except IOError:
            warnings.warn("Failed to save corrections to file")
    
    def add_correction(self, wrong_text: str, correct_text: str) -> None:
        """
        Add a user correction to memory.
        
        Args:
            wrong_text: The text that was incorrectly matched
            correct_text: The correct product name
        """
        normalized_wrong = normalize_product_name(wrong_text, aggressive=True)
        normalized_correct = normalize_product_name(correct_text, aggressive=True)
        
        self.correction_memory[normalized_wrong] = normalized_correct
        self._save_corrections()
        self.stats["corrections_applied"] += 1
    
    def _apply_user_memory(self, query: str) -> Optional[str]:
        """
        Apply user correction memory before matching.
        
        Args:
            query: Normalized query text
            
        Returns:
            Corrected text if found in memory, None otherwise
        """
        normalized_query = normalize_product_name(query, aggressive=True)
        return self.correction_memory.get(normalized_query)
    
    def filter_candidates(self, query: str, db: List[str], max_candidates: int = 50) -> List[Tuple[int, str]]:
        """
        Fast candidate filtering using keyword overlap.
        
        Args:
            query: Normalized query text
            db: List of normalized database entries
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of (index, product) tuples for candidates
        """
        if not query:
            return []
        
        query_words = set(query.split())
        if not query_words:
            return []
        
        candidates = []
        
        for idx, product in enumerate(db):
            if not product:
                continue
            
            product_words = set(product.split())
            if not product_words:
                continue
            
            overlap = len(query_words.intersection(product_words))
            total_unique = len(query_words.union(product_words))
            
            if total_unique == 0:
                continue
            
            jaccard_similarity = overlap / total_unique
            
            substring_match = any(qw in product for qw in query_words if len(qw) > 2) or \
                             any(pw in query for pw in product_words if len(pw) > 2)
            
            if jaccard_similarity > 0.1 or substring_match:
                candidates.append((idx, product, jaccard_similarity))
        
        candidates.sort(key=lambda x: x[2], reverse=True)
        return [(idx, product) for idx, product, _ in candidates[:max_candidates]]
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        if self.model is not None:
            return
            
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for semantic matching. "
                "Install with: pip install sentence-transformers"
            )
        
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def compute_embeddings(self, force_recompute: bool = False) -> np.ndarray:
        """
        Compute or load embeddings for the product database.
        
        Args:
            force_recompute: Force recomputation even if cache exists
            
        Returns:
            numpy array of embeddings
        """
        if self.embeddings is not None and not force_recompute:
            return self.embeddings
        
        if not force_recompute and self.embeddings_cache_file.exists():
            try:
                with open(self.embeddings_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.embeddings = cache_data['embeddings']
                    print(f"Loaded embeddings from cache: {self.embeddings.shape}")
                    return self.embeddings
            except (pickle.PickleError, IOError, KeyError):
                pass 
        
        self._load_model()
        
        print(f"Computing embeddings for {len(self.normalized_database)} products...")
        start_time = time.time()
        
        batch_size = 64
        all_embeddings = []
        
        for i in range(0, len(self.normalized_database), batch_size):
            batch = self.normalized_database[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            all_embeddings.append(batch_embeddings)
        
        self.embeddings = np.vstack(all_embeddings)
        
        if self.use_faiss:
            dimension = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(self.embeddings.astype('float32'))

        cache_data = {
            'embeddings': self.embeddings,
            'database_hash': self._get_db_hash(),
            'model_name': self.model_name,
            'timestamp': time.time()
        }
        
        try:
            with open(self.embeddings_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except IOError:
            warnings.warn("Failed to cache embeddings")
        
        elapsed = time.time() - start_time
        print(f"Embeddings computed in {elapsed:.2f}s: {self.embeddings.shape}")
        
        return self.embeddings
    
    def semantic_score(self, query: str, candidate_indices: List[int] = None) -> List[Tuple[int, float]]:
        """
        Compute semantic similarity scores between query and database products.
        
        Args:
            query: Normalized query text
            candidate_indices: Optional list of candidate indices to score
            
        Returns:
            List of (index, score) tuples sorted by score descending
        """
        self._load_model()
        self.compute_embeddings()
        
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        
        if candidate_indices is None:
            if self.use_faiss and self.faiss_index is not None:
                query_embedding_32 = query_embedding.astype('float32').reshape(1, -1)
                scores, indices = self.faiss_index.search(query_embedding_32, len(self.product_database))
                results = [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
                results.sort(key=lambda x: x[1], reverse=True)
                return results
            else:
                scores = np.dot(self.embeddings, query_embedding)
                results = [(i, float(score)) for i, score in enumerate(scores)]
                results.sort(key=lambda x: x[1], reverse=True)
                return results
        else:
 
            results = []
            for idx in candidate_indices:
                if 0 <= idx < len(self.embeddings):
                    score = float(np.dot(self.embeddings[idx], query_embedding))
                    results.append((idx, score))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results
    
    def fuzzy_score(self, query: str, candidate_indices: List[int] = None) -> List[Tuple[int, float]]:
        """
        Compute fuzzy matching scores between query and database products.
        
        Args:
            query: Normalized query text
            candidate_indices: Optional list of candidate indices to score
            
        Returns:
            List of (index, score) tuples sorted by score descending
        """
        if not RAPIDFUZZ_AVAILABLE:
            results = []
            targets = self.normalized_database if candidate_indices is None else \
                     [self.normalized_database[i] for i in candidate_indices]
            
            for i, target in enumerate(targets):
                if not target:
                    score = 0.0
                else:

                    query_words = set(query.split())
                    target_words = set(target.split())
                    if not query_words or not target_words:
                        score = 0.0
                    else:
                        intersection = len(query_words.intersection(target_words))
                        union = len(query_words.union(target_words))
                        score = intersection / union if union > 0 else 0.0
                
                actual_idx = i if candidate_indices is None else candidate_indices[i]
                results.append((actual_idx, score))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results
        
        results = []
        
        if candidate_indices is None:
            targets = self.normalized_database
            indices = range(len(targets))
        else:
            targets = [self.normalized_database[i] for i in candidate_indices]
            indices = candidate_indices
        
        for idx, target in zip(indices, targets):
            if not target:
                score = 0.0
            else:
                score = fuzz.token_sort_ratio(query, target) / 100.0
            
            results.append((idx, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def hybrid_match(self, query: str, aggressive_normalization: bool = True) -> MatchResult:
        """
        Main matching function using hybrid approach.
        
        Args:
            query: Original OCR-extracted product name
            aggressive_normalization: Use aggressive normalization (remove stop words)
            
        Returns:
            MatchResult object with match details
        """
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        normalized_query = normalize_product_name(query, aggressive=aggressive_normalization)
        correction = self._apply_user_memory(query)
        
        if correction:
            try:
                idx = self.normalized_database.index(correction)
                matched_product = self.product_database[idx]
                
                result = MatchResult(
                    original=query,
                    matched=matched_product,
                    score=1.0,
                    method_breakdown={"semantic": 1.0, "fuzzy": 1.0, "correction": 1.0},
                    candidate_count=1,
                    normalized_query=normalized_query,
                    is_correction=True
                )
                
                self.stats["matches_found"] += 1
                return result
            except ValueError:
                pass  
        
        candidates = self.filter_candidates(normalized_query, self.normalized_database)
        candidate_indices = [idx for idx, _ in candidates]
        
        if not candidate_indices:
            candidate_indices = list(range(len(self.product_database)))
        
        semantic_results = self.semantic_score(normalized_query, candidate_indices)
        fuzzy_results = self.fuzzy_score(normalized_query, candidate_indices)
        semantic_scores = {idx: score for idx, score in semantic_results}
        fuzzy_scores = {idx: score for idx, score in fuzzy_results}
        combined_scores = []
        for idx in candidate_indices:
            semantic = semantic_scores.get(idx, 0.0)
            fuzzy = fuzzy_scores.get(idx, 0.0)
            combined = (self.semantic_weight * semantic) + (self.fuzzy_weight * fuzzy)
            combined_scores.append((idx, combined, semantic, fuzzy))
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        if combined_scores:
            best_idx, best_score, best_semantic, best_fuzzy = combined_scores[0]
            matched_product = self.product_database[best_idx]
            if best_score >= self.match_threshold:
                result = MatchResult(
                    original=query,
                    matched=matched_product,
                    score=best_score,
                    method_breakdown={
                        "semantic": best_semantic,
                        "fuzzy": best_fuzzy,
                        "combined": best_score
                    },
                    candidate_count=len(candidate_indices),
                    normalized_query=normalized_query,
                    is_correction=False
                )
                self.stats["matches_found"] += 1
            else:
                result = MatchResult(
                    original=query,
                    matched=query, 
                    score=best_score,
                    method_breakdown={
                        "semantic": best_semantic,
                        "fuzzy": best_fuzzy,
                        "combined": best_score
                    },
                    candidate_count=len(candidate_indices),
                    normalized_query=normalized_query,
                    is_correction=False
                )
        else:
            result = MatchResult(
                original=query,
                matched=query,
                score=0.0,
                method_breakdown={"semantic": 0.0, "fuzzy": 0.0, "combined": 0.0},
                candidate_count=0,
                normalized_query=normalized_query,
                is_correction=False
            )
            
        elapsed_time = time.time() - start_time
        total_time = self.stats["avg_processing_time"] * (self.stats["total_queries"] - 1)
        self.stats["avg_processing_time"] = (total_time + elapsed_time) / self.stats["total_queries"]
        
        return result
    
    def batch_match(self, queries: List[str], aggressive_normalization: bool = True) -> List[MatchResult]:
        """
        Match multiple queries efficiently.
        
        Args:
            queries: List of OCR-extracted product names
            aggressive_normalization: Use aggressive normalization
            
        Returns:
            List of MatchResult objects
        """
        results = []
        for query in queries:
            result = self.hybrid_match(query, aggressive_normalization)
            results.append(result)
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        stats = self.stats.copy()
        stats["correction_memory_size"] = len(self.correction_memory)
        stats["database_size"] = len(self.product_database)
        stats["cache_dir"] = str(self.cache_dir)
        return stats
    
    def clear_cache(self) -> None:
        """Clear embeddings cache."""
        if self.embeddings_cache_file.exists():
            self.embeddings_cache_file.unlink()
        self.embeddings = None
        self.faiss_index = None
    
    def export_corrections(self, filepath: str) -> None:
        """Export corrections to a file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.correction_memory, f, ensure_ascii=False, indent=2)
    
    def import_corrections(self, filepath: str) -> None:
        """Import corrections from a file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            imported = json.load(f)
        self.correction_memory.update(imported)
        self._save_corrections()
