import csv
import json
from pathlib import Path
from datetime import datetime

from product_matcher_system import ProductMatcher
from ocr.product_matcher import normalize_product_name

MASTER_CSV = Path('cleaning/output/master_dataframe.csv')


# -----------------------------
# Load product database as a LIST of product NAMES (strings)
# -----------------------------
def load_product_database():
    """
    Loads product database and returns:
        - product_names: List[str] (actual product names for the matcher)
        - product_lookup: Dict[int, str] (mapping from index to product name)
    """
    product_names = []
    product_lookup = {}

    if not MASTER_CSV.exists():
        print(f"Warning: CSV not found at {MASTER_CSV}")
        return product_names, product_lookup

    with open(MASTER_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        for row_id, row in enumerate(reader):
            # Try to find the column that contains product names
            name = None
            
            # Common column names for product name column
            possible_columns = ['اسم الصنف', 'product_name', 'name', 'Product', 'الصنف', 'اسم المنتج']
            
            for col in possible_columns:
                if col in row and row[col] and str(row[col]).strip():
                    name = str(row[col]).strip()
                    break
            
            # Fallback: use first non-empty column if no match found
            if not name:
                for col in row:
                    val = row[col]
                    if val and str(val).strip() and not val.isdigit():  # Avoid numeric IDs as names
                        name = str(val).strip()
                        break
            
            if name and name.lower() != "nan":
                product_names.append(name)
                product_lookup[row_id] = name
            elif 'id' in row and row['id']:
                # If only ID exists but no name, use ID as fallback
                product_names.append(str(row['id']))
                product_lookup[row_id] = str(row['id'])

    print(f"Loaded {len(product_names)} products from CSV")
    return product_names, product_lookup


# -----------------------------
# Init matcher once
# -----------------------------
def build_matcher():
    product_names, _ = load_product_database()

    if not product_names:
        raise ValueError("No products loaded from master CSV")

    matcher = ProductMatcher(
        product_database=product_names,  # Now this is a list of actual names!
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        semantic_weight=0.7,
        fuzzy_weight=0.3,
        match_threshold=0.6,
        use_faiss=False
    )

    matcher.compute_embeddings()
    return matcher


# -----------------------------
# ALL matches - RETURNS ALL PRODUCTS (no limit)
# -----------------------------
def get_all_matches(matcher, query, min_score=0.0):
    """
    Get ALL matches for a query, sorted by score descending.
    
    Args:
        matcher: ProductMatcher instance
        query: Product name to match
        min_score: Minimum score threshold (default 0.0 = return all)
    
    Returns:
        List of all matches with scores
    """
    normalized_query = normalize_product_name(query, aggressive=True)

    # Get candidates (all products, no filtering limit)
    candidates = matcher.filter_candidates(
        normalized_query,
        matcher.normalized_database,
        max_candidates=len(matcher.product_database)  # Get ALL candidates
    )

    candidate_indices = [idx for idx, _ in candidates]

    if not candidate_indices:
        candidate_indices = list(range(len(matcher.product_database)))

    # Get semantic scores for all candidates
    semantic = matcher.semantic_score(normalized_query, candidate_indices)
    fuzzy = matcher.fuzzy_score(normalized_query, candidate_indices)

    semantic_dict = dict(semantic)
    fuzzy_dict = dict(fuzzy)

    combined = []

    for idx in candidate_indices:
        s = semantic_dict.get(idx, 0.0)
        f = fuzzy_dict.get(idx, 0.0)

        score = matcher.semantic_weight * s + matcher.fuzzy_weight * f
        
        # Filter by minimum score if specified
        if score >= min_score:
            combined.append((idx, score, s, f))

    # Sort by combined score descending
    combined.sort(key=lambda x: x[1], reverse=True)

    results = []

    for idx, score, s, f in combined:
        # Get the ACTUAL product name from the database
        if idx < len(matcher.product_database):
            product_name = matcher.product_database[idx]
        else:
            product_name = f"unknown_{idx}"
        
        results.append({
            "name": product_name,  # This is the actual product name
            "score": float(score),
            "semantic": float(s),
            "fuzzy": float(f),
            "rank": len(results) + 1,  # Add rank position
            "id": idx
        })

    return results


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def run_fallback_matching(extraction: dict, matcher, output_dir: Path, min_score=0.0):
    """
    Run matching for all items and return ALL matches (no top-k limit).
    
    Args:
        extraction: Extracted invoice data
        matcher: ProductMatcher instance
        output_dir: Output directory for results
        min_score: Minimum score threshold (default 0.0 = return all)
    """
    items = extraction.get("items", []) if isinstance(extraction, dict) else []

    summary = []

    for it in items:
        name = it.get("product_name") if isinstance(it, dict) else str(it)

        if not name:
            continue

        # Get ALL matches (no limit)
        all_matches = get_all_matches(matcher, name, min_score=min_score)

        summary.append({
            "query": name,
            "matched": all_matches,  # Now contains ALL matches
            "total_matches": len(all_matches),  # Total count
            "top_score": float(all_matches[0]["score"]) if all_matches else 0.0
        })

    result = {
        "generated_at": datetime.now().isoformat(),
        "summary": summary,
        "metadata": {
            "total_queries": len(summary),
            "min_score_threshold": min_score,
            "total_products_in_db": len(matcher.product_database)
        }
    }

    output_path = output_dir / "matches_summary_fallback.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result, output_path


# -----------------------------
# OPTIONAL: file runner (for testing only)
# -----------------------------
def run_from_files(input_file: Path, output_file: Path):
    matcher = build_matcher()

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    extraction = data.get('extracted') or data

    # Need to provide output_dir
    result, _ = run_fallback_matching(extraction, matcher, output_file.parent)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Saved all matches to", output_file)