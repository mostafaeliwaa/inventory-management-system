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
# top-k matching - RETURNS ACTUAL PRODUCT NAMES
# -----------------------------
def get_top_matches(matcher, query, top_k=10):
    normalized_query = normalize_product_name(query, aggressive=True)

    candidates = matcher.filter_candidates(
        normalized_query,
        matcher.normalized_database
    )

    candidate_indices = [idx for idx, _ in candidates]

    if not candidate_indices:
        candidate_indices = list(range(len(matcher.product_database)))

    semantic = matcher.semantic_score(normalized_query, candidate_indices)
    fuzzy = matcher.fuzzy_score(normalized_query, candidate_indices)

    semantic_dict = dict(semantic)
    fuzzy_dict = dict(fuzzy)

    combined = []

    for idx in candidate_indices:
        s = semantic_dict.get(idx, 0.0)
        f = fuzzy_dict.get(idx, 0.0)

        score = matcher.semantic_weight * s + matcher.fuzzy_weight * f

        combined.append((idx, score, s, f))

    combined.sort(key=lambda x: x[1], reverse=True)

    results = []

    for idx, score, s, f in combined[:top_k]:
        # IMPORTANT: Get the ACTUAL product name from the database
        if idx < len(matcher.product_database):
            product_name = matcher.product_database[idx]
        else:
            product_name = f"unknown_{idx}"
        
        results.append({
            "name": product_name,  # Now this is the actual product name, not an ID!
            "score": float(score),
            "semantic": float(s),
            "fuzzy": float(f),
            "id": idx
        })

    return results


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def run_fallback_matching(extraction: dict, matcher, output_dir: Path, top_k=10):
    items = extraction.get("items", []) if isinstance(extraction, dict) else []

    summary = []

    for it in items:
        name = it.get("product_name") if isinstance(it, dict) else str(it)

        if not name:
            continue

        top = get_top_matches(matcher, name, top_k=top_k)

        summary.append({
            "query": name,
            "matched": top,
            "top_score": float(top[0]["score"]) if top else 0.0
        })

    result = {
        "generated_at": datetime.now().isoformat(),
        "summary": summary
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

    print("Saved fallback matches to", output_file)