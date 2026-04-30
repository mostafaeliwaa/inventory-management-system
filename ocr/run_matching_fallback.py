import csv
import json
from pathlib import Path
from datetime import datetime
import os
import sys

# Add parent directory to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from product_matcher_system import ProductMatcher
from ocr.product_matcher import normalize_product_name
from ocr.supabase_client import get_supabase_client


# -----------------------------
# Load product database as a LIST of product NAMES (strings)
# -----------------------------
def load_product_database():
    """
    Load product database from Supabase instead of CSV.
    
    Args:
        table_name: Name of the table in Supabase (default: "products")
        product_column: Column name containing product names (default: "name")
    
    Returns:
        - product_names: List[str] (actual product names for the matcher)
        - product_lookup: Dict[int, str] (mapping from index to product name)
    """
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Get all products
        product_names = supabase.get_all_products(
            table_name=table_name,
            product_name_column=product_column
        )
        
        # Create lookup dictionary
        product_lookup = {idx: name for idx, name in enumerate(product_names)}
        
        print(f"✅ Loaded {len(product_names)} products from Supabase table '{table_name}'")
        
        # Optional: Print sample products for verification
        if product_names:
            print(f"📋 Sample products: {product_names[:5]}")
        
        return product_names, product_lookup
        
    except Exception as e:
        print(f"❌ Error loading products from Supabase: {e}")
        print("⚠️ Falling back to CSV file...")
        return load_product_database_from_csv()


# -----------------------------
# Fallback to CSV if Supabase fails
# -----------------------------
def load_product_database_from_csv():
    """Fallback method to load products from CSV file"""
    master_csv = REPO_ROOT / 'cleaning' / 'output' / 'master_dataframe.csv'
    product_names = []
    product_lookup = {}

    if not master_csv.exists():
        print(f"⚠️ CSV not found at {master_csv}")
        return product_names, product_lookup

    with open(master_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        for row_id, row in enumerate(reader):
            name = None
            
            possible_columns = ['اسم الصنف', 'product_name', 'name', 'Product', 'الصنف', 'اسم المنتج']
            
            for col in possible_columns:
                if col in row and row[col] and str(row[col]).strip():
                    name = str(row[col]).strip()
                    break
            if not name:
                for col in row:
                    val = row[col]
                    if val and str(val).strip() and not val.isdigit():
                        name = str(val).strip()
                        break
            
            if name and name.lower() != "nan":
                product_names.append(name)
                product_lookup[row_id] = name
            elif 'id' in row and row['id']:
                product_names.append(str(row['id']))
                product_lookup[row_id] = str(row['id'])

    print(f"📦 Loaded {len(product_names)} products from CSV (fallback)")
    return product_names, product_lookup

# Init matcher once
# -----------------------------
def build_matcher():
    product_names, _ = load_product_database()

    if not product_names:
        raise ValueError("No products loaded from database")

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

    # Get candidates (all products, no filtering limit)
    candidates = matcher.filter_candidates(
        normalized_query,
        matcher.normalized_database,
        max_candidates=len(matcher.product_database)
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
        
        if score >= min_score:
            combined.append((idx, score, s, f))

    combined.sort(key=lambda x: x[1], reverse=True)

    results = []

<<<<<<< HEAD
    for idx, score, s, f in combined[:top_k]:
=======
    for idx, score, s, f in combined:
>>>>>>> OCR_Handwritten_Invoices
        if idx < len(matcher.product_database):
            product_name = matcher.product_database[idx]
        else:
            product_name = f"unknown_{idx}"
        
        results.append({
<<<<<<< HEAD
            "name": product_name, 
=======
            "name": product_name,
>>>>>>> OCR_Handwritten_Invoices
            "score": float(score),
            "semantic": float(s),
            "fuzzy": float(f),
            "rank": len(results) + 1,
            "id": idx
        })

    return results

# MAIN FUNCTION
<<<<<<< HEAD
def run_fallback_matching(extraction: dict, matcher, output_dir: Path, top_k=10):
=======
# -----------------------------
def run_fallback_matching(extraction: dict, matcher, output_dir: Path, min_score=0.0):
    """
    Run matching for all items and return ALL matches.
    """
>>>>>>> OCR_Handwritten_Invoices
    items = extraction.get("items", []) if isinstance(extraction, dict) else []

    summary = []

    for it in items:
        name = it.get("product_name") if isinstance(it, dict) else str(it)

        if not name:
            continue

        all_matches = get_all_matches(matcher, name, min_score=min_score)

        summary.append({
            "query": name,
            "matched": all_matches,
            "total_matches": len(all_matches),
            "top_score": float(all_matches[0]["score"]) if all_matches else 0.0
        })

    result = {
        "generated_at": datetime.now().isoformat(),
        "summary": summary,
        "metadata": {
            "total_queries": len(summary),
            "min_score_threshold": min_score,
            "total_products_in_db": len(matcher.product_database),
            "data_source": "Supabase"
        }
    }

    output_path = output_dir / "matches_summary_fallback.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result, output_path

# OPTIONAL: file runner (for testing only)
def run_from_files(input_file: Path, output_file: Path):
    matcher = build_matcher(use_supabase=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    extraction = data.get('extracted') or data
<<<<<<< HEAD
=======

>>>>>>> OCR_Handwritten_Invoices
    result, _ = run_fallback_matching(extraction, matcher, output_file.parent)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

<<<<<<< HEAD
    print("Saved fallback matches to", output_file)
=======
    print("✅ Saved all matches from Supabase to", output_file)
>>>>>>> OCR_Handwritten_Invoices
