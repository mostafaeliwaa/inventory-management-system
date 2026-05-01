import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json

from ocr.Models.product_matcher_system import ProductMatcher, MatchResult


def test_normalization():
    """Test Arabic text normalization."""
    from ocr.Models.product_matcher import normalize_product_name, normalize_arabic, normalize_mixed_script
    
    test_cases = [
        ("إسكواش  فراولة ٥٠٠ جرام", "اسكواش فراولة 500 جرام"),
        ("مياه معدنية   (natural water)", "مياه معدنية natural water"),
        ("خبز ٌإفرنجي ُطازج", "خبز افرنجي طازج"),
        ("Coca-Cola ١.٥ لتر", "coca-cola 1.5 لتر"),
    ]
    
    print("Testing normalization functions:")
    print("=" * 60)
    
    for input_text, expected in test_cases:
        result = normalize_product_name(input_text)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{input_text}'")
        print(f"  → '{result}'")
        if result != expected:
            print(f"  Expected: '{expected}'")
    print()


def test_product_matching():
    """Test the complete product matching system."""
    product_database = [
        "إسكواش فراولة 500 جرام",
        "مياه معدنية طبيعية 1.5 لتر",
        "خبز إفرنجي طازج",
        "جبنة رومي 250 جرام",
        "زيت زيتون بكر 1 لتر",
        "سكر أبيض 1 كيلو",
        "أرز مصري 5 كيلو",
        "شاي أحمد 100 جرام",
        "قهوة تركية 250 جرام",
        "حليب طازج 1 لتر",
    ]
    
    print("Initializing ProductMatcher...")
    matcher = ProductMatcher(
        product_database=product_database,
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        semantic_weight=0.7,
        fuzzy_weight=0.3,
        match_threshold=0.6,
        cache_dir="./.test_cache"
    )
    test_cases = [
        ("إسكواش  فراولة ٥٠٠ جرام", "إسكواش فراولة 500 جرام", True),
        ("مياه معدنية   (طبيعية)", "مياه معدنية طبيعية 1.5 لتر", True),
        ("خبز ٌإفرنجي ُطازج", "خبز إفرنجي طازج", True),
        ("جبنه رومى ٢٥٠ جرام", "جبنة رومي 250 جرام", True),
        ("سكر ابيض 1 ك", "سكر أبيض 1 كيلو", True),
        ("منتج غير معروف", "منتج غير معروف", False),
    ]
    
    print("\nTesting product matching:")
    print("=" * 60)
    
    for query, expected_match, should_match in test_cases:
        print(f"\nQuery: '{query}'")
        result = matcher.hybrid_match(query)
        
        print(f"  → Matched: '{result.matched}'")
        print(f"  → Score: {result.score:.3f}")
        print(f"  → Method: {result.method_breakdown}")
        print(f"  → Candidates: {result.candidate_count}")
        
        if should_match:
            if result.matched == expected_match and result.score >= matcher.match_threshold:
                print(f"  ✓ Correct match")
            else:
                print(f"  ✗ Expected: '{expected_match}'")
        else:
            if result.matched == query:
                print(f"  ✓ Correctly returned original (no match)")
            else:
                print(f"  ✗ Should have returned original")
    
    return matcher


def test_learning_system(matcher):
    """Test the learning from user corrections."""
    print("\n\nTesting learning system:")
    print("=" * 60)
    
    wrong_query = "جبنه رومى ٢٥٠ جرام"
    correct_product = "جبنة رومي 250 جرام"
    
    print(f"Adding correction: '{wrong_query}' → '{correct_product}'")
    matcher.add_correction(wrong_query, correct_product)

    print(f"\nTesting query after correction: '{wrong_query}'")
    result = matcher.hybrid_match(wrong_query)
    
    print(f"  → Matched: '{result.matched}'")
    print(f"  → Score: {result.score:.3f}")
    print(f"  → Is correction: {result.is_correction}")
    
    if result.is_correction and result.matched == correct_product:
        print("  ✓ Correction applied successfully")
    else:
        print("  ✗ Correction not applied")

    stats = matcher.get_stats()
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def test_batch_matching():
    """Test batch matching functionality."""
    print("\n\nTesting batch matching:")
    print("=" * 60)
    
    product_database = [
        "إسكواش فراولة 500 جرام",
        "مياه معدنية طبيعية 1.5 لتر",
        "خبز إفرنجي طازج",
        "سكر أبيض 1 كيلو",
    ]
    
    matcher = ProductMatcher(
        product_database=product_database,
        match_threshold=0.5
    )
    
    queries = [
        "إسكواش  فراولة ٥٠٠ جرام",
        "مياه معدنية",
        "خبز طازج",
        "منتج غير موجود",
    ]
    
    results = matcher.batch_match(queries)
    
    for query, result in zip(queries, results):
        print(f"'{query}' → '{result.matched}' (score: {result.score:.3f})")


def run_matching_from_results(results_dir: str = "ocr/resultsOfModel",
                              master_csv: str = "cleaning/output/master_dataframe.csv",
                              output_file: str = "ocr/resultsOfModel/matches_summary.json"):
    """Load OCR outputs from `results_dir`, match items against product names
    from the master CSV, and write a JSON summary to `output_file`.

    This function does NOT modify any files under `cleaning/output`.
    """
    import csv
    from pathlib import Path

    results_path = Path(results_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master_path = Path(master_csv)
    if not master_path.exists():
        print(f"Master CSV not found: {master_path}")
        return

    product_database = []
    with open(master_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            print("Master CSV has no header")
            return
        try:
            name_col_idx = next(i for i, h in enumerate(header) if h.strip() == 'اسم الصنف')
        except StopIteration:
            name_col_idx = 1

        for row in reader:
            name = row[name_col_idx].strip() if len(row) > name_col_idx else ''
            if name:
                product_database.append(name)

    if not product_database:
        print("No products loaded from master CSV.")
        return

    print(f"Loaded {len(product_database)} products from master CSV")

    matcher = ProductMatcher(product_database=product_database, cache_dir=".pm_cache")

    try:
        matcher.compute_embeddings()
    except Exception as e:
        print(f"Warning: embeddings not precomputed: {e}")

    summary = []

    json_files = sorted(results_path.glob("*.json"))
    if not json_files:
        print(f"No JSON result files found in {results_path}")
        return

    for jf in json_files:
        try:
            data = json.load(open(jf, 'r', encoding='utf-8'))
        except Exception as e:
            print(f"Failed to read {jf}: {e}")
            continue
        extracted = None
        if isinstance(data, dict) and 'extracted' in data:
            extracted = data['extracted']
        elif isinstance(data, dict) and 'extracted_data' in data:
            extracted = data['extracted_data']
        else:
            extracted = data

        if not extracted or (isinstance(extracted, dict) and 'error' in extracted):
            summary.append({
                'source_file': str(jf),
                'status': 'error',
                'error': extracted.get('error') if isinstance(extracted, dict) else 'no_data'
            })
            continue

        items = []
        if isinstance(extracted, dict) and 'items' in extracted and isinstance(extracted['items'], list):
            items = extracted['items']
        elif isinstance(extracted, list):
            items = extracted
        else:
            for key in ('lines', 'products', 'lines_of_text'):
                if isinstance(extracted, dict) and key in extracted and isinstance(extracted[key], list):
                    items = extracted[key]
                    break

        image_results = {
            'source_file': str(jf),
            'matches': []
        }

        for it in items:
            if isinstance(it, dict):
                name = it.get('product_name') or it.get('extracted_text') or it.get('text') or it.get('name')
            else:
                name = str(it)

            if not name:
                continue
            from rapidfuzz import process
            top_matches = process.extract(name, product_database, limit=10, score_cutoff=0.1)
            matched_list = [{'name': match_name, 'score': float(score / 100.0)} for match_name, score, _ in top_matches]

            image_results['matches'].append({
                'query': name,
                'matched': matched_list,
                'top_score': float(top_matches[0][1] / 100.0) if top_matches else 0.0
            })

        summary.append(image_results)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'generated_at': __import__('datetime').datetime.now().isoformat(), 'summary': summary}, f, ensure_ascii=False, indent=2)

    print(f"Saved matches summary to {output_path}")


if __name__ == "__main__":
    try:
        run_matching_from_results()
    except Exception as e:
        print(f"Error running matching: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("Product Matching System - Comprehensive Test")
    print("=" * 70)
    test_normalization()
    matcher = test_product_matching()
    test_learning_system(matcher)
    test_batch_matching()
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    import shutil
    if os.path.exists("./.test_cache"):
        shutil.rmtree("./.test_cache")
        print("Test cache cleaned up.")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nPlease install required packages:")
        print("  pip install -r requirements.txt")
        print("\nOr at minimum:")
        print("  pip install sentence-transformers rapidfuzz numpy")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
