import csv
import json
from pathlib import Path
from difflib import SequenceMatcher

RESULTS_DIR = Path('ocr/resultsOfModel')
MASTER_CSV = Path('cleaning/output/master_dataframe.csv')
INPUT_FILE = RESULTS_DIR / 'sample_invoice.json'
OUTPUT_FILE = RESULTS_DIR / 'matches_summary_fallback.json'

# Load master product names
product_database = []
if MASTER_CSV.exists():
    with open(MASTER_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header:
            # try to find 'اسم الصنف'
            idx = None
            for i, h in enumerate(header):
                if h.strip() == 'اسم الصنف':
                    idx = i
                    break
            if idx is None:
                idx = 0 if len(header) > 0 else None
            for row in reader:
                try:
                    name = row[idx].strip()
                except Exception:
                    continue
                if name:
                    product_database.append(name)

if not product_database:
    print('No products loaded from master CSV for fallback matcher.')
    raise SystemExit(1)

# Load input
if not INPUT_FILE.exists():
    print('Input invoice JSON not found:', INPUT_FILE)
    raise SystemExit(1)

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

extracted = data.get('extracted') or data
items = extracted.get('items', []) if isinstance(extracted, dict) else []

summary = []
for it in items:
    name = it.get('product_name') if isinstance(it, dict) else str(it)
    if not name:
        continue
    scores = []
    for prod in product_database:
        # simple ratio
        r = SequenceMatcher(None, name, prod).ratio()
        scores.append((prod, r))
    scores.sort(key=lambda x: x[1], reverse=True)
    top = [{'name': p, 'score': float(s)} for p, s in scores[:10]]
    summary.append({'query': name, 'matched': top, 'top_score': float(top[0]['score']) if top else 0.0})

out = {'generated_at': __import__('datetime').datetime.now().isoformat(), 'summary': summary}
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print('Saved fallback matches to', OUTPUT_FILE)
