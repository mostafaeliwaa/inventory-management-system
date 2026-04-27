**Overview**
- **Purpose**: Wrapper and UI for the existing OCR + Gemini extraction and product-matching pipeline.
- **Key components**: FastAPI backend ([ocr/api/main.py](ocr/api/main.py)), Streamlit frontend ([ocr/app.py](ocr/app.py)), helper scripts ([ocr/process_uploaded_image.py](ocr/process_uploaded_image.py), [ocr/run_matching_fallback.py](ocr/run_matching_fallback.py)).

**Quickstart**
- **Create and activate venv** (if not already):
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate
```
- **Install required packages** (minimum):
```bash
pip install -r requirements.txt
```
- **Optional packages** (recommended for better matching/embeddings):
```bash
pip install rapidfuzz sentence-transformers faiss-cpu google-generativeai
```

**Run the API**
- Start FastAPI (development):
```bash
cd ocr/api
python main.py
```
- Endpoints of interest:
  - **/process-invoice** (POST) — accepts multipart file `file` (invoice image). Produces per-upload artifacts and JSON response.
  - **/confirm-items** (POST) — accepts corrections to teach the matcher and optionally append to the master CSV.
  - **/health** and **/stats** — service status and matcher statistics.
  - **/debug/startup** — startup diagnostics used by the Streamlit control panel.

**Run the Streamlit UI**
- Launch the frontend which calls the API only:
```bash
streamlit run ocr/app.py
```
- Use the UI to upload an image and press "Process Invoice". The UI shows invoice metadata and a product table with top-10 matches per line; you can edit quantity/unit and confirm matches.

**Process a single image locally**
- The helper script `ocr/process_uploaded_image.py` runs the pipeline for one image and writes three artifacts to a per-image folder under `ocr/results/`:
  - processed image (same extension)
  - sample_invoice.json (Gemini extraction result)
  - matches_summary_fallback.json (top-10 fallback matches)

Usage:
```bash
python ocr/process_uploaded_image.py path/to/invoice.jpg
# or without args: uses newest image in results/
python ocr/process_uploaded_image.py
```

**Artifacts & Paths**
- Per-request results: [ocr/results/](ocr/results/) — each request gets a folder named `<result_id>/` (or `<stem>_processed` for local runs).
- Fallback matcher intermediate files: [ocr/resultsOfModel/](ocr/resultsOfModel/)
- Master product CSV: [cleaning/output/master_dataframe.csv](cleaning/output/master_dataframe.csv)

**Configuration**
- Backend matcher options (thresholds, model name, cache dir) are set in [ocr/product_matcher_system.py](ocr/product_matcher_system.py) during ProductMatcher initialisation in [ocr/api/main.py](ocr/api/main.py).
- To change behavior (e.g., match threshold), edit the ProductMatcher args passed in the API startup.

**Dependencies**
- Core: FastAPI, uvicorn, Streamlit, Pillow, pandas, requests
- Optional (better results): rapidfuzz (fuzzy), sentence-transformers (semantic embeddings), faiss-cpu (fast vector search), google-generativeai (Gemini integration)

**Troubleshooting**
- If the Streamlit UI shows empty product lists:
  - Confirm the API is running and reachable at the URL shown in the Streamlit control panel.
  - Check the API logs for `process_uploaded_image.py` stdout lines — the helper script writes `Done. Output directory:` on success.
- If you see `ModuleNotFoundError` for `sentence_transformers` or `rapidfuzz`, install the optional packages above or rely on the difflib fallback (slower/less accurate).
- If `MatchResult is not JSON serializable` appears, ensure the API version in [ocr/api/main.py](ocr/api/main.py) converts `MatchResult` via `to_dict()` (already implemented).

**Developer notes**
- The API will attempt to run `ocr/process_uploaded_image.py` for each `/process-invoice` call; if it fails the endpoint falls back to inline preprocessing + matching.
- Corrections sent via `/confirm-items` are added to the ProductMatcher memory and persisted to the master CSV.

**Contact / Next steps**
- If you want me to: run an end-to-end test, add authentication, or improve the matching UI, tell me which task to run next.