from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from routers.chatbot import router as chatbot_router
from routers.ocr import router as ocr_router, _state, log, setup_gemini, ProductMatcher, get_supabase_client, PRODUCT_DB_PATH, RESULTS_DIR
import pandas as pd

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up Inventory Management System...")

    # OCR Services
    try:
        _state["gemini_model"] = setup_gemini()
        log.info("Gemini model loaded successfully.")
    except Exception as exc:
        log.error("Failed to load Gemini model: %s", exc)

    
    try:
        supabase = get_supabase_client()
        table_info = supabase.get_table_info(table_name="products")
        
        if table_info.get("exists"):
            product_names = supabase.get_all_products(
                table_name="products",
                product_name_column="product_name"
            )
            
            if product_names:
                _state["product_matcher"] = ProductMatcher(product_database=product_names)
                _state["product_matcher"].compute_embeddings()
                log.info("✅ ProductMatcher ready with %d products from Supabase.", len(product_names))
                log.info("📋 Sample products: %s", product_names[:5])
            else:
                log.warning("⚠️ No products found in Supabase table 'products'")
                _state["product_matcher"] = None
        else:
            log.warning("⚠️ Table 'products' not found in Supabase")
            _state["product_matcher"] = None
            
    except Exception as exc:
        log.error("❌ Failed to load products from Supabase: %s", exc)
        log.info("🔄 Falling back to CSV file...")
        
        try:
            if PRODUCT_DB_PATH.exists():
                df = pd.read_csv(PRODUCT_DB_PATH)
                
                product_names = []
                
                possible_name_cols = ['اسم الصنف', 'product_name', 'name', 'Product', 'الصنف', 'اسم المنتج']
                name_col = None
                
                for col in possible_name_cols:
                    if col in df.columns:
                        name_col = col
                        break
                
                if name_col is None and len(df.columns) > 0:
                    first_col = df.columns[0]
                    sample_vals = df[first_col].dropna().head(10).astype(str)
                    if not sample_vals.str.match(r'^\d+$').all():
                        name_col = first_col
                    elif len(df.columns) > 1:
                        name_col = df.columns[1]
                
                if name_col:
                    product_names = df[name_col].dropna().astype(str).str.strip().tolist()
                    product_names = [p for p in product_names if p and p.lower() != 'nan']
                else:
                    for col in df.columns:
                        for val in df[col].dropna():
                            str_val = str(val).strip()
                            if str_val and not str_val.isdigit() and str_val.lower() != 'nan':
                                product_names.append(str_val)
                
                if not product_names:
                    log.warning("No product names found, using IDs as fallback")
                    first_col = df.columns[0]
                    product_names = df[first_col].dropna().astype(str).str.strip().tolist()
                    product_names = [p for p in product_names if p and p.lower() != 'nan']
                
                _state["product_matcher"] = ProductMatcher(product_database=product_names)
                _state["product_matcher"].compute_embeddings()
                log.info("ProductMatcher ready with %d products from CSV.", len(product_names))
                log.info("Sample products: %s", product_names[:5])
            else:
                log.warning("Product database not found at %s", PRODUCT_DB_PATH)
        except Exception as csv_exc:
            log.error("Failed to load from CSV as well: %s", csv_exc)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _state["ready"] = _state["gemini_model"] is not None and _state["product_matcher"] is not None
    log.info("API startup complete. Ready=%s", _state["ready"])

    yield
    
    # ---- shutdown ----
    log.info("Shutting down Inventory Management System.")

app = FastAPI(title="Inventory Management System", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    chatbot_router,
    prefix="/chatbot",
    tags=["Chatbot"]
)

app.include_router(
    ocr_router,
    prefix="/ocr",
    tags=["OCR"]
)