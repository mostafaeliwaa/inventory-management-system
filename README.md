# Inventory Management System 📦

A Python-based inventory management system for retail distribution business with 13 branches across Egypt. Includes data analysis, visualization dashboards, and data cleaning pipelines.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Pandas, NumPy, Plotly
- Jupyter Notebook

### Installation
```bash
pip install -r requirements.txt
```

### Run Analysis
```bash
cd analysis
jupyter notebook EXP.ipynb
```

## 📁 Project Structure

```
inventory-management-system/
├── README.md                 # This file
├── CLAUDE.md                 # Project documentation for AI assistant
├── requirements.txt          # Python dependencies
│
├── data/                     # Cleaned CSV datasets (ready for analysis)
│   ├── cleaned_sales_stock.csv      # 2,428 products with sales & stock by branch
│   ├── cleaned_pricing.csv          # Product pricing and costs
│   ├── cleaned_suppliers.csv        # Supplier information
│   └── cleaned_warehouse.csv        # Warehouse inventory levels
│
├── analysis/                 # Jupyter notebooks and visualizations
│   ├── README.md            # Analysis documentation
│   ├── EXP.ipynb           # Main analysis notebook
│   ├── output/
│   │   ├── dashboard.html           # Master dashboard
│   │   └── charts/
│   │       ├── sales_by_branch.html
│   │       ├── stock_by_branch.html
│   │       ├── dead_stock.html
│   │       ├── top_suppliers.html
│   │       └── ... (more charts)
│   └── .vscode/settings.json
│
├── cleaning/                # ETL scripts and data cleaning
│   ├── README.md           # Cleaning pipeline documentation
│   ├── output/             # Cleaned datasets (before final /data/)
│   └── ...
│
├── ML models/               # Demand classifier + dead-stock predictor
│   ├── README.md
│   ├── streamlit_app.py    # Streamlit UI (port 8501) — manual testing
│   ├── api.py              # FastAPI service (port 8003)
│   ├── src/                # data loader (CSV or Supabase), training, evaluation, inference
│   ├── notebooks/          # demo_and_evaluation.ipynb walkthrough
│   ├── test_data/          # JSON payloads + API suite runner
│   ├── artifacts/          # serialized .joblib models
│   └── reports/            # JSON evaluation reports
│
├── forecasting/             # Time-series forecasting module
│   ├── src/                 # ML pipeline: data prep, features, models (Prophet/LGBM)
│   ├── reports/             # CV metrics and JSON forecast snapshots
│   └── api.py               # FastAPI service (port 8004)
│
├── chatbot/                 # LangGraph chatbot + FastAPI (port 8002)
│
└── ocr/                     # ⚠️ Empty - OCR processing (future)
```

## 📊 Data Overview

### Branches (13 total)
- اسيوط, اكتوبر, العباسيه, معادى, تجمع
- توى المعادى, توى التجمع, شيراتون, محرم
- مدينتى, مينى ستور محرم, النخيل, الاداره

### Product Categories
- **العب** (Toys/Games) - Major category
- **أدوات تجميل** (Cosmetics)
- **شيفون**, **جرابات**, and others

### Dataset Summary
| File | Records | Description |
|------|---------|-------------|
| cleaned_sales_stock.csv | 2,428 | Product sales & stock per branch |
| cleaned_pricing.csv | ~2,428 | Cost & retail prices by product |
| cleaned_suppliers.csv | Varies | Supplier details & performance |
| cleaned_warehouse.csv | Varies | Warehouse inventory snapshots |

## ⚠️ Data Limitations & Time Series

- **Initial Core Data**: Was aggregated snapshots WITHOUT timestamps.
- **New Forecasting Data**: We have incorporated `merged_new_sales.csv` (144 days of POS data for 8 branches, from 2026-01-01 to 2026-05-24) to unlock branch-level forecasting.
- **Still Blocked**: ❌ SKU/Product-level demand forecasting (receipts don't carry product codes).

## ✅ Current Capabilities

### Analytics Available
- ✅ **ABC Classification** - Segment products by sales volume
- ✅ **Dead Stock Detection** - Identify non-moving inventory
- ✅ **Branch Performance** - Compare metrics across 13 branches
- ✅ **Anomaly Detection** - Find unusual stock/sales ratios
- ✅ **Product Clustering** - Group similar products
- ✅ **Supplier Ranking** - Performance metrics by supplier

### Machine Learning Models (`/ML models/`)
- ✅ **Demand Classification** — Random Forest predicts fast / medium / slow movers
  (robust macro F1 ≈ 0.87–0.91 with rule-defining features hidden)
- ✅ **Dead-Stock Prediction** — Gradient Boosting predicts non-moving inventory
  (robust F1 ≈ 0.95–0.97, ROC-AUC ≈ 0.998 with rule-defining features hidden)
- ✅ **Streamlit UI** on port 8501 — pick a product, manual feature entry, batch test, eval reports
- ✅ **FastAPI service** on port 8003 with `/predict/demand`, `/predict/dead-stock`,
  `/predict/by-code/{code}`, `/predict/batch`, `/metrics`, `/health`
- ✅ **Data source switch** — `DATA_SOURCE=csv` (default) or `DATA_SOURCE=supabase`
  reads directly from the normalized Supabase schema in `data/schema.sql`

### Time-Series Forecasting (`/forecasting/`)
- ✅ **Daily Revenue Forecasts** — 7-day horizon using an Ensemble of Prophet and LightGBM (WAPE 0.479).
- ✅ **Weekly Revenue Forecasts** — 4-week horizon using an Ensemble3 (Prophet + LGBM + Naive Mean4) trimming variance for high accuracy (WAPE 0.229).
- ✅ Handles Ramadan/Eid shocks, weekly cycle, intra-week volatility, Egyptian holidays, and payday/school-holiday windows.
- ✅ **FastAPI service** on port 8004 with `/forecast`, `/forecast/weekly`, `/metrics` endpoints.

```bash
cd "ML models"
pip install -r requirements.txt
python -m src.train_demand && python -m src.train_dead_stock && python -m src.evaluate

# UI (recommended first step)
streamlit run streamlit_app.py            # http://localhost:8501

# API
uvicorn api:app --port 8003 --reload      # http://localhost:8003/docs

# Pull data from Supabase instead of the CSVs:
DATA_SOURCE=supabase streamlit run streamlit_app.py
```

See [ML models/README.md](ML%20models/README.md) for full details.

### Dashboards
- Interactive Plotly visualizations in `/analysis/output/`
- Responsive HTML dashboards
- Multiple chart types: bars, heatmaps, treemaps, scatter plots

## 📖 Documentation

- **[CLAUDE.md](CLAUDE.md)** - Complete project documentation
- **[analysis/README.md](analysis/README.md)** - Analysis & visualization guide
- **[cleaning/README.md](cleaning/README.md)** - Data cleaning pipeline
- **[analysis/EXP.ipynb](analysis/EXP.ipynb)** - Main analysis notebook with examples

## 🛠️ Tech Stack

| Component | Tools |
|-----------|-------|
| Language | Python 3.8+ |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Notebooks | Jupyter |
| Data Format | CSV |

## 📈 Next Steps

### Priority 1: With Current Data
1. Implement ABC classification
2. Build dead stock dashboard
3. Create anomaly detection alerts
4. Supplier performance ranking

### Priority 2: With Historical Data
1. ✅ Collect transaction-level data with dates (Done for branch-level)
2. ✅ Implement demand forecasting (Done via Prophet/LGBM ensemble)
3. ✅ Build seasonal models (Done - captures weekly cycle & Ramadan/Eid)
4. Predict SKU-level stock-outs (Blocked by lack of SKU keys in time-series)

## 🔄 Data Flow

```
Raw Data → Cleaning Pipeline → /cleaning/output/ → Final /data/ → Analysis → Dashboards
   ↓                                                                          ↓
External                                                              HTML Visualizations
Sources                                                              (Plotly Charts)
```

## 📝 Requirements

```
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.0.0
jupyter>=1.0.0
xlrd>=2.0.0
openpyxl>=3.0.0
```

## 🤝 Contributing

When adding new analysis:
1. Add notebooks to `/analysis/`
2. Export visualizations to `/analysis/output/charts/`
3. Document in this README
4. Update CLAUDE.md if structure changes

## 📞 Support

- Check [CLAUDE.md](CLAUDE.md) for detailed project info
- See [analysis/README.md](analysis/README.md) for notebook documentation
- Review [cleaning/README.md](cleaning/README.md) for data pipeline info

---

**Last Updated:** 2026-04-27  
**Data Last Refreshed:** Single snapshot (No time dimension)  
**Products:** 2,428  
**Branches:** 13
