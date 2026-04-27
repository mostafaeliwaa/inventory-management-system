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
├── forcasting models/       # ⚠️ Empty - Requires time-series data
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

## ⚠️ Critical Data Limitation

**NO TIME SERIES DATA** - All datasets are aggregated snapshots WITHOUT timestamps.

**Blocked Applications:**
- ❌ Demand forecasting (ARIMA, Prophet, LSTM)
- ❌ Trend analysis across time
- ❌ Seasonality detection

**Required for Forecasting:**
- Daily/weekly transaction-level data with dates
- 50-100+ observations per product per branch
- Historical data spanning months/years

## ✅ Current Capabilities

### Analytics Available
- ✅ **ABC Classification** - Segment products by sales volume
- ✅ **Dead Stock Detection** - Identify non-moving inventory
- ✅ **Branch Performance** - Compare metrics across 13 branches
- ✅ **Anomaly Detection** - Find unusual stock/sales ratios
- ✅ **Product Clustering** - Group similar products
- ✅ **Supplier Ranking** - Performance metrics by supplier

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
1. Collect transaction-level data with dates
2. Implement demand forecasting
3. Build seasonal models
4. Predict stock-outs

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
