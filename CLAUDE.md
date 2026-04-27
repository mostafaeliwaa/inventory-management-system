# Inventory Management System - Project Documentation

## 📋 Project Overview
Inventory management system for a retail/distribution business with **13 branches** across Egypt. Built in Python with Jupyter notebooks for analysis and visualization.

## 📊 Data Structure

### Available Datasets (cleaned CSV format in `/data/`)
| File | Records | Contents |
|------|---------|----------|
| `cleaned_sales_stock.csv` | 2,428 products | Product codes, names, sales & stock by branch |
| `cleaned_pricing.csv` | Products | Cost & retail prices, categories |
| `cleaned_suppliers.csv` | Suppliers | Supplier information |
| `cleaned_warehouse.csv` | Warehouse | Inventory levels by warehouse |

### Branches (13 total)
اسيوط, اكتوبر, العباسيه, معادى, تجمع, توى المعادى, توى التجمع, شيراتون, محرم, مدينتى, مينى ستور محرم, النخيل, الاداره

### Product Categories
- العب (Toys/Games - major category)
- أدوات تجميل (Cosmetics)
- شيفون, جرابات, and others

## ⚠️ CRITICAL DATA LIMITATION
**NO TIME SERIES DATA** - All datasets are aggregated snapshots WITHOUT dates/timestamps.

**This blocks:**
- ❌ Demand forecasting (ARIMA, Prophet, LSTM)
- ❌ Trend analysis
- ❌ Seasonality modeling

**To enable forecasting, you need:**
- Daily/weekly transaction-level data with dates
- 50-100+ observations per product/branch
- Historical data spanning months/years

## ✅ Feasible ML Applications (with current data)
- **ABC Classification** - Classify products as A/B/C based on sales
- **Dead Stock Detection** - Identify non-moving inventory
- **Branch Performance Analysis** - Compare branches by metrics
- **Anomaly Detection** - Find unusual stock/sales ratios
- **Product Clustering** - Group similar products
- **Supplier Optimization** - Rank suppliers by performance

## 📁 Project Structure
```
├── /data/              - Cleaned CSV datasets
├── /analysis/          - Jupyter notebooks & visualizations
│   ├── EXP.ipynb      - Main analysis notebook
│   ├── /output/       - HTML dashboards (Plotly)
├── /cleaning/         - ETL & data cleaning scripts
├── /forcasting models/ - (empty - needs time series data)
├── /ocr/              - (empty)
└── requirements.txt   - Python dependencies
```

## 🔧 Tech Stack
- **Language**: Python
- **Data**: Pandas, NumPy
- **Visualization**: Plotly
- **Notebooks**: Jupyter
- **Format**: CSV (cleaned data)

## 📊 Analysis Output
- Interactive Plotly dashboards in `/analysis/output/`
- Charts: branch performance, category analysis, dead stock, supplier rankings, etc.

## 🎯 Next Steps for ML
1. **Collect time-series data** with transaction dates
2. **Implement ABC classification** using current data
3. **Build anomaly detection** for inventory issues
4. **Once you have historical data**: Add forecasting models

---
*Last updated: 2026-04-27*
