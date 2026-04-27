# Data Cleaning Pipeline 🧹

This folder contains the ETL (Extract, Transform, Load) scripts that clean and prepare raw inventory data for analysis.

## 📁 Folder Structure

```
cleaning/
├── README.md              # This file
├── output/               # Cleaned datasets (intermediate output)
│   ├── cleaned_sales_stock.csv      # Cleaned sales & inventory
│   ├── cleaned_pricing.csv          # Cleaned product pricing
│   ├── cleaned_suppliers.csv        # Cleaned supplier data
│   ├── cleaned_warehouse.csv        # Cleaned warehouse inventory
│   └── master_dataframe.csv         # Combined master dataset
│
└── ... (cleaning scripts - to be added)
```

## 🔄 Data Pipeline

```
Raw Data (XLS/Excel)
    ↓
[Cleaning Scripts]
    ├── Remove duplicates
    ├── Handle missing values
    ├── Standardize column names
    ├── Convert data types
    ├── Clean text fields
    └── Validate data quality
    ↓
/output/ (Intermediate CSV)
    ↓
Final /data/ (Ready for Analysis)
    ↓
Analysis & Visualizations
```

## 📊 Cleaning Steps

### 1. Data Loading
- Read XLS/Excel files
- Parse headers in Arabic
- Identify data structure

### 2. Column Standardization
- Rename columns to consistent format
- Convert Arabic column names if needed
- Remove extra spaces

### 3. Data Type Conversion
- Convert product codes to strings
- Convert quantities to numeric
- Parse prices as floats
- Fix date fields (if available)

### 4. Handle Missing Values
- Fill missing stock quantities with 0
- Mark missing prices as NULL
- Document missing data patterns

### 5. Text Cleaning
- Remove leading/trailing whitespace
- Standardize product names
- Normalize category names
- Clean branch names

### 6. Deduplication
- Remove duplicate product codes
- Resolve conflicting records
- Keep first occurrence or aggregate

### 7. Validation
- Check for data type consistency
- Verify required fields populated
- Validate ranges (quantities, prices)
- Generate quality report

## 📁 Output Files

### cleaned_sales_stock.csv
**Purpose:** Main sales and inventory dataset

**Columns:**
- `الكود` - Product code
- `اسم الصنف` - Product name
- `اجمالى مبيعات` - Total sales
- `اجمالى استوك` - Total inventory
- `[branch]_مبيعات` - Sales by branch (e.g., `اسيوط_مبيعات`)
- `[branch]_استوك` - Stock by branch (e.g., `اسيوط_استوك`)

**Rows:** 2,428 products

### cleaned_pricing.csv
**Purpose:** Product pricing and costs

**Columns:**
- `الكود` - Product code
- `اسم الصنف` - Product name
- `مجموعه الصنف` - Product category
- `سعر التكلفه` - Cost price
- `سعر القطعة` - Retail price
- Branch-specific inventory levels

### cleaned_suppliers.csv
**Purpose:** Supplier information and performance

**Columns:**
- Supplier ID/Name
- Contact information
- Performance metrics
- Product count
- Reliability rating

### cleaned_warehouse.csv
**Purpose:** Warehouse-level inventory

**Columns:**
- `الكود` - Product code
- `اسم الصنف` - Product name
- `المخزن` - Warehouse name
- `الكمية` - Quantity in stock
- `سعر بيع الوحدة` - Unit selling price
- `المجموعه الرئيسيه` - Main category

### master_dataframe.csv
**Purpose:** Combined dataset with all information

**Contains:** Merged data from all CSV files above

## 🛠️ Cleaning Rules

### Product Code Handling
```
- Remove decimal points (10000019.0 → 10000019)
- Convert to string for consistency
- Remove leading zeros if not significant
```

### Branch Name Standardization
```
Original → Standardized
اسيوط مع → اسيوط_مبيعات
اسيوط اس → اسيوط_استوك
[branch] مع → [branch]_مبيعات
[branch] اس → [branch]_استوك
```

### Missing Value Handling
```
- Missing quantities → 0 (no sales/stock)
- Missing prices → NULL or computed
- Missing text → "Unknown" or category default
```

### Price Validation
```
- Cost price ≤ Retail price
- No negative prices
- Reasonable price ranges (1-50,000 EGP)
```

### Quantity Validation
```
- Non-negative quantities
- No extreme values (>1,000,000)
- Consistent decimal places
```

## 📋 Data Quality Checks

Before finalizing cleaned data:

- [ ] **Completeness**: All required columns present
- [ ] **Uniqueness**: No duplicate product codes
- [ ] **Consistency**: Branch names match across files
- [ ] **Validity**: Data types correct, values in valid ranges
- [ ] **Accuracy**: No obvious data entry errors
- [ ] **Integrity**: Foreign key relationships maintained

## 🔗 Data Relationships

```
Products (cleaned_sales_stock.csv)
    ├── Pricing (cleaned_pricing.csv)
    │   └── Cost & Retail Prices
    ├── Suppliers (cleaned_suppliers.csv)
    │   └── Supplier Information
    └── Warehouse (cleaned_warehouse.csv)
        └── Warehouse Inventory
```

## 📊 Statistics After Cleaning

| Metric | Value |
|--------|-------|
| Total Products | 2,428 |
| Total Branches | 13 |
| Categories | 10+ |
| Suppliers | TBD |
| Warehouses | TBD |
| Completeness | 95%+ |

## ⚠️ Known Issues

### Missing Time Dimension
- All data is aggregated snapshot
- No transaction-level timestamps
- Cannot track changes over time

### Incomplete Branch Coverage
- Some branches have sparse data
- Many products not sold in all branches
- Consider when filtering by branch

### Price Data
- Some products missing pricing information
- Cost/retail prices may be historical
- Update prices regularly for accuracy

## 🔄 Data Refresh Process

### Manual Refresh
1. Obtain new XLS files from source system
2. Run cleaning scripts
3. Validate output files
4. Copy to `/data/` for analysis
5. Update analysis notebooks if needed

### Automated Refresh (Future)
- Schedule cleaning jobs
- Monitor data quality metrics
- Alert on anomalies
- Archive historical versions

## 💡 Best Practices

### When Cleaning
- ✅ Document all transformations
- ✅ Keep backup of raw data
- ✅ Version control cleaned outputs
- ✅ Log quality metrics
- ✅ Test on sample before full run

### When Using Cleaned Data
- ✅ Verify expectations before analysis
- ✅ Check for NULL/missing values
- ✅ Validate date ranges
- ✅ Document any filtering applied
- ✅ Reference source for transparency

## 🚀 Next Steps

### Short-term
1. Document raw data sources
2. Create automated cleaning scripts
3. Establish quality thresholds
4. Build cleaning validation tests

### Medium-term
1. Add time dimension to data
2. Implement incremental cleaning
3. Create data profiling reports
4. Add anomaly detection in pipeline

### Long-term
1. Implement data warehouse
2. Real-time data ingestion
3. Automated quality monitoring
4. Historical data archival

## 📝 Adding New Data Sources

To add new data to the cleaning pipeline:

1. **Identify source file** (XLS/CSV/DB)
2. **Map columns** to standardized names
3. **Create cleaning script** for transformation
4. **Add validation rules** for quality checks
5. **Test output** before production
6. **Update documentation** (this file)
7. **Integrate** into master dataset

## 🔗 Related Documentation

- [Main README](../README.md) - Project overview
- [CLAUDE.md](../CLAUDE.md) - Complete project documentation
- [Analysis README](../analysis/README.md) - Analysis & visualization guide

## 📞 Questions?

For data-related questions:
- Check data structure in CSV files
- Review this documentation
- See analysis README for usage examples
- Check CLAUDE.md for project context

---

**Last Updated:** 2026-04-27  
**Data Freshness:** Single snapshot (No time dimension)  
**Cleaning Status:** ✅ Complete  
**Quality Score:** TBD (to be established)
