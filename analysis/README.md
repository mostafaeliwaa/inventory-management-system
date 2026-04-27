# Analysis & Visualizations 📊

This folder contains Jupyter notebooks and interactive Plotly dashboards for analyzing the inventory management data.

## 📁 Folder Structure

```
analysis/
├── README.md                    # This file
├── EXP.ipynb                   # Main analysis notebook
├── .vscode/settings.json       # VS Code workspace settings
└── output/
    ├── dashboard.html          # Master dashboard (all charts combined)
    └── charts/
        ├── sales_by_branch.html           # Sales volume per branch
        ├── stock_by_branch.html           # Inventory levels per branch
        ├── sales_vs_stock_ratio.html      # Stock efficiency analysis
        ├── category_performance.html      # Performance by product category
        ├── margin_by_category.html        # Profit margins by category
        ├── stock_heatmap.html             # Branch-product heatmap
        ├── top_products.html              # Best-selling products
        ├── top_suppliers.html             # Top suppliers by performance
        ├── supplier_treemap.html          # Supplier distribution
        ├── dead_stock.html                # Non-moving inventory
        ├── product_scatter.html           # Sales vs Stock scatter plot
        ├── branch_treemap.html            # Branch size comparison
        └── inventory_value_by_branch.html # Inventory value distribution
```

## 📓 Main Notebook: EXP.ipynb

The `EXP.ipynb` notebook contains:

### 1. Data Loading & Cleaning
- Read sales & stock data
- Handle missing values (fill with 0)
- Clean product codes and names
- Standardize branch names

### 2. Analysis Sections

#### Branch Performance Analysis
- Total sales by branch
- Total inventory by branch
- Sales-to-stock ratio
- Comparison charts

#### Product Analysis
- Top 20 products by sales
- Top 20 products by inventory value
- Product distribution by category
- Slow-moving/dead stock detection

#### Category Performance
- Sales by category
- Profit margins by category
- Stock levels by category

#### Supplier Analysis
- Top suppliers ranked
- Performance metrics
- Supplier distribution

#### Anomaly Detection
- Unusual stock/sales ratios
- Branches with low/high inventory

### 3. Visualizations
All charts are exported as interactive HTML files using **Plotly**

## 🎨 Dashboard Types

### Interactive HTML Dashboards
Each dashboard is standalone and can be:
- Opened in any web browser
- Zoomed and panned
- Hovered for detailed values
- Downloaded as PNG

### Key Visualizations

| Chart | Purpose | File |
|-------|---------|------|
| Bar Charts | Branch/category comparison | `*_by_branch.html` |
| Heatmap | Cross-branch-product view | `stock_heatmap.html` |
| Scatter Plot | Correlation analysis | `product_scatter.html` |
| Treemap | Hierarchical distribution | `*_treemap.html` |
| Line/Area | Trends & accumulation | Various |

## 🚀 How to Use

### View Dashboards
1. Open any `.html` file in the `output/` folder with a web browser
2. Interact with charts:
   - Hover for details
   - Click legend to toggle series
   - Zoom/pan with mouse
   - Download as PNG using camera icon

### Run Notebook Analysis
```bash
# Navigate to analysis folder
cd analysis

# Start Jupyter
jupyter notebook EXP.ipynb

# Run all cells or individual sections
```

### Modify Analysis
1. Open `EXP.ipynb` in Jupyter
2. Edit data loading paths if needed
3. Modify filtering/grouping logic
4. Re-run cells to regenerate visualizations
5. Update output HTML files

## 📊 Data Structure Expected

The notebook expects CSV files in `/data/`:
- `cleaned_sales_stock.csv` - Main sales & inventory data
- `cleaned_pricing.csv` - Product pricing information
- `cleaned_suppliers.csv` - Supplier details
- `cleaned_warehouse.csv` - Warehouse inventory

## 🔄 Workflow

```
1. Load CSV Data
   ↓
2. Clean & Transform
   ↓
3. Perform Analysis
   ↓
4. Generate Visualizations
   ↓
5. Export to HTML
   ↓
6. View in Browser
```

## 📈 Key Metrics Calculated

### Branch Level
- Total Sales
- Total Inventory
- Sales-to-Stock Ratio
- Average Product Price
- Active Products

### Product Level
- Total Sales Volume
- Current Stock
- Inventory Value
- Category
- Supplier

### Category Level
- Category Sales
- Category Inventory
- Average Margin
- Number of Products

## ⚠️ Data Limitations

**No Time Series:**
- All data is a single snapshot
- No historical trending
- No seasonal patterns
- Cannot forecast future demand

**For Future Enhancements:**
- Collect daily/weekly transaction data
- Add timestamps to all records
- Track inventory over time

## 🛠️ Technologies Used

- **Pandas** - Data manipulation & analysis
- **Plotly** - Interactive visualizations
- **Jupyter** - Notebook environment
- **NumPy** - Numerical operations

## 📝 Adding New Analysis

To add new analysis to the notebook:

1. **Add new cells** for your analysis
2. **Load required data** from CSV files
3. **Create visualizations** using Plotly
4. **Export chart** to HTML:
   ```python
   fig.write_html('output/charts/my_chart.html')
   ```
5. **Update this README** with chart description

## 🎯 Common Analysis Patterns

### Filter by Branch
```python
branch_data = df[df['branch_column'] == 'branch_name']
```

### Filter by Category
```python
category_data = df[df['category'] == 'category_name']
```

### Top N Products
```python
top_n = df.nlargest(20, 'sales_column')
```

### Sales to Stock Ratio
```python
df['ratio'] = df['sales'] / (df['stock'] + 1)
```

## 📊 Sample Outputs

All visualizations are ready to view in `/output/`:
- Open `dashboard.html` for overview
- Browse individual charts for detailed analysis
- Interact with charts directly in browser

## 🔗 Related Documentation

- [Main README](../README.md) - Project overview
- [CLAUDE.md](../CLAUDE.md) - Complete project documentation
- [Cleaning README](../cleaning/README.md) - Data preparation pipeline

---

**Last Updated:** 2026-04-27  
**Total Charts:** 14+  
**Data Source:** /data/ folder (CSV files)  
**Output Format:** Interactive HTML (Plotly)
