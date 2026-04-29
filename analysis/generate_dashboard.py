import os
import sys
import json
import datetime
import warnings

os.environ['PYTHONUTF8'] = '1'
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from core.data_fetcher import (
    fetch_branches, fetch_branch_sales, fetch_branch_stock,
    fetch_products, fetch_pricing, fetch_suppliers,
    fetch_product_suppliers, fetch_warehouses, fetch_warehouse_stock,
)

def log(msg):
    print(msg, flush=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# SECTION 1: DATA LOADING (from Supabase)
# ============================================================
log('Connecting to Supabase and loading data...')

df_branches = fetch_branches()
log(f'  branches: {len(df_branches)} rows')

df_sales = fetch_branch_sales()
log(f'  branch_sales: {len(df_sales)} rows')

df_stock = fetch_branch_stock()
log(f'  branch_stock: {len(df_stock)} rows')

df_products = fetch_products()
log(f'  products: {len(df_products)} rows')

df_pricing = fetch_pricing()
log(f'  pricing: {len(df_pricing)} rows')

df_suppliers = fetch_suppliers()
log(f'  suppliers: {len(df_suppliers)} rows')

df_product_suppliers = fetch_product_suppliers()
log(f'  product_suppliers: {len(df_product_suppliers)} rows')

df_warehouses = fetch_warehouses()
log(f'  warehouses: {len(df_warehouses)} rows')

df_warehouse_stock = fetch_warehouse_stock()
log(f'  warehouse_stock: {len(df_warehouse_stock)} rows')

# ============================================================
# SECTION 1.5: PREPARE MERGED VIEWS
# ============================================================
log('Preparing merged data views...')

branch_map = dict(zip(df_branches['branch_id'], df_branches['branch_name']))
SALES_BRANCHES = df_branches['branch_name'].tolist()

df_sales['branch_name'] = df_sales['branch_id'].map(branch_map)
df_stock['branch_name'] = df_stock['branch_id'].map(branch_map)

product_info = df_products[['product_id', 'product_name', 'category_name', 'main_category_name']].copy()
pricing_info = df_pricing[['product_id', 'cost_price', 'item_price']].copy()

product_pricing = product_info.merge(pricing_info, on='product_id', how='left')

sales_with_info = df_sales.merge(product_pricing, on='product_id', how='left')
stock_with_info = df_stock.merge(product_pricing, on='product_id', how='left')

# Aggregate totals per product across all branches
product_total_sales = df_sales.groupby('product_id')['quantity_sold'].sum().reset_index()
product_total_sales.columns = ['product_id', 'total_sales']

product_total_stock = df_stock.groupby('product_id')['stock_quantity'].sum().reset_index()
product_total_stock.columns = ['product_id', 'total_stock']

product_totals = product_total_sales.merge(product_total_stock, on='product_id', how='outer').fillna(0)
product_totals = product_totals.merge(product_info, on='product_id', how='left')
product_totals = product_totals.merge(pricing_info, on='product_id', how='left')

log(f'  product_totals: {len(product_totals)} rows')

# ============================================================
# SECTION 2: KPI COMPUTATION
# ============================================================
log('Computing KPIs...')

# --- Branch Performance ---
branch_perf = []
for _, branch_row in df_branches.iterrows():
    bid = branch_row['branch_id']
    bname = branch_row['branch_name']

    b_sales = sales_with_info[sales_with_info['branch_id'] == bid]
    b_stock = stock_with_info[stock_with_info['branch_id'] == bid]

    total_sales_qty = b_sales['quantity_sold'].sum()
    total_stock_qty = b_stock['stock_quantity'].sum()
    active_items = (b_sales['quantity_sold'] > 0).sum()
    ratio = total_sales_qty / total_stock_qty if total_stock_qty > 0 else 0

    sales_value = (b_sales['quantity_sold'] * b_sales['item_price'].fillna(0)).sum()
    net_profit = (b_sales['quantity_sold'] * (b_sales['item_price'].fillna(0) - b_sales['cost_price'].fillna(0))).sum()

    out_of_stock_count = (b_stock['stock_quantity'] == 0).sum()

    if len(b_sales) > 0 and total_sales_qty > 0:
        top_idx = b_sales['quantity_sold'].idxmax()
        top_product = b_sales.loc[top_idx, 'product_name'] or ''
        top_product_sales = b_sales.loc[top_idx, 'quantity_sold']
    else:
        top_product = ''
        top_product_sales = 0

    branch_perf.append({
        'الفرع': bname,
        'الكمية المباعة': round(float(total_sales_qty), 1),
        'اجمالى المبيعات_جنيه': round(float(sales_value), 1),
        'صافي الربح': round(float(net_profit), 1),
        'عدد المنتجات غير المتوفرة': int(out_of_stock_count),
        'أكثر منتج مبيعًا': top_product,
        'كمية المنتج الأكثر مبيعًا': round(float(top_product_sales), 1),
        'اجمالى الاستوك': round(float(total_stock_qty), 1),
        'نسبة البيع للمخزون': round(float(ratio), 3),
        'اصناف نشطة': int(active_items),
    })
df_branch_perf = pd.DataFrame(branch_perf)

# --- Branch Stock Value ---
branch_value = []
for _, branch_row in df_branches.iterrows():
    bid = branch_row['branch_id']
    bname = branch_row['branch_name']

    b_stock = stock_with_info[stock_with_info['branch_id'] == bid]
    cost_val = (b_stock['stock_quantity'] * b_stock['cost_price'].fillna(0)).sum()
    retail_val = (b_stock['stock_quantity'] * b_stock['item_price'].fillna(0)).sum()
    profit = retail_val - cost_val
    qty = b_stock['stock_quantity'].sum()
    margin = (retail_val - cost_val) / retail_val * 100 if retail_val > 0 else 0

    branch_value.append({
        'الفرع': bname,
        'الكمية': round(float(qty)),
        'قيمة التكلفة': round(float(cost_val)),
        'قيمة البيع': round(float(retail_val)),
        'نسبة الهامش': round(float(margin), 1),
        'الربح': round(float(profit)),
    })
df_branch_value = pd.DataFrame(branch_value)

# --- Top Sellers ---
top_sellers = product_totals.nlargest(20, 'total_sales')[['product_name', 'product_id', 'total_sales', 'total_stock']].copy()
top_sellers.columns = ['اسم الصنف', 'الكود', 'اجمالى مبيعات', 'اجمالى استوك']

# --- Dead Stock (stock > 5, sales = 0) ---
dead_stock_mask = (product_totals['total_stock'] > 5) & (product_totals['total_sales'] == 0)
df_dead_stock = product_totals[dead_stock_mask][['product_name', 'product_id', 'total_stock']].copy()
df_dead_stock.columns = ['اسم الصنف', 'الكود', 'اجمالى استوك']
df_dead_stock = df_dead_stock.sort_values('اجمالى استوك', ascending=False)

# --- Alerts ---
alerts_data = {'out_of_stock': [], 'low_stock': [], 'healthy_stock': []}

for _, row in product_totals.iterrows():
    product_name = row.get('product_name', '')
    product_code = str(row['product_id'])
    total_stock = float(row['total_stock'])
    total_sales_val = float(row['total_sales'])

    if total_stock <= 0:
        alerts_data['out_of_stock'].append({
            'اسم المنتج': product_name, 'الكود': product_code,
            'المخزون': total_stock, 'المبيعات': total_sales_val,
            'الحالة': 'نفد المخزون', 'اللون': 'red'
        })
    elif total_stock <= 5:
        alerts_data['low_stock'].append({
            'اسم المنتج': product_name, 'الكود': product_code,
            'المخزون': total_stock, 'المبيعات': total_sales_val,
            'الحالة': 'قرب ينفد', 'اللون': 'yellow'
        })
    else:
        alerts_data['healthy_stock'].append({
            'اسم المنتج': product_name, 'الكود': product_code,
            'المخزون': total_stock, 'المبيعات': total_sales_val,
            'الحالة': 'متوفر', 'اللون': 'green'
        })

# --- Inventory Movement ---
inventory_movement = []
for _, branch_row in df_branches.iterrows():
    bid = branch_row['branch_id']
    bname = branch_row['branch_name']

    b_sales_qty = df_sales[df_sales['branch_id'] == bid]['quantity_sold'].sum()
    b_stock_qty = df_stock[df_stock['branch_id'] == bid]['stock_quantity'].sum()

    inbound = float(b_stock_qty + b_sales_qty)
    outbound = float(b_sales_qty)
    net_change = inbound - outbound

    inventory_movement.append({
        'الفرع': bname,
        'الوارد': round(inbound, 1),
        'المنصرف': round(outbound, 1),
        'التغير الصافي': round(net_change, 1),
        'نسبة الاستهلاك': round(outbound / inbound * 100, 1) if inbound > 0 else 0
    })

# --- Summary KPIs ---
total_sales = float(product_totals['total_sales'].sum())
total_stock = float(product_totals['total_stock'].sum())
sell_through_rate = round(total_sales / (total_sales + total_stock) * 100, 1) if (total_sales + total_stock) > 0 else 0
best_branch = df_branch_perf.sort_values('نسبة البيع للمخزون', ascending=False).iloc[0]['الفرع'] if not df_branch_perf.empty else 'N/A'
worst_branch = df_branch_perf.sort_values('نسبة البيع للمخزون', ascending=True).iloc[0]['الفرع'] if not df_branch_perf.empty else 'N/A'

summary_kpis = {
    'اجمالى المبيعات_وحدة': round(total_sales, 1),
    'اجمالى المخزون_وحدة': round(total_stock, 1),
    'معدل البيع': sell_through_rate,
    'عدد الراكد': int(len(df_dead_stock)),
    'اجمالى الربح': round(float(df_branch_value['الربح'].sum()), 1),
    'أفضل فرع': best_branch,
    'أسوأ فرع': worst_branch,
}

log(f'  KPIs computed: {len(summary_kpis)} summary metrics')

# ============================================================
# SECTION 2.5: NEW KPIs (Category, Supplier, Warehouse)
# ============================================================
log('Computing enhanced KPIs...')

# --- Category Analysis ---
cat_data = product_totals.dropna(subset=['category_name']).copy()
cat_data = cat_data[cat_data['category_name'].str.strip() != '']

category_analysis = cat_data.groupby('category_name').agg(
    total_sales=('total_sales', 'sum'),
    total_stock=('total_stock', 'sum'),
    product_count=('product_id', 'count'),
    avg_cost=('cost_price', 'mean'),
    avg_price=('item_price', 'mean'),
).reset_index()
category_analysis['sell_through'] = category_analysis.apply(
    lambda r: round(r['total_sales'] / (r['total_sales'] + r['total_stock']) * 100, 1)
    if (r['total_sales'] + r['total_stock']) > 0 else 0, axis=1
)
category_analysis = category_analysis.sort_values('total_sales', ascending=False)

# Main category analysis
main_cat_data = product_totals.dropna(subset=['main_category_name']).copy()
main_cat_data = main_cat_data[main_cat_data['main_category_name'].str.strip() != '']

main_category_analysis = main_cat_data.groupby('main_category_name').agg(
    total_sales=('total_sales', 'sum'),
    total_stock=('total_stock', 'sum'),
    product_count=('product_id', 'count'),
).reset_index()
main_category_analysis = main_category_analysis.sort_values('total_sales', ascending=False)

log(f'  categories: {len(category_analysis)}, main_categories: {len(main_category_analysis)}')

# --- Supplier Performance ---
supplier_map = dict(zip(df_suppliers['supplier_id'], df_suppliers['supplier_name']))
ps = df_product_suppliers.copy()
ps['supplier_name'] = ps['supplier_id'].map(supplier_map)

supplier_products = ps.merge(product_totals[['product_id', 'total_sales', 'total_stock']], on='product_id', how='left')
supplier_perf = supplier_products.groupby('supplier_name').agg(
    product_count=('product_id', 'count'),
    total_sales=('total_sales', 'sum'),
    total_stock=('total_stock', 'sum'),
).reset_index()
supplier_perf = supplier_perf.sort_values('total_sales', ascending=False)

log(f'  supplier performance: {len(supplier_perf)} suppliers')

# --- Warehouse vs Branch Stock ---
wh_stock = df_warehouse_stock.merge(product_info, on='product_id', how='left')
wh_stock = wh_stock.merge(pricing_info, on='product_id', how='left')

warehouse_total_qty = float(wh_stock['quantity'].sum())
warehouse_total_cost = float((wh_stock['quantity'] * wh_stock['cost_price'].fillna(0)).sum())
warehouse_total_retail = float((wh_stock['quantity'] * wh_stock['item_price'].fillna(0)).sum())

branch_total_qty = float(df_stock['stock_quantity'].sum())

warehouse_overview = {
    'warehouse_qty': round(warehouse_total_qty),
    'warehouse_cost_value': round(warehouse_total_cost),
    'warehouse_retail_value': round(warehouse_total_retail),
    'branch_total_qty': round(branch_total_qty),
    'total_inventory_qty': round(warehouse_total_qty + branch_total_qty),
    'warehouse_pct': round(warehouse_total_qty / (warehouse_total_qty + branch_total_qty) * 100, 1) if (warehouse_total_qty + branch_total_qty) > 0 else 0,
}

log(f'  warehouse overview computed')

# ============================================================
# SECTION 3: JSON EXPORT
# ============================================================
log('Building export data...')

# Scatter data
scatter_df = product_totals[(product_totals['total_sales'] > 0) | (product_totals['total_stock'] > 0)].copy()
scatter_products = scatter_df.rename(columns={
    'product_name': 'name', 'total_sales': 'sales', 'total_stock': 'stock'
})[['name', 'sales', 'stock']].to_dict(orient='records')
scatter_med_sales = round(float(scatter_df['total_sales'].median()), 1)
scatter_med_stock = round(float(scatter_df['total_stock'].median()), 1)

# Per-item per-branch data
log('Building per-item per-branch data...')
items_records = []

sales_pivot = df_sales.pivot_table(index='product_id', columns='branch_name', values='quantity_sold', fill_value=0)
stock_pivot = df_stock.pivot_table(index='product_id', columns='branch_name', values='stock_quantity', fill_value=0)

for pid in product_totals['product_id'].values:
    p_row = product_totals[product_totals['product_id'] == pid].iloc[0]
    item = {
        'code': str(pid),
        'name': str(p_row.get('product_name', '') or ''),
        'total_sales': float(p_row['total_sales']),
        'total_stock': float(p_row['total_stock']),
        'category': str(p_row.get('category_name', '') or 'غير محدد'),
    }

    branch_performance = []
    best_branch_item = {'name': '', 'sell_through': 0}
    worst_branch_item = {'name': '', 'sell_through': 100}
    out_of_stock_branches = []
    low_stock_branches = []

    for bname in SALES_BRANCHES:
        sales = float(sales_pivot.loc[pid, bname]) if pid in sales_pivot.index and bname in sales_pivot.columns else 0
        stock = float(stock_pivot.loc[pid, bname]) if pid in stock_pivot.index and bname in stock_pivot.columns else 0

        sell_through = 0
        if stock > 0:
            sell_through = round((sales / stock) * 100, 1)
        elif sales > 0 and stock == 0:
            sell_through = 100

        if sell_through >= 70:
            status = 'ممتاز'
            status_emoji = '🔥'
        elif sell_through >= 40:
            status = 'متوسط'
            status_emoji = '⚠️'
        else:
            status = 'ضعيف'
            status_emoji = '❌'

        if sell_through > best_branch_item['sell_through']:
            best_branch_item = {'name': bname, 'sell_through': sell_through}
        if 0 < sell_through < worst_branch_item['sell_through']:
            worst_branch_item = {'name': bname, 'sell_through': sell_through}

        if sales > 0 and stock == 0:
            out_of_stock_branches.append(bname)
        elif 0 < stock <= 5:
            low_stock_branches.append(bname)

        item[f'{bname}_sales'] = sales
        item[f'{bname}_stock'] = stock
        item[f'{bname}_sell_through'] = sell_through
        item[f'{bname}_status'] = status
        item[f'{bname}_status_emoji'] = status_emoji

        branch_performance.append({
            'branch': bname, 'sales': sales, 'stock': stock,
            'sell_through': sell_through, 'status': status, 'status_emoji': status_emoji,
        })

    item['best_branch'] = best_branch_item['name'] if best_branch_item['name'] else 'لا يوجد'
    item['best_branch_sell_through'] = best_branch_item['sell_through']
    item['worst_branch'] = worst_branch_item['name'] if worst_branch_item['name'] else 'لا يوجد'
    item['worst_branch_sell_through'] = worst_branch_item['sell_through']
    item['out_of_stock_count'] = len(out_of_stock_branches)
    item['out_of_stock_branches'] = out_of_stock_branches
    item['low_stock_count'] = len(low_stock_branches)
    item['low_stock_branches'] = low_stock_branches
    item['branch_performance'] = branch_performance

    items_records.append(item)

log(f'  Items data exported: {len(items_records)} items')

# ============================================================
# SECTION 4: ASSEMBLE & WRITE JSON
# ============================================================
log('Exporting dashboard data to JSON...')

dashboard_data = {
    'metadata': {
        'generated_at': datetime.datetime.now().isoformat(),
        'version': '4.0',
        'source': 'Supabase',
    },
    'summary': summary_kpis,
    'branch_performance': df_branch_perf.to_dict(orient='records'),
    'branch_stock_value': df_branch_value.to_dict(orient='records'),
    'top_sellers': top_sellers.to_dict(orient='records'),
    'dead_stock': {
        'count': len(df_dead_stock),
        'items': df_dead_stock.head(30).to_dict(orient='records'),
    },
    'alerts': alerts_data,
    'inventory_movement': inventory_movement,
    'scatter_data': {
        'products': scatter_products,
        'median_sales': scatter_med_sales,
        'median_stock': scatter_med_stock,
    },
    'sales_branches': SALES_BRANCHES,
    'items_data': items_records,
    'category_analysis': category_analysis.to_dict(orient='records'),
    'main_category_analysis': main_category_analysis.to_dict(orient='records'),
    'supplier_performance': supplier_perf.head(30).to_dict(orient='records'),
    'warehouse_overview': warehouse_overview,
}

with open(os.path.join(OUTPUT_DIR, 'dashboard_data.json'), 'w', encoding='utf-8') as f:
    json.dump(dashboard_data, f, ensure_ascii=False, indent=2, default=str)

log('')
log('=' * 50)
log('Done! Output generated:')
log(f'  Dashboard Data: {os.path.join(OUTPUT_DIR, "dashboard_data.json")}')
log('=' * 50)
