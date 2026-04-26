import os
import json
import datetime
import warnings

os.environ['PYTHONUTF8'] = '1'
warnings.filterwarnings('ignore')

import pandas as pd

def log(msg):
    print(msg, flush=True)

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_DIR = os.path.join(BASE_DIR, '..', 'cleaning', 'output')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SECTION 1: DATA LOADING (from cleaned CSVs)
# ============================================================
log('Loading cleaned data files...')

df_branch_stock = pd.read_csv(os.path.join(CLEANED_DIR, 'cleaned_pricing.csv'))
df_branch_stock.columns = df_branch_stock.columns.str.strip()
df_branch_stock['الكود'] = df_branch_stock['الكود'].astype(str).str.strip()

BRANCH_COLS = ['الاداره', 'النخيل', 'هوالك', 'المعادي', 'اكتوبر', 'شيراتون',
               'سبورتينج', 'محرم', 'اسيوط', 'توى لاند- المعادي', 'توى لاند النخيل', 'مدينتي']

df_sales = pd.read_csv(os.path.join(CLEANED_DIR, 'cleaned_sales_stock.csv'))
df_sales.columns = df_sales.columns.str.strip()
df_sales['الكود'] = df_sales['الكود'].astype(str).str.strip()

SALES_BRANCHES = [
    'اسيوط', 'اكتوبر', 'العباسيه', 'معادى', 'تجمع',
    'توى المعادى', 'توى التجمع', 'شيراتون', 'محرم', 'مدينتى', 'مينى ستور محرم',
]

log(f'  branch stock: {len(df_branch_stock)} rows')
log(f'  sales: {len(df_sales)} rows')

# ============================================================
# SECTION 2: KPI COMPUTATION
# ============================================================
log('Computing KPIs...')

# --- Branch Performance ---
branch_perf = []
for name in SALES_BRANCHES:
    total_sales = df_sales[f'{name}_مبيعات'].sum()
    total_stock = df_sales[f'{name}_استوك'].sum()
    active_items = (df_sales[f'{name}_مبيعات'] > 0).sum()
    ratio = total_sales / total_stock if total_stock > 0 else 0

    sales_value = 0
    net_profit = 0
    out_of_stock_count = 0
    top_product = ''
    top_product_sales = 0

    for idx, row in df_sales.iterrows():
        sales_qty = row.get(f'{name}_مبيعات', 0)
        stock_qty = row.get(f'{name}_استوك', 0)
        product_code = str(row['الكود'])

        matching = df_branch_stock[df_branch_stock['الكود'] == product_code]
        if not matching.empty:
            unit_price = matching.iloc[0]['سعر القطعة']
            unit_cost = matching.iloc[0]['سعر التكلفه']
            sales_value += sales_qty * unit_price
            net_profit += sales_qty * (unit_price - unit_cost)

        if stock_qty == 0:
            out_of_stock_count += 1

        if sales_qty > top_product_sales:
            top_product_sales = sales_qty
            top_product = row.get('اسم الصنف', '')

    branch_perf.append({
        'الفرع': name,
        'الكمية المباعة': round(total_sales, 1),
        'اجمالى المبيعات_جنيه': round(sales_value, 1),
        'صافي الربح': round(net_profit, 1),
        'عدد المنتجات غير المتوفرة': out_of_stock_count,
        'أكثر منتج مبيعًا': top_product,
        'كمية المنتج الأكثر مبيعًا': round(top_product_sales, 1),
        'اجمالى الاستوك': round(total_stock, 1),
        'نسبة البيع للمخزون': round(ratio, 3),
        'اصناف نشطة': int(active_items),
    })
df_branch_perf = pd.DataFrame(branch_perf)

# --- Branch Stock Value (from branch stock file) ---
branch_value = []
for col in BRANCH_COLS:
    cost_val = (df_branch_stock[col] * df_branch_stock['سعر التكلفه']).sum()
    retail_val = (df_branch_stock[col] * df_branch_stock['سعر القطعة']).sum()
    profit = retail_val - cost_val
    qty = df_branch_stock[col].sum()
    margin = (retail_val - cost_val) / retail_val * 100 if retail_val > 0 else 0
    branch_value.append({
        'الفرع': col,
        'الكمية': round(qty),
        'قيمة التكلفة': round(cost_val),
        'قيمة البيع': round(retail_val),
        'نسبة الهامش': round(margin, 1),
        'الربح': round(profit),
    })
df_branch_value = pd.DataFrame(branch_value)

# --- Top Sellers ---
df_top_sellers = df_sales.nlargest(20, 'اجمالى مبيعات')[['اسم الصنف', 'الكود', 'اجمالى مبيعات', 'اجمالى استوك']].copy()

# --- Dead Stock (stock > 5, sales = 0) ---
df_dead_stock = df_sales[(df_sales['اجمالى استوك'] > 5) & (df_sales['اجمالى مبيعات'] == 0)].copy()
df_dead_stock = df_dead_stock[['اسم الصنف', 'الكود', 'اجمالى استوك']].sort_values('اجمالى استوك', ascending=False)

# --- Alerts ---
alerts_data = {
    'out_of_stock': [],
    'low_stock': [],
    'healthy_stock': []
}

LOW_STOCK_THRESHOLD = 5
OUT_OF_STOCK_THRESHOLD = 0

for _, row in df_sales.iterrows():
    product_name = row.get('اسم الصنف', '')
    product_code = str(row['الكود'])
    total_stock = row['اجمالى استوك']
    total_sales = row['اجمالى مبيعات']

    if total_stock <= OUT_OF_STOCK_THRESHOLD:
        status = 'نفد المخزون'
        color = 'red'
        alerts_data['out_of_stock'].append({
            'اسم المنتج': product_name,
            'الكود': product_code,
            'المخزون': total_stock,
            'المبيعات': total_sales,
            'الحالة': status,
            'اللون': color
        })
    elif total_stock <= LOW_STOCK_THRESHOLD:
        status = 'قرب ينفد'
        color = 'yellow'
        alerts_data['low_stock'].append({
            'اسم المنتج': product_name,
            'الكود': product_code,
            'المخزون': total_stock,
            'المبيعات': total_sales,
            'الحالة': status,
            'اللون': color
        })
    else:
        status = 'متوفر'
        color = 'green'
        alerts_data['healthy_stock'].append({
            'اسم المنتج': product_name,
            'الكود': product_code,
            'المخزون': total_stock,
            'المبيعات': total_sales,
            'الحالة': status,
            'اللون': color
        })

# --- Inventory Movement ---
inventory_movement = []
for name in SALES_BRANCHES:
    inbound = df_sales[f'{name}_استوك'].sum() + df_sales[f'{name}_مبيعات'].sum()
    outbound = df_sales[f'{name}_مبيعات'].sum()
    net_change = inbound - outbound

    inventory_movement.append({
        'الفرع': name,
        'الوارد': round(inbound, 1),
        'المنصرف': round(outbound, 1),
        'التغير الصافي': round(net_change, 1),
        'نسبة الاستهلاك': round(outbound / inbound * 100, 1) if inbound > 0 else 0
    })

# --- Summary KPIs ---
total_sales = df_sales['اجمالى مبيعات'].sum()
total_stock = df_sales['اجمالى استوك'].sum()
sell_through_rate = round(total_sales / (total_sales + total_stock) * 100, 1) if (total_sales + total_stock) > 0 else 0
best_branch = df_branch_perf.sort_values('نسبة البيع للمخزون', ascending=False).iloc[0]['الفرع'] if not df_branch_perf.empty else 'N/A'
worst_branch = df_branch_perf.sort_values('نسبة البيع للمخزون', ascending=True).iloc[0]['الفرع'] if not df_branch_perf.empty else 'N/A'

summary_kpis = {
    'اجمالى المبيعات_وحدة': round(total_sales, 1),
    'اجمالى المخزون_وحدة': round(total_stock, 1),
    'معدل البيع': sell_through_rate,
    'عدد الراكد': int(len(df_dead_stock)),
    'اجمالى الربح': round(df_branch_value['الربح'].sum(), 1),
    'أفضل فرع': best_branch,
    'أسوأ فرع': worst_branch,
}

log(f'  KPIs computed: {len(summary_kpis)} summary metrics')

# ============================================================
# SECTION 3: JSON EXPORT
# ============================================================
log('Exporting dashboard data to JSON...')

# Scatter data
df_scatter_export = df_sales[['اسم الصنف', 'اجمالى مبيعات', 'اجمالى استوك']].copy()
df_scatter_export = df_scatter_export[
    (df_scatter_export['اجمالى مبيعات'] > 0) | (df_scatter_export['اجمالى استوك'] > 0)
]
scatter_products = df_scatter_export.rename(columns={
    'اسم الصنف': 'name', 'اجمالى مبيعات': 'sales', 'اجمالى استوك': 'stock'
}).to_dict(orient='records')
scatter_med_sales = round(float(df_scatter_export['اجمالى مبيعات'].median()), 1)
scatter_med_stock = round(float(df_scatter_export['اجمالى استوك'].median()), 1)

# Per-item per-branch data
items_records = []
for _, row in df_sales.iterrows():
    item = {
        'code': str(row['الكود']),
        'name': str(row.get('اسم الصنف', '')),
        'total_sales': float(row['اجمالى مبيعات']),
        'total_stock': float(row['اجمالى استوك']),
    }

    category_columns = ['فئة', 'category', 'نوع', 'type', 'صنف', 'مجموعة']
    item_category = 'غير محدد'
    for col in category_columns:
        if col in df_sales.columns:
            item_category = str(row.get(col, 'غير محدد'))
            break
    item['category'] = item_category

    branch_performance = []
    best_branch_item = {'name': '', 'sell_through': 0}
    worst_branch_item = {'name': '', 'sell_through': 100}
    out_of_stock_branches = []
    low_stock_branches = []

    for branch in SALES_BRANCHES:
        sales = float(row.get(f'{branch}_مبيعات', 0))
        stock = float(row.get(f'{branch}_استوك', 0))

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
            best_branch_item = {'name': branch, 'sell_through': sell_through}
        if 0 < sell_through < worst_branch_item['sell_through']:
            worst_branch_item = {'name': branch, 'sell_through': sell_through}

        if sales > 0 and stock == 0:
            out_of_stock_branches.append(branch)
        elif 0 < stock <= 5:
            low_stock_branches.append(branch)

        item[f'{branch}_sales'] = sales
        item[f'{branch}_stock'] = stock
        item[f'{branch}_sell_through'] = sell_through
        item[f'{branch}_status'] = status
        item[f'{branch}_status_emoji'] = status_emoji

        branch_performance.append({
            'branch': branch,
            'sales': sales,
            'stock': stock,
            'sell_through': sell_through,
            'status': status,
            'status_emoji': status_emoji
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
log(f'  Items data exported: {len(items_records)} items with enhanced metrics')

dashboard_data = {
    'metadata': {
        'generated_at': datetime.datetime.now().isoformat(),
        'version': '3.0',
        'source': 'generate_dashboard.py',
    },
    'summary': summary_kpis,
    'branch_performance': df_branch_perf.to_dict(orient='records'),
    'branch_stock_value': df_branch_value.to_dict(orient='records'),
    'top_sellers': df_top_sellers.to_dict(orient='records'),
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
}

with open(os.path.join(OUTPUT_DIR, 'dashboard_data.json'), 'w', encoding='utf-8') as f:
    json.dump(dashboard_data, f, ensure_ascii=False, indent=2, default=str)

log('')
log('=' * 50)
log('Done! Output generated:')
log(f'  Dashboard Data: {os.path.join(OUTPUT_DIR, "dashboard_data.json")}')
log('=' * 50)
