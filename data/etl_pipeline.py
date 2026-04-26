"""
ETL Pipeline for Inventory Data Normalization
Transforms wide-format CSV data into normalized PostgreSQL schema (3NF)
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'postgres',
    'password': 'abdosaid200455',  # Change this to your PostgreSQL password
    'database': 'normalized_inventory_db'
}

CSV_FILES = {
    'sales_stock': 'cleaned_sales_stock.csv',
    'pricing': 'cleaned_pricing.csv',
    'suppliers': 'cleaned_suppliers.csv',
    'warehouse': 'cleaned_warehouse.csv'
}

# ============================================================================
# STEP 1: EXTRACT - Load CSV Files
# ============================================================================

def load_csv_files():
    """Load all CSV files with proper encoding"""
    print("=" * 80)
    print("STEP 1: EXTRACTING DATA FROM CSV FILES")
    print("=" * 80)

    dataframes = {}

    for key, filename in CSV_FILES.items():
        try:
            df = pd.read_csv(filename, encoding='utf-8-sig')
            # Remove BOM if present
            df.columns = df.columns.str.replace('﻿', '')
            dataframes[key] = df
            print(f"✓ Loaded {filename}: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"✗ Error loading {filename}: {e}")
            raise

    return dataframes

# ============================================================================
# STEP 2: TRANSFORM - Normalize Data
# ============================================================================

def clean_text(series):
    """Clean and normalize text data"""
    if series.dtype == 'object':
        return series.str.strip().replace('', None)
    return series

def extract_products(dataframes):
    """Extract and normalize products table"""
    print("\n" + "=" * 80)
    print("TRANSFORMING: products table")
    print("=" * 80)

    # Collect product info from all sources
    products_list = []

    # From sales_stock
    if 'sales_stock' in dataframes:
        df = dataframes['sales_stock']
        products_list.append(df[['الكود', 'اسم الصنف']].rename(columns={
            'الكود': 'product_id',
            'اسم الصنف': 'product_name'
        }))

    # From pricing
    if 'pricing' in dataframes:
        df = dataframes['pricing']
        products_list.append(df[['الكود', 'اسم الصنف', 'مجموعه الصنف']].rename(columns={
            'الكود': 'product_id',
            'اسم الصنف': 'product_name',
            'مجموعه الصنف': 'category_name'
        }))

    # From warehouse
    if 'warehouse' in dataframes:
        df = dataframes['warehouse']
        products_list.append(df[['الكود', 'اسم الصنف', 'مجموعة الصنف', 'المجموعه الرئيسيه', 'الوحدة']].rename(columns={
            'الكود': 'product_id',
            'اسم الصنف': 'product_name',
            'مجموعة الصنف': 'category_name',
            'المجموعه الرئيسيه': 'main_category_name',
            'الوحدة': 'unit'
        }))

    # From suppliers
    if 'suppliers' in dataframes:
        df = dataframes['suppliers']
        products_list.append(df[['الكود', 'اسم الصنف', 'المجموعة']].rename(columns={
            'الكود': 'product_id',
            'اسم الصنف': 'product_name',
            'المجموعة': 'category_name'
        }))

    # Combine all product data
    products = pd.concat(products_list, ignore_index=True)

    # Group by product_id and take first non-null value for each field
    products = products.groupby('product_id').agg({
        'product_name': 'first',
        'category_name': 'first',
        'main_category_name': 'first',
        'unit': 'first'
    }).reset_index()

    # Clean text fields
    for col in products.columns:
        products[col] = clean_text(products[col])

    # Remove duplicates
    products = products.drop_duplicates(subset=['product_id'])

    print(f"✓ Extracted {len(products)} unique products")
    return products

def extract_suppliers(dataframes):
    """Extract and normalize suppliers table"""
    print("\n" + "=" * 80)
    print("TRANSFORMING: suppliers table")
    print("=" * 80)

    df = dataframes['suppliers']

    # Extract unique suppliers
    suppliers = df[['المورد']].rename(columns={'المورد': 'supplier_name'})
    suppliers = suppliers.drop_duplicates()
    suppliers['supplier_name'] = clean_text(suppliers['supplier_name'])
    suppliers = suppliers[suppliers['supplier_name'].notna()]

    # Add supplier_id
    suppliers = suppliers.reset_index(drop=True)
    suppliers.insert(0, 'supplier_id', range(1, len(suppliers) + 1))

    print(f"✓ Extracted {len(suppliers)} unique suppliers")
    return suppliers

def extract_product_suppliers(dataframes, suppliers_df):
    """Extract product-supplier relationships"""
    print("\n" + "=" * 80)
    print("TRANSFORMING: product_suppliers table")
    print("=" * 80)

    df = dataframes['suppliers']

    # Map supplier names to IDs
    supplier_map = dict(zip(suppliers_df['supplier_name'], suppliers_df['supplier_id']))

    product_suppliers = df[['الكود', 'المورد']].rename(columns={
        'الكود': 'product_id',
        'المورد': 'supplier_name'
    })

    product_suppliers['product_id'] = clean_text(product_suppliers['product_id'])
    product_suppliers['supplier_name'] = clean_text(product_suppliers['supplier_name'])

    # Map to supplier_id
    product_suppliers['supplier_id'] = product_suppliers['supplier_name'].map(supplier_map)
    product_suppliers = product_suppliers[['product_id', 'supplier_id']].dropna()
    product_suppliers = product_suppliers.drop_duplicates()

    print(f"✓ Extracted {len(product_suppliers)} product-supplier relationships")
    return product_suppliers

def extract_branches(dataframes):
    """Extract unique branch names from column headers"""
    print("\n" + "=" * 80)
    print("TRANSFORMING: branches table")
    print("=" * 80)

    df = dataframes['sales_stock']

    # Extract branch names from columns ending with _مبيعات or _استوك
    branch_names = set()
    for col in df.columns:
        if col.endswith('_مبيعات'):
            branch_name = col.replace('_مبيعات', '')
            branch_names.add(branch_name)
        elif col.endswith('_استوك'):
            branch_name = col.replace('_استوك', '')
            branch_names.add(branch_name)

    branches = pd.DataFrame({
        'branch_name': sorted(list(branch_names))
    })
    branches.insert(0, 'branch_id', range(1, len(branches) + 1))

    print(f"✓ Extracted {len(branches)} unique branches")
    return branches

def extract_branch_sales(dataframes, branches_df):
    """Unpivot sales data from wide to long format"""
    print("\n" + "=" * 80)
    print("TRANSFORMING: branch_sales table (UNPIVOT)")
    print("=" * 80)

    df = dataframes['sales_stock']

    # Get sales columns
    sales_cols = [col for col in df.columns if col.endswith('_مبيعات')]

    # Prepare data for melting
    id_cols = ['الكود']
    melt_df = df[id_cols + sales_cols].copy()

    # Melt the dataframe
    sales_long = pd.melt(
        melt_df,
        id_vars=id_cols,
        value_vars=sales_cols,
        var_name='branch_column',
        value_name='quantity_sold'
    )

    # Extract branch name
    sales_long['branch_name'] = sales_long['branch_column'].str.replace('_مبيعات', '')

    # Rename and clean
    sales_long = sales_long.rename(columns={'الكود': 'product_id'})
    sales_long['product_id'] = clean_text(sales_long['product_id'])

    # Map to branch_id
    branch_map = dict(zip(branches_df['branch_name'], branches_df['branch_id']))
    sales_long['branch_id'] = sales_long['branch_name'].map(branch_map)

    # Select final columns and remove nulls
    sales_long = sales_long[['product_id', 'branch_id', 'quantity_sold']]
    sales_long = sales_long[sales_long['quantity_sold'].notna()]
    sales_long = sales_long[sales_long['product_id'].notna()]

    # Add sale_id
    sales_long = sales_long.reset_index(drop=True)
    sales_long.insert(0, 'sale_id', range(1, len(sales_long) + 1))

    print(f"✓ Unpivoted {len(sales_long)} sales records")
    return sales_long

def extract_branch_stock(dataframes, branches_df):
    """Unpivot stock data from wide to long format"""
    print("\n" + "=" * 80)
    print("TRANSFORMING: branch_stock table (UNPIVOT)")
    print("=" * 80)

    df = dataframes['sales_stock']

    # Get stock columns
    stock_cols = [col for col in df.columns if col.endswith('_استوك')]

    # Prepare data for melting
    id_cols = ['الكود']
    melt_df = df[id_cols + stock_cols].copy()

    # Melt the dataframe
    stock_long = pd.melt(
        melt_df,
        id_vars=id_cols,
        value_vars=stock_cols,
        var_name='branch_column',
        value_name='stock_quantity'
    )

    # Extract branch name
    stock_long['branch_name'] = stock_long['branch_column'].str.replace('_استوك', '')

    # Rename and clean
    stock_long = stock_long.rename(columns={'الكود': 'product_id'})
    stock_long['product_id'] = clean_text(stock_long['product_id'])

    # Map to branch_id
    branch_map = dict(zip(branches_df['branch_name'], branches_df['branch_id']))
    stock_long['branch_id'] = stock_long['branch_name'].map(branch_map)

    # Select final columns and remove nulls
    stock_long = stock_long[['product_id', 'branch_id', 'stock_quantity']]
    stock_long = stock_long[stock_long['stock_quantity'].notna()]
    stock_long = stock_long[stock_long['product_id'].notna()]

    # Add branch_stock_id
    stock_long = stock_long.reset_index(drop=True)
    stock_long.insert(0, 'branch_stock_id', range(1, len(stock_long) + 1))

    print(f"✓ Unpivoted {len(stock_long)} stock records")
    return stock_long

def extract_warehouses(dataframes):
    """Extract unique warehouses"""
    print("\n" + "=" * 80)
    print("TRANSFORMING: warehouses table")
    print("=" * 80)

    df = dataframes['warehouse']

    warehouses = df[['المخزن']].rename(columns={'المخزن': 'warehouse_name'})
    warehouses = warehouses.drop_duplicates()
    warehouses['warehouse_name'] = clean_text(warehouses['warehouse_name'])
    warehouses = warehouses[warehouses['warehouse_name'].notna()]

    warehouses = warehouses.reset_index(drop=True)
    warehouses.insert(0, 'warehouse_id', range(1, len(warehouses) + 1))

    print(f"✓ Extracted {len(warehouses)} unique warehouses")
    return warehouses

def extract_warehouse_stock(dataframes, warehouses_df):
    """Extract warehouse stock data"""
    print("\n" + "=" * 80)
    print("TRANSFORMING: warehouse_stock table")
    print("=" * 80)

    df = dataframes['warehouse']

    # Map warehouse names to IDs
    warehouse_map = dict(zip(warehouses_df['warehouse_name'], warehouses_df['warehouse_id']))

    warehouse_stock = df[['الكود', 'المخزن', 'الكمية']].rename(columns={
        'الكود': 'product_id',
        'المخزن': 'warehouse_name',
        'الكمية': 'quantity'
    })

    warehouse_stock['product_id'] = clean_text(warehouse_stock['product_id'])
    warehouse_stock['warehouse_name'] = clean_text(warehouse_stock['warehouse_name'])

    # Map to warehouse_id
    warehouse_stock['warehouse_id'] = warehouse_stock['warehouse_name'].map(warehouse_map)
    warehouse_stock = warehouse_stock[['product_id', 'warehouse_id', 'quantity']]
    warehouse_stock = warehouse_stock.dropna(subset=['product_id', 'warehouse_id'])

    warehouse_stock = warehouse_stock.reset_index(drop=True)
    warehouse_stock.insert(0, 'warehouse_stock_id', range(1, len(warehouse_stock) + 1))

    print(f"✓ Extracted {len(warehouse_stock)} warehouse stock records")
    return warehouse_stock

def extract_pricing(dataframes):
    """Extract pricing data"""
    print("\n" + "=" * 80)
    print("TRANSFORMING: pricing table")
    print("=" * 80)

    df = dataframes['pricing']

    pricing = df[['الكود', 'سعر التكلفه', 'سعر القطعة']].rename(columns={
        'الكود': 'product_id',
        'سعر التكلفه': 'cost_price',
        'سعر القطعة': 'item_price'
    })

    # Add unit_sale_price from warehouse data if available
    if 'warehouse' in dataframes:
        warehouse_pricing = dataframes['warehouse'][['الكود', 'سعر بيع الوحدة']].rename(columns={
            'الكود': 'product_id',
            'سعر بيع الوحدة': 'unit_sale_price'
        })
        pricing = pricing.merge(warehouse_pricing, on='product_id', how='left')

    pricing['product_id'] = clean_text(pricing['product_id'])
    pricing = pricing.dropna(subset=['product_id'])
    pricing = pricing.drop_duplicates(subset=['product_id'])

    pricing = pricing.reset_index(drop=True)
    pricing.insert(0, 'pricing_id', range(1, len(pricing) + 1))

    print(f"✓ Extracted {len(pricing)} pricing records")
    return pricing

# ============================================================================
# STEP 3: LOAD - Create Database and Load Data
# ============================================================================

def create_database():
    """Create PostgreSQL database"""
    print("\n" + "=" * 80)
    print("STEP 3: CREATING DATABASE")
    print("=" * 80)

    try:
        # Connect to default postgres database
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database='postgres'
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Drop database if exists
        cursor.execute(f"DROP DATABASE IF EXISTS {DB_CONFIG['database']}")
        print(f"✓ Dropped existing database (if any)")

        # Create database
        cursor.execute(f"CREATE DATABASE {DB_CONFIG['database']}")
        print(f"✓ Created database: {DB_CONFIG['database']}")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print(f"✗ Error creating database: {e}")
        return False

def create_tables(engine):
    """Create all tables with proper schema"""
    print("\n" + "=" * 80)
    print("CREATING TABLES")
    print("=" * 80)

    ddl_statements = [
        # Products table
        """
        CREATE TABLE products (
            product_id TEXT PRIMARY KEY,
            product_name TEXT,
            category_name TEXT,
            main_category_name TEXT,
            unit TEXT
        )
        """,

        # Suppliers table
        """
        CREATE TABLE suppliers (
            supplier_id SERIAL PRIMARY KEY,
            supplier_name TEXT UNIQUE NOT NULL
        )
        """,

        # Product-Supplier relationship
        """
        CREATE TABLE product_suppliers (
            product_id TEXT,
            supplier_id INTEGER,
            PRIMARY KEY (product_id, supplier_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
            FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id) ON DELETE CASCADE
        )
        """,

        # Branches table
        """
        CREATE TABLE branches (
            branch_id SERIAL PRIMARY KEY,
            branch_name TEXT UNIQUE NOT NULL
        )
        """,

        # Branch sales
        """
        CREATE TABLE branch_sales (
            sale_id SERIAL PRIMARY KEY,
            product_id TEXT,
            branch_id INTEGER,
            quantity_sold NUMERIC,
            FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
            FOREIGN KEY (branch_id) REFERENCES branches(branch_id) ON DELETE CASCADE
        )
        """,

        # Branch stock
        """
        CREATE TABLE branch_stock (
            branch_stock_id SERIAL PRIMARY KEY,
            product_id TEXT,
            branch_id INTEGER,
            stock_quantity NUMERIC,
            FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
            FOREIGN KEY (branch_id) REFERENCES branches(branch_id) ON DELETE CASCADE
        )
        """,

        # Warehouses table
        """
        CREATE TABLE warehouses (
            warehouse_id SERIAL PRIMARY KEY,
            warehouse_name TEXT UNIQUE NOT NULL
        )
        """,

        # Warehouse stock
        """
        CREATE TABLE warehouse_stock (
            warehouse_stock_id SERIAL PRIMARY KEY,
            product_id TEXT,
            warehouse_id INTEGER,
            quantity NUMERIC,
            FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
            FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id) ON DELETE CASCADE
        )
        """,

        # Pricing table
        """
        CREATE TABLE pricing (
            pricing_id SERIAL PRIMARY KEY,
            product_id TEXT,
            cost_price NUMERIC,
            item_price NUMERIC,
            unit_sale_price NUMERIC,
            FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE
        )
        """
    ]

    try:
        with engine.connect() as conn:
            for ddl in ddl_statements:
                conn.execute(text(ddl))
                conn.commit()
        print("✓ All tables created successfully")
        return True
    except Exception as e:
        print(f"✗ Error creating tables: {e}")
        return False

def create_indexes(engine):
    """Create indexes for better query performance"""
    print("\n" + "=" * 80)
    print("CREATING INDEXES")
    print("=" * 80)

    indexes = [
        "CREATE INDEX idx_product_suppliers_product ON product_suppliers(product_id)",
        "CREATE INDEX idx_product_suppliers_supplier ON product_suppliers(supplier_id)",
        "CREATE INDEX idx_branch_sales_product ON branch_sales(product_id)",
        "CREATE INDEX idx_branch_sales_branch ON branch_sales(branch_id)",
        "CREATE INDEX idx_branch_stock_product ON branch_stock(product_id)",
        "CREATE INDEX idx_branch_stock_branch ON branch_stock(branch_id)",
        "CREATE INDEX idx_warehouse_stock_product ON warehouse_stock(product_id)",
        "CREATE INDEX idx_warehouse_stock_warehouse ON warehouse_stock(warehouse_id)",
        "CREATE INDEX idx_pricing_product ON pricing(product_id)"
    ]

    try:
        with engine.connect() as conn:
            for idx in indexes:
                conn.execute(text(idx))
                conn.commit()
        print("✓ All indexes created successfully")
        return True
    except Exception as e:
        print(f"✗ Error creating indexes: {e}")
        return False

def load_data_to_db(engine, tables_data):
    """Load all transformed data into PostgreSQL"""
    print("\n" + "=" * 80)
    print("LOADING DATA INTO DATABASE")
    print("=" * 80)

    load_order = [
        'products',
        'suppliers',
        'product_suppliers',
        'branches',
        'branch_sales',
        'branch_stock',
        'warehouses',
        'warehouse_stock',
        'pricing'
    ]

    for table_name in load_order:
        if table_name in tables_data:
            df = tables_data[table_name]
            try:
                df.to_sql(table_name, engine, if_exists='append', index=False)
                print(f"✓ Loaded {len(df)} rows into {table_name}")
            except Exception as e:
                print(f"✗ Error loading {table_name}: {e}")
                raise

    print("\n✓ All data loaded successfully")

# ============================================================================
# STEP 4: VALIDATION
# ============================================================================

def validate_data(engine):
    """Validate loaded data"""
    print("\n" + "=" * 80)
    print("STEP 4: VALIDATING DATA")
    print("=" * 80)

    validation_queries = {
        'products': "SELECT COUNT(*) FROM products",
        'suppliers': "SELECT COUNT(*) FROM suppliers",
        'product_suppliers': "SELECT COUNT(*) FROM product_suppliers",
        'branches': "SELECT COUNT(*) FROM branches",
        'branch_sales': "SELECT COUNT(*) FROM branch_sales",
        'branch_stock': "SELECT COUNT(*) FROM branch_stock",
        'warehouses': "SELECT COUNT(*) FROM warehouses",
        'warehouse_stock': "SELECT COUNT(*) FROM warehouse_stock",
        'pricing': "SELECT COUNT(*) FROM pricing"
    }

    results = {}
    with engine.connect() as conn:
        for table, query in validation_queries.items():
            result = conn.execute(text(query))
            count = result.scalar()
            results[table] = count
            print(f"  {table}: {count:,} rows")

    # Check for orphan records
    print("\n" + "-" * 80)
    print("Checking referential integrity...")
    print("-" * 80)

    integrity_checks = [
        ("product_suppliers", "SELECT COUNT(*) FROM product_suppliers ps LEFT JOIN products p ON ps.product_id = p.product_id WHERE p.product_id IS NULL"),
        ("branch_sales", "SELECT COUNT(*) FROM branch_sales bs LEFT JOIN products p ON bs.product_id = p.product_id WHERE p.product_id IS NULL"),
        ("branch_stock", "SELECT COUNT(*) FROM branch_stock bs LEFT JOIN products p ON bs.product_id = p.product_id WHERE p.product_id IS NULL"),
        ("warehouse_stock", "SELECT COUNT(*) FROM warehouse_stock ws LEFT JOIN products p ON ws.product_id = p.product_id WHERE p.product_id IS NULL"),
        ("pricing", "SELECT COUNT(*) FROM pricing pr LEFT JOIN products p ON pr.product_id = p.product_id WHERE p.product_id IS NULL")
    ]

    all_valid = True
    with engine.connect() as conn:
        for table, query in integrity_checks:
            result = conn.execute(text(query))
            orphans = result.scalar()
            if orphans > 0:
                print(f"  ✗ {table}: {orphans} orphan records found")
                all_valid = False
            else:
                print(f"  ✓ {table}: No orphan records")

    if all_valid:
        print("\n✓ All referential integrity checks passed")
    else:
        print("\n✗ Some referential integrity issues found")

    return results

# ============================================================================
# STEP 5: EXPORT DATABASE
# ============================================================================

def export_database():
    """Export PostgreSQL database as backup file"""
    print("\n" + "=" * 80)
    print("STEP 5: EXPORTING DATABASE BACKUP")
    print("=" * 80)

    import subprocess
    import os

    db_name = DB_CONFIG['database']
    user = DB_CONFIG['user']
    host = DB_CONFIG['host']
    port = DB_CONFIG['port']

    # Set password environment variable
    env = os.environ.copy()
    env['PGPASSWORD'] = DB_CONFIG['password']

    # Try custom format first
    backup_file = f"{db_name}.backup"
    try:
        cmd = [
            'pg_dump',
            '-U', user,
            '-h', host,
            '-p', str(port),
            '-F', 'c',
            '-b',
            '-v',
            '-f', backup_file,
            db_name
        ]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Database exported to: {backup_file}")
            return backup_file
        else:
            print(f"Custom format failed: {result.stderr}")
    except Exception as e:
        print(f"Custom format error: {e}")

    # Fallback to SQL format
    backup_file = f"{db_name}.sql"
    try:
        cmd = [
            'pg_dump',
            '-U', user,
            '-h', host,
            '-p', str(port),
            '-f', backup_file,
            db_name
        ]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Database exported to: {backup_file}")
            return backup_file
        else:
            print(f"✗ SQL format failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"✗ SQL format error: {e}")
        return None

# ============================================================================
# STEP 6: GENERATE SUMMARY REPORT
# ============================================================================

def generate_summary_report(validation_results, assumptions, cleaning_steps):
    """Generate final summary report"""
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY REPORT")
    print("=" * 80)

    report = []
    report.append("=" * 80)
    report.append("ETL PIPELINE EXECUTION SUMMARY")
    report.append("=" * 80)
    report.append("")
    report.append(f"Database: {DB_CONFIG['database']}")
    report.append(f"Execution Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    report.append("-" * 80)
    report.append("TABLE ROW COUNTS")
    report.append("-" * 80)
    for table, count in validation_results.items():
        report.append(f"  {table:.<30} {count:>10,} rows")
    report.append("")

    report.append("-" * 80)
    report.append("DATA CLEANING STEPS")
    report.append("-" * 80)
    for step in cleaning_steps:
        report.append(f"  • {step}")
    report.append("")

    report.append("-" * 80)
    report.append("ASSUMPTIONS MADE")
    report.append("-" * 80)
    for assumption in assumptions:
        report.append(f"  • {assumption}")
    report.append("")

    report.append("-" * 80)
    report.append("SCHEMA DESIGN")
    report.append("-" * 80)
    report.append("  • Normalized to 3NF (Third Normal Form)")
    report.append("  • Wide-format branch columns unpivoted to long format")
    report.append("  • Foreign key constraints enforced")
    report.append("  • Indexes created on foreign keys for performance")
    report.append("  • All table/column names in English (snake_case)")
    report.append("  • Data values preserved in Arabic")
    report.append("")

    report.append("=" * 80)
    report.append("ETL PIPELINE COMPLETED SUCCESSFULLY")
    report.append("=" * 80)

    report_text = "\n".join(report)
    print(report_text)

    # Save to file
    with open('etl_summary_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print("\n✓ Summary report saved to: etl_summary_report.txt")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main ETL pipeline execution"""
    print("\n")
    print("=" * 80)
    print("INVENTORY DATA NORMALIZATION ETL PIPELINE")
    print("=" * 80)
    print("")

    # Track assumptions and cleaning steps
    assumptions = [
        "Product ID (الكود) is the primary identifier across all sources",
        "Missing values are converted to NULL in database",
        "Duplicate products are merged, taking first non-null value for each field",
        "Branch names extracted from column suffixes (_مبيعات, _استوك)",
        "Zero values in sales/stock are preserved (not treated as NULL)",
        "Pricing data merged from both pricing and warehouse files",
        "Text fields are trimmed of whitespace",
        "Empty strings converted to NULL"
    ]

    cleaning_steps = [
        "Removed BOM (Byte Order Mark) from CSV headers",
        "Trimmed whitespace from all text fields",
        "Converted empty strings to NULL",
        "Removed duplicate products based on product_id",
        "Unpivoted wide-format branch columns to long format",
        "Mapped branch names and warehouse names to integer IDs",
        "Validated foreign key relationships",
        "Created indexes on foreign key columns"
    ]

    try:
        # Step 1: Extract
        dataframes = load_csv_files()

        # Step 2: Transform
        print("\n" + "=" * 80)
        print("STEP 2: TRANSFORMING DATA")
        print("=" * 80)

        products = extract_products(dataframes)
        suppliers = extract_suppliers(dataframes)
        product_suppliers = extract_product_suppliers(dataframes, suppliers)
        branches = extract_branches(dataframes)
        branch_sales = extract_branch_sales(dataframes, branches)
        branch_stock = extract_branch_stock(dataframes, branches)
        warehouses = extract_warehouses(dataframes)
        warehouse_stock = extract_warehouse_stock(dataframes, warehouses)
        pricing = extract_pricing(dataframes)

        tables_data = {
            'products': products,
            'suppliers': suppliers,
            'product_suppliers': product_suppliers,
            'branches': branches,
            'branch_sales': branch_sales,
            'branch_stock': branch_stock,
            'warehouses': warehouses,
            'warehouse_stock': warehouse_stock,
            'pricing': pricing
        }

        # Step 3: Load
        if not create_database():
            raise Exception("Failed to create database")

        # Create engine
        connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        engine = create_engine(connection_string)

        if not create_tables(engine):
            raise Exception("Failed to create tables")

        load_data_to_db(engine, tables_data)

        if not create_indexes(engine):
            raise Exception("Failed to create indexes")

        # Step 4: Validate
        validation_results = validate_data(engine)

        # Step 5: Export
        backup_file = export_database()

        # Step 6: Generate Report
        generate_summary_report(validation_results, assumptions, cleaning_steps)

        print("\n" + "=" * 80)
        print("SUCCESS! ETL PIPELINE COMPLETED")
        print("=" * 80)
        print(f"\nDatabase: {DB_CONFIG['database']}")
        if backup_file:
            print(f"Backup file: {backup_file}")
        print(f"Summary report: etl_summary_report.txt")
        print("\nTo restore the database:")
        if backup_file and backup_file.endswith('.backup'):
            print(f"  pg_restore -U postgres -d {DB_CONFIG['database']} {backup_file}")
        elif backup_file:
            print(f"  psql -U postgres -d {DB_CONFIG['database']} -f {backup_file}")

    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR: ETL PIPELINE FAILED")
        print("=" * 80)
        print(f"\n{str(e)}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    main()
