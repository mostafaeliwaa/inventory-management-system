-- ============================================================================
-- NORMALIZED INVENTORY DATABASE SCHEMA (3NF)
-- Database: normalized_inventory_db
-- ============================================================================

-- ============================================================================
-- TABLE: products
-- Description: Master product catalog with category and unit information
-- ============================================================================
CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    product_name TEXT,
    category_name TEXT,
    main_category_name TEXT,
    unit TEXT
);

-- ============================================================================
-- TABLE: suppliers
-- Description: Unique supplier entities
-- ============================================================================
CREATE TABLE suppliers (
    supplier_id SERIAL PRIMARY KEY,
    supplier_name TEXT UNIQUE NOT NULL
);

-- ============================================================================
-- TABLE: product_suppliers
-- Description: Many-to-many relationship between products and suppliers
-- ============================================================================
CREATE TABLE product_suppliers (
    product_id TEXT,
    supplier_id INTEGER,
    PRIMARY KEY (product_id, supplier_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
    FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id) ON DELETE CASCADE
);

-- ============================================================================
-- TABLE: branches
-- Description: Store branches/locations
-- ============================================================================
CREATE TABLE branches (
    branch_id SERIAL PRIMARY KEY,
    branch_name TEXT UNIQUE NOT NULL
);

-- ============================================================================
-- TABLE: branch_sales
-- Description: Sales data per product per branch (unpivoted from wide format)
-- ============================================================================
CREATE TABLE branch_sales (
    sale_id SERIAL PRIMARY KEY,
    product_id TEXT,
    branch_id INTEGER,
    quantity_sold NUMERIC,
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
    FOREIGN KEY (branch_id) REFERENCES branches(branch_id) ON DELETE CASCADE
);

-- ============================================================================
-- TABLE: branch_stock
-- Description: Stock levels per product per branch (unpivoted from wide format)
-- ============================================================================
CREATE TABLE branch_stock (
    branch_stock_id SERIAL PRIMARY KEY,
    product_id TEXT,
    branch_id INTEGER,
    stock_quantity NUMERIC,
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
    FOREIGN KEY (branch_id) REFERENCES branches(branch_id) ON DELETE CASCADE
);

-- ============================================================================
-- TABLE: warehouses
-- Description: Warehouse/storage locations
-- ============================================================================
CREATE TABLE warehouses (
    warehouse_id SERIAL PRIMARY KEY,
    warehouse_name TEXT UNIQUE NOT NULL
);

-- ============================================================================
-- TABLE: warehouse_stock
-- Description: Stock quantities per product per warehouse
-- ============================================================================
CREATE TABLE warehouse_stock (
    warehouse_stock_id SERIAL PRIMARY KEY,
    product_id TEXT,
    warehouse_id INTEGER,
    quantity NUMERIC,
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id) ON DELETE CASCADE
);

-- ============================================================================
-- TABLE: pricing
-- Description: Product pricing information (cost, item price, unit sale price)
-- ============================================================================
CREATE TABLE pricing (
    pricing_id SERIAL PRIMARY KEY,
    product_id TEXT,
    cost_price NUMERIC,
    item_price NUMERIC,
    unit_sale_price NUMERIC,
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================
CREATE INDEX idx_product_suppliers_product ON product_suppliers(product_id);
CREATE INDEX idx_product_suppliers_supplier ON product_suppliers(supplier_id);
CREATE INDEX idx_branch_sales_product ON branch_sales(product_id);
CREATE INDEX idx_branch_sales_branch ON branch_sales(branch_id);
CREATE INDEX idx_branch_stock_product ON branch_stock(product_id);
CREATE INDEX idx_branch_stock_branch ON branch_stock(branch_id);
CREATE INDEX idx_warehouse_stock_product ON warehouse_stock(product_id);
CREATE INDEX idx_warehouse_stock_warehouse ON warehouse_stock(warehouse_id);
CREATE INDEX idx_pricing_product ON pricing(product_id);

-- ============================================================================
-- USEFUL QUERIES
-- ============================================================================

-- Get product with all its information
-- SELECT p.*, pr.cost_price, pr.item_price, pr.unit_sale_price
-- FROM products p
-- LEFT JOIN pricing pr ON p.product_id = pr.product_id
-- WHERE p.product_id = 'YOUR_PRODUCT_ID';

-- Get sales by branch
-- SELECT b.branch_name, SUM(bs.quantity_sold) as total_sales
-- FROM branch_sales bs
-- JOIN branches b ON bs.branch_id = b.branch_id
-- GROUP BY b.branch_name
-- ORDER BY total_sales DESC;

-- Get stock levels by branch
-- SELECT b.branch_name, p.product_name, bst.stock_quantity
-- FROM branch_stock bst
-- JOIN branches b ON bst.branch_id = b.branch_id
-- JOIN products p ON bst.product_id = p.product_id
-- WHERE bst.stock_quantity > 0
-- ORDER BY b.branch_name, p.product_name;

-- Get products by supplier
-- SELECT s.supplier_name, p.product_name, p.category_name
-- FROM product_suppliers ps
-- JOIN suppliers s ON ps.supplier_id = s.supplier_id
-- JOIN products p ON ps.product_id = p.product_id
-- ORDER BY s.supplier_name, p.product_name;
