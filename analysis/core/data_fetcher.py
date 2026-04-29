import pandas as pd
from .supabase_client import get_client

PAGE_SIZE = 1000


def _fetch_all(table: str, select: str = '*', order_col: str | None = None) -> list[dict]:
    client = get_client()
    rows = []
    offset = 0
    while True:
        q = client.table(table).select(select).range(offset, offset + PAGE_SIZE - 1)
        if order_col:
            q = q.order(order_col)
        res = q.execute()
        batch = res.data
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return rows


def fetch_branches() -> pd.DataFrame:
    return pd.DataFrame(_fetch_all('branches', order_col='branch_id'))


def fetch_branch_sales() -> pd.DataFrame:
    return pd.DataFrame(_fetch_all('branch_sales', order_col='sale_id'))


def fetch_branch_stock() -> pd.DataFrame:
    return pd.DataFrame(_fetch_all('branch_stock', order_col='branch_stock_id'))


def fetch_products() -> pd.DataFrame:
    return pd.DataFrame(_fetch_all('products', order_col='product_id'))


def fetch_pricing() -> pd.DataFrame:
    return pd.DataFrame(_fetch_all('pricing', order_col='pricing_id'))


def fetch_suppliers() -> pd.DataFrame:
    return pd.DataFrame(_fetch_all('suppliers', order_col='supplier_id'))


def fetch_product_suppliers() -> pd.DataFrame:
    return pd.DataFrame(_fetch_all('product_suppliers'))


def fetch_warehouses() -> pd.DataFrame:
    return pd.DataFrame(_fetch_all('warehouses', order_col='warehouse_id'))


def fetch_warehouse_stock() -> pd.DataFrame:
    return pd.DataFrame(_fetch_all('warehouse_stock', order_col='warehouse_stock_id'))
