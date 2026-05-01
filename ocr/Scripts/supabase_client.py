"""
Supabase Database Client for Product Management
"""
import os
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

class SupabaseProductDB:
    """Manager for product database operations using Supabase"""
    
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """
        Initialize Supabase client.
        
        Args:
            url: Supabase project URL
            key: Supabase API key (anon or service_role)
        """
        self.url = url if url is not None else os.getenv("SUPABASE_URL")
        self.key = key if key is not None else os.getenv("SUPABASE_KEY")
        
        print(f"🔧 SUPABASE_URL loaded: {'✅' if self.url else '❌'}")
        print(f"🔧 SUPABASE_KEY loaded: {'✅' if self.key else '❌'}")
        
        self.client: Optional[Client] = None
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to Supabase"""
        if not self.url or not self.key:
            print("❌ Missing Supabase credentials. Please check your .env file")
            print("   Expected variables: SUPABASE_URL and SUPABASE_KEY")
            raise ValueError("supabase_url and supabase_key are required")
        
        try:
            self.client = create_client(self.url, self.key)
            print("✅ Connected to Supabase successfully")
        except Exception as e:
            print(f"❌ Failed to connect to Supabase: {e}")
            raise
    
    def get_all_products(self, 
                        table_name: str = "products", 
                        product_name_column: str = "product_name",
                        limit: Optional[int] = None) -> List[str]:
        """
        Get all product names from the database.
        
        Args:
            table_name: Name of the table containing products
            product_name_column: Column name that contains product names
            limit: Optional limit on number of products returned
            
        Returns:
            List of product names
        """
        try:
            query = self.client.table(table_name).select(product_name_column)
            
            if limit is not None:
                query = query.limit(limit)
            
            response = query.execute()
            
            if response.data:
                products = [row.get(product_name_column) for row in response.data 
                           if row.get(product_name_column)]
                products = [str(p).strip() for p in products if p and str(p).strip()]
                print(f"📦 Loaded {len(products)} products from Supabase table '{table_name}'")
                if products:
                    print(f"📋 Sample: {products[:3]}")
                return products
            else:
                print(f"⚠️ No products found in table '{table_name}'")
                return []
                
        except Exception as e:
            print(f"❌ Error fetching products from Supabase: {e}")
            return []
    
    def get_products_by_category(self, 
                                category: Optional[str] = None,
                                table_name: str = "products",
                                product_name_column: str = "product_name",
                                category_column: str = "category") -> List[str]:
        """
        Get products filtered by category.
        
        Args:
            category: Category to filter by
            table_name: Name of the table
            product_name_column: Column with product names
            category_column: Column with categories
            
        Returns:
            List of product names in the specified category
        """
        try:
            query = self.client.table(table_name).select(product_name_column)
            
            if category is not None:
                query = query.eq(category_column, category)
            
            response = query.execute()
            
            if response.data:
                products = [row.get(product_name_column) for row in response.data 
                           if row.get(product_name_column)]
                products = [str(p).strip() for p in products if p and str(p).strip()]
                print(f"📦 Loaded {len(products)} products from category '{category}'")
                return products
            return []
            
        except Exception as e:
            print(f"❌ Error fetching products by category: {e}")
            return []
    
    def search_products(self, 
                       search_term: Optional[str] = None,
                       table_name: str = "products",
                       product_name_column: str = "product_name",
                       limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for products by name (case-insensitive).
        
        Args:
            search_term: Text to search for
            table_name: Name of the table
            product_name_column: Column with product names
            limit: Maximum number of results
            
        Returns:
            List of matching products with details
        """
        try:
            query = self.client.table(table_name).select("*")
            
            if search_term is not None:
                query = query.ilike(product_name_column, f"%{search_term}%")
            
            response = query.limit(limit).execute()
            
            if response.data:
                print(f"🔍 Found {len(response.data)} products matching '{search_term}'")
                return response.data
            return []
            
        except Exception as e:
            print(f"❌ Error searching products: {e}")
            return []
    
    def add_product(self, 
                   product_name: str,
                   table_name: str = "products",
                   additional_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a new product to the database.
        
        Args:
            product_name: Name of the product to add
            table_name: Name of the table
            additional_data: Additional fields to insert
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.product_name_column:
                self._detect_columns(table_name)
            
            column_name = self.product_name_column if self.product_name_column else "product_name"
            
            data = {column_name: product_name}
            if additional_data is not None:
                data.update(additional_data)
            
            response = self.client.table(table_name).insert(data).execute()
            
            if response.data:
                print(f"✅ Added product '{product_name}' to Supabase")
                return True
            return False
            
        except Exception as e:
            print(f"❌ Error adding product: {e}")
            return False
    
    def update_product(self, 
                      product_id: str,
                      updates: Dict[str, Any],
                      table_name: str = "products") -> bool:
        """
        Update an existing product.
        
        Args:
            product_id: ID of the product to update
            updates: Dictionary of fields to update
            table_name: Name of the table
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.client.table(table_name).update(updates).eq("id", product_id).execute()
            
            if response.data:
                print(f"✅ Updated product {product_id}")
                return True
            return False
            
        except Exception as e:
            print(f"❌ Error updating product: {e}")
            return False
    
    def get_table_info(self, table_name: str = "products") -> Dict[str, Any]:
        """
        Get information about a table (columns, row count).
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with table information
        """
        try:
            count_response = self.client.table(table_name).select("*", count="exact").limit(0).execute()
            row_count = count_response.count if hasattr(count_response, 'count') else 0
            
            sample_response = self.client.table(table_name).select("*").limit(1).execute()
            
            columns = []
            if sample_response.data and len(sample_response.data) > 0:
                columns = list(sample_response.data[0].keys())
            
            return {
                "table_name": table_name,
                "row_count": row_count,
                "columns": columns,
                "exists": True
            }
            
        except Exception as e:
            print(f"❌ Error getting table info: {e}")
            return {
                "table_name": table_name,
                "exists": False,
                "error": str(e)
            }
    
    def _detect_columns(self, table_name: str = "products") -> None:
        """
        Detect available columns in the products table.
        """
        try:
            response = self.client.table(table_name).select("*").limit(1).execute()
            if response.data and len(response.data) > 0:
                self.available_columns = list(response.data[0].keys())
                print(f"📊 Detected columns: {self.available_columns}")
                
                possible_name_columns = ['product_name', 'name', 'product', 'title', 'اسم', 'اسم_المنتج', 'name_ar']
                self.product_name_column = None
                
                for col in possible_name_columns:
                    if col in self.available_columns:
                        self.product_name_column = col
                        print(f"✅ Using '{col}' as product name column")
                        break
                
                if not self.product_name_column and self.available_columns:
                    self.product_name_column = self.available_columns[0]
                    print(f"⚠️ Using '{self.product_name_column}' as fallback product name column")
            else:
                self.available_columns = []
                self.product_name_column = "product_name"
                print("⚠️ No data found in products table, using default column name 'product_name'")
                
        except Exception as e:
            print(f"❌ Error detecting columns: {e}")
            self.available_columns = []
            self.product_name_column = "product_name"
    
    def close(self) -> None:
        """Close the connection"""
        print("🔌 Supabase connection closed")

_supabase_instance = None

def get_supabase_client() -> SupabaseProductDB:
    """Get or create Supabase client instance"""
    global _supabase_instance
    if _supabase_instance is None:
        _supabase_instance = SupabaseProductDB()
    return _supabase_instance
