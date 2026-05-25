import os
from typing import List,Optional
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

class SupabaseProductDB:
    """Manager for product database operations using Supabase"""
    
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
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
                    limit: Optional[int] = None,
                    batch_size: int = 1000) -> List[str]:
        """
        Get all product names from the database with pagination support.
    
        Args:
            table_name: Name of the table containing products
            product_name_column: Column name that contains product names
            limit: Optional limit on number of products returned (None = get all)
            batch_size: Number of records to fetch per request (default 1000)
        
        Returns:
            List of product names
        """
        try:
            if limit is not None and limit <= batch_size:
                query = self.client.table(table_name).select(product_name_column).limit(limit)
                response = query.execute()
            
                if response.data:
                    products = [row.get(product_name_column) for row in response.data 
                               if row.get(product_name_column)]
                    products = [str(p).strip() for p in products if p and str(p).strip()]
                    print(f"📦 Loaded {len(products)} products from Supabase table '{table_name}'")
                    return products
                return []
        
            all_products = []
            start = 0
            page_count = 0
        
            print(f"🔄 Fetching products from '{table_name}' in batches of {batch_size}...")
        
            while True:
                end = start + batch_size - 1
                response = self.client.table(table_name)\
                    .select(product_name_column)\
                    .range(start, end)\
                    .execute()
            
                if not response.data:
                    break
            
                batch_products = [
                    row.get(product_name_column) for row in response.data 
                    if row.get(product_name_column)
                ]
            
                batch_products = [
                    str(p).strip() for p in batch_products 
                    if p and str(p).strip()
                ]
            
                all_products.extend(batch_products)
                page_count += 1
            
                if len(response.data) < batch_size:
                    break
                if limit is not None and len(all_products) >= limit:
                    all_products = all_products[:limit]
                    break
                start += batch_size
            
                if page_count % 10 == 0:
                    print(f"   ... fetched {len(all_products)} products so far")

            if limit is not None and len(all_products) > limit:
                all_products = all_products[:limit]
        
            print(f"📦 Loaded {len(all_products)} products from Supabase table '{table_name}' (fetched in {page_count} batches)")
            if all_products:
                print(f"📋 Sample: {all_products[:3]}")
        
            return all_products
        
        except Exception as e:
            print(f"❌ Error fetching products from Supabase: {e}")
            return []

_supabase_instance = None

def get_supabase_client() -> SupabaseProductDB:
    """Get or create Supabase client instance"""
    global _supabase_instance
    if _supabase_instance is None:
        _supabase_instance = SupabaseProductDB()
    return _supabase_instance
