from supabase import Client, create_client
from typing import List, Tuple

class DatabaseManager:

    
    def __init__(self, url: str, key: str):
        self.url = url
        self.key = key
        self.client: Client = None
        self._connect()
    
    def _connect(self) -> None:
        
        self.client = create_client(self.url, self.key)
        print("Connected to Supabase")
    
    def execute_query(self, query: str):
        if not query.lower().strip().startswith("select"):
            return []

        try:
            response = self.client.rpc(
                "execute_select",
                {"query": query}
            ).execute()

            if response.data:
                result = []
                for row in response.data:
                    result.append(tuple(row.values()))
                return result

            return []

        except Exception as e:
            print(f" Supabase Query Error: {e}")
            return [{"error": str(e)}]
    
    def close(self) -> None:

        print(" Supabase connection closed")