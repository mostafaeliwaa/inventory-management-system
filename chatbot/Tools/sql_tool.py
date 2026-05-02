from typing import List, Tuple
from .tools_base import Tool
from chatbot.DB_Management import DatabaseManager

class SQLExecutionTool(Tool):
    
    
    def __init__(self, db_manager: 'DatabaseManager'):
        super().__init__(
            name="run_sql",
            description="Execute SQL SELECT query on inventory database"
        )
        self.db_manager = db_manager
    
    def execute(self, query: str) -> List[Tuple]:

        if not query.lower().startswith("select"):
            return []
        
        print(f" TOOL EXECUTION: {self.name}")
        print(f" SQL: {query}")
        
        try:
    
            result = self.db_manager.execute_query(query)
            print(f"RAW: {result}")
            return result
        except Exception as e:
            print(f"Supabase Error: {e}")
            return [{"error": str(e)}]