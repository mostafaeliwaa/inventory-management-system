from config import get_settings
from .nodes import GraphNodes
from .graph import GraphBuilder
from DB_Management import DatabaseManager
from models import LLMManager
from Tools import SQLExecutionTool

settings = get_settings()

class InventoryAssistant:
    
    def __init__(self):
        self.db_manager = DatabaseManager(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        self.sql_tool = SQLExecutionTool(self.db_manager)
        self.llm_manager = LLMManager()
        self.nodes = GraphNodes(self.llm_manager, self.db_manager)
        self.graph_builder = GraphBuilder(self.nodes)
        self.app = self.graph_builder.build()
    
    def run(self) -> None:
        """Run the interactive assistant"""
        print(" Inventory Assistant Started (type 'exit' to quit)")
        print(f" Available Tool: {self.sql_tool.name} - {self.sql_tool.description}")
        print()
        
        while True:
            question = input(" اسألي: ")
            
            if question.lower() == "exit":
                break
            
            result = self.app.invoke({"question": question})
            print(f" {result['final_answer']}")
        
        self.cleanup()
    
    def cleanup(self) -> None:
        
        self.db_manager.close()
        
    def get_app():
        assistant = InventoryAssistant()
        return assistant.app.with_config({"run_name": "inventory-agent"})


