from langchain_google_genai import ChatGoogleGenerativeAI
from config import get_settings
from langchain_core.messages import HumanMessage
from typing import Dict, Any, TypedDict, List
from models import LLMManager
from DB_Management import DatabaseManager
from models import PromptTemplates
from Tools import SQLExecutionTool


settings = get_settings()


llm = ChatGoogleGenerativeAI(
    model=settings.LLM_MODEL,
    temperature=settings.LLM_TEMPERATURE,
    google_api_key=settings.GOOGLE_API_KEY,
)



class State(TypedDict):
    question: str
    type: str
    query: str
    result: List[Any]
    final_answer: str


class GraphNodes:
    
    
    def __init__(self, llm_manager: LLMManager, db_manager: DatabaseManager):
        self.llm = llm_manager
        self.sql_tool = SQLExecutionTool(db_manager)
    
    def process_llm_node(self, state: State) -> Dict[str, Any]:
    
        question = state["question"]
        prompt = PromptTemplates.SQL_GENERATION.format(question=question)
        
        response = self.llm.invoke(prompt)
        print(f" LLM: {response}")
        
        data = self.llm.parse_json_response(response)
        
        return {
            "type": data.get("type", "general"),
            "query": data.get("query", "")
        }
    
    def process_tool_node(self, state: State) -> Dict[str, Any]:
    
        query = state["query"]
        
        # Execute the SQL tool
        result = self.sql_tool.execute(query)
        
        return {"result": result}
    
    def process_format_node(self, state: State) -> Dict[str, Any]:
        
        question = state["question"]
        result = state.get("result", [])
        
        # Check for errors
        if self._is_error_result(result):
            return {"final_answer": "حصل خطأ أثناء تنفيذ الاستعلام."}
        
        if not result:
            return {"final_answer": "مفيش نتائج."}
        
        rows_text = "\n".join([", ".join(map(str, row)) for row in result])
        prompt = PromptTemplates.FORMAT_ANSWER.format(
            question=question,
            rows_text=rows_text
        )
        
        answer = self.llm.invoke(prompt)
        return {"final_answer": answer if answer else "حصل خطأ في الإجابة."}
    
    def process_general_node(self, state: State) -> Dict[str, Any]:
        
        question = state["question"]
        prompt = PromptTemplates.GENERAL_RESPONSE.format(question=question)
        
        answer = self.llm.invoke(prompt)
        return {"final_answer": answer if answer else "مش قادر أرد دلوقتي."}
    
    @staticmethod
    def _is_error_result(result: List[Any]) -> bool:

        if not isinstance(result, list) or not result:
            return False
        return isinstance(result[0], dict) and "error" in result[0]

