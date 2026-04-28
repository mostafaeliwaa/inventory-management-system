from langchain_google_genai import ChatGoogleGenerativeAI
from config import get_settings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
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
    user_id: str
    question: str
    memory: List[BaseMessage]
    type: str
    query: str
    result: List[Any]
    final_answer: str


class GraphNodes:


    def __init__(self, llm_manager: LLMManager, db_manager: DatabaseManager):
        self.llm = llm_manager
        self.sql_tool = SQLExecutionTool(db_manager)
        self.db_manager = db_manager

    async def load_memory_node(self, state: State) -> Dict[str, Any]:
        """
        Load conversation memory from database.
        Fetches last 10 messages, extracts current question from latest human message,
        and builds memory from previous messages.
        """
        user_id = state["user_id"]

        messages = await self.db_manager.get_last_messages(user_id, limit=10)

        if not messages:
            return {
                "question": "",
                "memory": []
            }

        # Messages come in DESC order (newest first), reverse to chronological
        messages = list(reversed(messages))

        # Latest human message becomes the current question
        latest_human_msg = None
        memory_messages = []

        for msg in messages:
            if msg["role"] == "human":
                latest_human_msg = msg
            elif msg["role"] == "ai":
                memory_messages.append(AIMessage(content=msg["content"]))

        # Remove the latest human message from memory (it's the current question)
        if latest_human_msg:
            # Build memory from all messages except the latest human one
            memory_messages = []
            for msg in messages:
                if msg["id"] == latest_human_msg["id"]:
                    continue
                if msg["role"] == "human":
                    memory_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "ai":
                    memory_messages.append(AIMessage(content=msg["content"]))

        current_question = latest_human_msg["content"] if latest_human_msg else ""

        return {
            "question": current_question,
            "memory": memory_messages
        }

    def process_llm_node(self, state: State) -> Dict[str, Any]:

        question = state["question"]
        memory = state.get("memory", [])

        # Format memory for prompt
        memory_text = ""
        if memory:
            memory_text = "\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in memory
            ])
        else:
            memory_text = "No previous conversation"

        prompt = PromptTemplates.SQL_GENERATION.format(
            question=question,
            memory=memory_text
        )

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
        memory = state.get("memory", [])

        # Check for errors
        if self._is_error_result(result):
            return {"final_answer": "حصل خطأ أثناء تنفيذ الاستعلام."}

        if not result:
            return {"final_answer": "مفيش نتائج."}

        # Format memory for prompt
        memory_text = ""
        if memory:
            memory_text = "\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in memory
            ])
        else:
            memory_text = "No previous conversation"

        rows_text = "\n".join([", ".join(map(str, row)) for row in result])
        prompt = PromptTemplates.FORMAT_ANSWER.format(
            question=question,
            rows_text=rows_text,
            memory=memory_text
        )

        answer = self.llm.invoke(prompt)
        return {"final_answer": answer if answer else "حصل خطأ في الإجابة."}
    
    def process_general_node(self, state: State) -> Dict[str, Any]:

        question = state["question"]
        memory = state.get("memory", [])

        # Format memory for prompt
        memory_text = ""
        if memory:
            memory_text = "\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in memory
            ])
        else:
            memory_text = "No previous conversation"

        prompt = PromptTemplates.GENERAL_RESPONSE.format(
            question=question,
            memory=memory_text
        )

        answer = self.llm.invoke(prompt)
        return {"final_answer": answer if answer else "مش قادر أرد دلوقتي."}
    
    async def save_response_node(self, state: State) -> Dict[str, Any]:
        """
        Save the assistant's final response to the database.
        """
        user_id = state["user_id"]
        final_answer = state.get("final_answer", "")

        if final_answer:
            await self.db_manager.insert_message(
                user_id=user_id,
                role="ai",
                content=final_answer
            )

        return {}

    @staticmethod
    def _is_error_result(result: List[Any]) -> bool:

        if not isinstance(result, list) or not result:
            return False
        return isinstance(result[0], dict) and "error" in result[0]

