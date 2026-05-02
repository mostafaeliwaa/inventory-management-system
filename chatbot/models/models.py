from chatbot.config import get_settings
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import os
from typing import Dict, Any


settings = get_settings()

class LLMManager:

    
    def __init__(self):
        self.model = self._initialize_model()
    
    def _initialize_model(self):
        
        os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
        return ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            timeout=settings.LLM_TIMEOUT,
            max_retries=settings.LLM_MAX_RETRIES,
        )
    
    def invoke(self, prompt: str) -> str:
        
        try:
            response = self.model.invoke(prompt).content.strip()
            response = response.replace("```json", "").replace("```", "").strip()
            return response
        except Exception as e:
            print(f" LLM Error: {e}")
            return ""
    
    def parse_json_response(self, response: str) -> Dict[str, Any]:
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f" JSON Parse Error: {e}")
            return {"type": "general", "query": ""}