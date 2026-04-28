from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()  

class Settings(BaseSettings):
    
    GOOGLE_API_KEY: str
    LLM_MODEL: str = "gemini-2.5-flash"
    LLM_TEMPERATURE: float = 0
    LLM_MAX_TOKENS : int = 1000
    LLM_TIMEOUT : int = 30
    LLM_MAX_RETRIES : int = 6

    
    SUPABASE_URL: str
    SUPABASE_KEY: str

    
    LANGCHAIN_API_KEY: str | None = None
    LANGCHAIN_TRACING: bool = False
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_PROJECT: str = "Chatbot"

    # # ===== Database Settings =====
    # DB_DRIVER: str
    # DB_SERVER: str
    # DB_NAME: str
    # DB_TRUSTED_CONNECTION: str 
    
    # @property
    # def DB_CONNECTION_STRING(self) -> str:
    #     return (
    #         f"DRIVER={self.DB_DRIVER};"
    #         f"SERVER={self.DB_SERVER};"
    #         f"DATABASE={self.DB_NAME};"
    #         f"Trusted_Connection={self.DB_TRUSTED_CONNECTION};"
    #     )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    return Settings()