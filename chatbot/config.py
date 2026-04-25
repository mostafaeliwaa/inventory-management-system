"""
Application configuration and environment settings.
Loads variables from the .env file using Pydantic Settings,
such as API keys, model configuration, and external services.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ===== LLM =====
    GOOGLE_API_KEY: str
    LLM_MODEL: str = "gemini-2.5-flash"
    LLM_TEMPERATURE: float = 0

    # ===== Supabase =====
    SUPABASE_URL: str
    SUPABASE_KEY: str

    # ===== LangSmith =====
    LANGCHAIN_API_KEY: str | None = None
    LANGCHAIN_TRACING: bool = False
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_PROJECT: str = "Chatbot"


    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    return Settings()