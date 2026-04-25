"""
Defines data models and schemas used across the application.
Includes request/response models and any structured data objects.
"""
from config import get_settings
from langchain_google_genai import ChatGoogleGenerativeAI

settings = get_settings()

def get_model():

   Model = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_retries=2,
            )

   return Model