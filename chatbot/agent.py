"""
This file contains the core AI agent logic.
Handles interaction with the LLM (e.g., Gemini), processes user input,
and generates responses. It may also orchestrate tools and prompts.
"""
from langgraph.graph import StateGraph, END, START
from langgraph.graph import MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI
from config import get_settings
from langchain_core.messages import HumanMessage

# ===== Load settings =====
settings = get_settings()

# ===== LLM =====
llm = ChatGoogleGenerativeAI(
    model=settings.LLM_MODEL,
    temperature=settings.LLM_TEMPERATURE,
    google_api_key=settings.GOOGLE_API_KEY,
)

# ===== State =====

class State(MessagesState):
    pass

def chatbot_node(state: State):
    messages = state["messages"]

    
    last_message = messages[-1]

     
    if isinstance(last_message, dict):
        user_input = last_message.get("content", "")
    else:
        user_input = last_message.content

    response = llm.invoke([
        HumanMessage(content=user_input)
    ])

    return {
        "messages": [response]
    }

# ===== Graph =====
graph = StateGraph(State)

graph.add_node("chatbot", chatbot_node)

graph.add_edge(START, "chatbot")

graph.add_edge("chatbot", END)

# Compile
app = graph.compile()


