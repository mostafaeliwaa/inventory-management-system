from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
from datetime import datetime
from config import get_settings
from DB_Management import DatabaseManager
from models import LLMManager
from Agents import GraphNodes, GraphBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(title="Inventory Chatbot API")

db_manager = DatabaseManager(settings.SUPABASE_URL, settings.SUPABASE_KEY)
llm_manager = LLMManager()
nodes = GraphNodes(llm_manager, db_manager)
graph_builder = GraphBuilder(nodes)
chatbot_graph = graph_builder.build()


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint that handles user messages with conversation memory.

    Flow:
    1. Validate input
    2. Store user message in database
    3. Run chatbot graph (which loads memory, processes, and saves response)
    4. Return assistant response
    """

    if not request.user_id or not request.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required")

    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    try:
        logger.info(f"Received message from user {request.user_id}")

        await db_manager.insert_message(
            user_id=request.user_id,
            role="human",
            content=request.message.strip()
        )
        logger.info("User message stored in database")

        initial_state = {
            "user_id": request.user_id,
            "question": "",
            "memory": [],
            "type": "",
            "query": "",
            "result": [],
            "final_answer": ""
        }

        from pathlib import Path


        # generate image bytes
        img_bytes = chatbot_graph.get_graph().draw_mermaid_png()

        # save to file
        output_path = Path("graph.png")
        with open(output_path, "wb") as f:
            f.write(img_bytes)

        print(f"Graph saved to {output_path}")

        result = await chatbot_graph.ainvoke(initial_state)

        final_answer = result.get("final_answer", "")

        if not final_answer:
            logger.error("Graph returned empty final_answer")
            raise HTTPException(status_code=500, detail="Failed to generate response")

        logger.info(f"Successfully processed message for user {request.user_id}")

        return ChatResponse(response=final_answer)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
