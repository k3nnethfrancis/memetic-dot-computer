import os
import uvicorn
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cognition.llms.ollama import OllamaModel
from cognition.engines.chat_engine import ChatEngine, Conversation
from cognition.models.chat_models import ChatMessage, ChatHistory
from personality.k3nn import system_prompt as k3nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths for SSL certificates
CERTS_DIR = "/Users/kenneth/Desktop/lab/k3nn.computer/certs"
ssl_keyfile = os.getenv('SSL_KEYFILE', CERTS_DIR + '/key.pem')
ssl_certfile = os.getenv('SSL_CERTFILE', CERTS_DIR + '/cert.pem')

# globals
chat_engine = None
system_prompt = k3nn

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up...")
    global chat_engine
    llm = OllamaModel()
    conversation = Conversation()
    conversation.add_message(ChatMessage(role="system", content=system_prompt))
    chat_engine = ChatEngine(llm, conversation)
    yield
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

# CORS configuration
origins = [
    "http://0.0.0.0:3333",
    "http://localhost:3333",
    "http://127.0.0.1:3333",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_chat_engine():
    global chat_engine
    return chat_engine

class ChatRequest(BaseModel):
    message: str

@app.post("/chat-engine")
async def chat_engine(request: ChatRequest, chat_engine: ChatEngine = Depends(get_chat_engine)):
    logger.info(f"Received chat request: {request}")
    try:
        response = chat_engine.chat(request.message)
        logger.info(f"Chat response: {response}")
        logger.info(f"Conversation history: {chat_engine.conversation.conversation_history}")
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
# @app.post("/chat")
# async def chat(request: ChatRequest):
#     llm = OllamaModel(model="llama3.1")
#     chat_history = [
#         ChatMessage(role="system", content=system_prompt),
#     ]
#     new_message = ChatMessage(role="user", content=request.message)
#     chat_history.append(new_message)

#     try:
#         response = llm.generate(chat_history)
#         return response
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(chat_history: ChatHistory):
    llm = OllamaModel(model="llama3.1")

    # make sure the object is a chat history object with chat messages
    if not isinstance(chat_history, ChatHistory):
        raise HTTPException(status_code=400, detail="Invalid chat history object")

    # Ensure system prompt is at the beginning
    if not chat_history.messages or chat_history.messages[0].role != "system":
        chat_history.messages.insert(0, ChatMessage(role="system", content=system_prompt))
    
    try:
        response = llm.generate(chat_history.messages)
        # response is a string, wrap it in a ChatMessage
        ai_message = ChatMessage(role="assistant", content=response)
        chat_history.messages.append(ai_message)
        return chat_history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Chat API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=1337,
        # Comment out SSL for now
        # ssl_keyfile=ssl_keyfile,
        # ssl_certfile=ssl_certfile
    )