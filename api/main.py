import os
import sys
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Ensure project root is on sys.path for utils imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from google import genai 
from utils.config import load_config
from utils.retriever import retrieve_top_k
from utils.intent_filter import is_dmrc_query
from utils.session_memory import session_memory
from utils.metro_prompts import get_metro_prompt


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    top_k: int = 3
    threshold: float = 0.55
    memory_enabled: bool = True


class ChatResponse(BaseModel):
    response: str
    source: str
    confidence: float
    session_id: str
    context: List[Dict[str, str]] = []


def build_contextual_prompt(user_query: str, top_k_context: List[Dict[str, str]], conversation_context: str = "") -> str:
    faq_context = "\n\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in top_k_context])
    final_prompt = (
        "You are a helpful assistant for the Delhi Metro Rail Corporation (DMRC).\n\n"
        f"{conversation_context}"
        f"Use the following FAQs as context:\n{faq_context}\n\n"
        f"Now answer the user's question clearly and conversationally, considering the conversation history:\n{user_query}\n\nAnswer:"
    )
    return final_prompt


def format_context(pairs: List[tuple]) -> List[Dict[str, str]]:
    return [{"question": q, "answer": a} for q, a in pairs]


# Initialize app and dependencies
load_dotenv()
config: Dict[str, Any] = load_config()

api_key = os.getenv(config["llm"]["api_key_env"])
client = None
model_name = config["llm"]["model"]
if api_key:
    client = genai.Client(api_key=api_key)

app = FastAPI(title="DMRC Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "service": "DMRC Chatbot API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> Dict[str, str]: 
    ready = bool(client)
    return {"status": "healthy", "llm": "ready" if ready else "not_configured"}


def _generate_text(prompt: str) -> str:
    if client is not None:
        response = client.models.generate_content(model=model_name, contents=prompt)
        return getattr(response, "text", str(response)).strip()
    raise RuntimeError("LLM not configured. Set GEMINI_API_KEY.")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    # Ensure a session exists
    session_id = req.session_id or session_memory.create_session()

    # Intent classification
    try:
        is_dmrc = is_dmrc_query(req.query)
    except Exception:
        is_dmrc = True

    if not is_dmrc:
        if not client:
            raise HTTPException(status_code=503, detail="LLM not configured. Set GEMINI_API_KEY.")

        conversation_context = session_memory.get_conversation_context(session_id, count=3) if req.memory_enabled else ""
        metro_prompt = get_metro_prompt(req.query, conversation_context)

        response_text = _generate_text(metro_prompt)

        if req.memory_enabled:
            session_memory.add_conversation(
                session_id=session_id,
                user_query=req.query,
                bot_response=response_text,
                source="metro_general",
                confidence=0.8,
                context_used=[],
                metadata={"top_k": req.top_k, "threshold": req.threshold, "memory_enabled": req.memory_enabled},
            )

        return ChatResponse(
            response=response_text,
            source="metro_general",
            confidence=0.8,
            session_id=session_id,
            context=[],
        )

    # DMRC path with retrieval
    top_k_pairs = retrieve_top_k(req.query, k=req.top_k, threshold=req.threshold)
    if not top_k_pairs:
        response_text = "I couldn't find specific information about that. Please rephrase or ask about Delhi Metro services."

        if req.memory_enabled:
            session_memory.add_conversation(
                session_id=session_id,
                user_query=req.query,
                bot_response=response_text,
                source="no_matches",
                confidence=0.0,
                context_used=[],
                metadata={"top_k": req.top_k, "threshold": req.threshold, "memory_enabled": req.memory_enabled},
            )

        return ChatResponse(
            response=response_text,
            source="no_matches",
            confidence=0.0,
            session_id=session_id,
            context=[],
        )

    if not client:
        raise HTTPException(status_code=503, detail="LLM not configured. Set GEMINI_API_KEY.")

    context_list = format_context(top_k_pairs)
    conversation_context = session_memory.get_conversation_context(session_id, count=3) if req.memory_enabled else ""
    final_prompt = build_contextual_prompt(req.query, context_list, conversation_context)

    response_text = _generate_text(final_prompt)

    if req.memory_enabled:
        session_memory.add_conversation(
            session_id=session_id,
            user_query=req.query,
            bot_response=response_text,
            source="dmrc_rag",
            confidence=0.8,
            context_used=top_k_pairs,
            metadata={"top_k": req.top_k, "threshold": req.threshold, "memory_enabled": req.memory_enabled},
        )

    return ChatResponse(
        response=response_text,
        source="dmrc_rag",
        confidence=0.8,
        session_id=session_id,
        context=context_list,
    )


