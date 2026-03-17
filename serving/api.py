"""
FastAPI REST API for the LLaMA-3.1-8B PT Debugger.

Endpoints:
  POST /summarize  — Summarize a test failure
  POST /qa         — Answer a question about a failure
  POST /chat       — Multi-turn chatbot (stateless per request; pass full history)
  GET  /health     — Health check
  GET  /ready      — Readiness check (returns 503 until model is loaded)

Launch:
  uvicorn serving.api:app --host 0.0.0.0 --port 8000
  # With env vars:
  MODEL_PATH=... ADAPTER_PATH=... uvicorn serving.api:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App configuration from environment
# ---------------------------------------------------------------------------

MODEL_PATH      = os.getenv("MODEL_PATH",      "meta-llama/Meta-Llama-3.1-8B-Instruct")
ADAPTER_PATH    = os.getenv("ADAPTER_PATH",    "outputs/llama-pt-debugger")
DEVICE          = os.getenv("DEVICE",          "auto")
MAX_NEW_TOKENS  = int(os.getenv("MAX_NEW_TOKENS", "400"))

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PT Test Failure Debugger API",
    description=(
        "LLaMA-3.1-8B fine-tuned on PyTorch test failures. "
        "Supports summarization, Q&A, and multi-turn chatbot workloads."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model singleton (loaded on first request or at startup)
# ---------------------------------------------------------------------------

_model      = None
_summarizer = None
_qa_module  = None
_is_ready   = False


@app.on_event("startup")
async def startup_event():
    global _model, _summarizer, _qa_module, _is_ready
    try:
        from inference.infer import PTDebuggerModel
        from inference.qa import QAModule
        from inference.summarize import Summarizer

        logger.info("Loading model at startup...")
        _model      = PTDebuggerModel(
            model_path=MODEL_PATH,
            adapter_path=ADAPTER_PATH,
            device=DEVICE,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        _summarizer = Summarizer(_model)
        _qa_module  = QAModule(_model)
        _is_ready   = True
        logger.info("Model loaded and ready.")
    except Exception as exc:
        logger.error("Failed to load model at startup: %s", exc)
        # Server stays up but /ready will return 503


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class SummarizeRequest(BaseModel):
    failure_input: str = Field(
        ...,
        description="Raw failure text, JUnit XML snippet, or structured failure description",
        min_length=10,
    )
    max_new_tokens: Optional[int] = Field(
        None,
        ge=1, le=1024,
        description="Maximum tokens to generate (default: server setting)",
    )


class SummarizeResponse(BaseModel):
    summary: str


class QARequest(BaseModel):
    context: str = Field(
        ...,
        description="Failure context (log, XML, or plain text)",
        min_length=10,
    )
    question: str = Field(
        ...,
        description="The question to answer about the failure",
        min_length=3,
    )
    max_new_tokens: Optional[int] = Field(None, ge=1, le=1024)


class QAResponse(BaseModel):
    question: str
    answer: str


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(
        ...,
        description="Conversation history (system + user/assistant turns)",
        min_length=1,
    )
    max_new_tokens: Optional[int] = Field(None, ge=1, le=1024)


class ChatResponse(BaseModel):
    response: str
    role: str = "assistant"


class HealthResponse(BaseModel):
    status: str
    model: str
    adapter: str
    device: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Status"])
async def health():
    return HealthResponse(
        status="ok",
        model=MODEL_PATH,
        adapter=ADAPTER_PATH,
        device=DEVICE,
    )


@app.get("/ready", tags=["Status"])
async def ready():
    if not _is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not yet loaded.",
        )
    return {"status": "ready"}


@app.post("/summarize", response_model=SummarizeResponse, tags=["Tasks"])
async def summarize(req: SummarizeRequest):
    if not _is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready.",
        )
    try:
        summary = _summarizer.run(
            req.failure_input,
            max_new_tokens=req.max_new_tokens or MAX_NEW_TOKENS,
        )
        return SummarizeResponse(summary=summary)
    except Exception as exc:
        logger.exception("Summarize endpoint error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/qa", response_model=QAResponse, tags=["Tasks"])
async def qa(req: QARequest):
    if not _is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready.",
        )
    try:
        answer = _qa_module.run(
            req.context,
            req.question,
            max_new_tokens=req.max_new_tokens or MAX_NEW_TOKENS,
        )
        return QAResponse(question=req.question, answer=answer)
    except Exception as exc:
        logger.exception("QA endpoint error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse, tags=["Tasks"])
async def chat(req: ChatRequest):
    """
    Stateless multi-turn chat — pass the full conversation history each time.
    The model generates the next assistant turn.
    """
    if not _is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready.",
        )
    try:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        response = _model.chat_generate(
            messages,
            max_new_tokens=req.max_new_tokens or MAX_NEW_TOKENS,
        )
        return ChatResponse(response=response)
    except Exception as exc:
        logger.exception("Chat endpoint error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
