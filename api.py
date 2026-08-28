from functools import lru_cache
import os
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "./hf_cache")

from agentic_rag import run_agentic_rag
from config import DATA_FILE, EMBEDDING_MODEL_NAME, ONLINE_LLM_MODEL, TOP_K
from data_utils import load_data
from milvus_utils import get_milvus_client, index_data_if_needed, setup_milvus_collection
from models import load_embedding_model


app = FastAPI(title="Agentic Medical RAG API", version="2.0.0")


class AskRequest(BaseModel):
    query: str


class RetrievedDoc(BaseModel):
    id: int
    title: str
    distance: float
    abstract: str


class WorkflowEvent(BaseModel):
    step: str
    status: str
    detail: str
    data: Dict[str, Any]


class AskResponse(BaseModel):
    query: str
    answer: str
    groundedness: str
    trace: List[str]
    workflow_events: List[WorkflowEvent]
    route: str
    query_analysis: Dict[str, Any]
    query_complexity: str
    sub_questions: List[str]
    tool_plan: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    search_queries: List[str]
    evidence_quality: str
    generation_mode: str
    retrieval_round: int
    retry_reason: str
    revision_count: int
    selected_evidence: List[Dict[str, Any]]
    latency_seconds: float
    retrieved_docs: List[RetrievedDoc]


@lru_cache(maxsize=1)
def get_runtime():
    client = get_milvus_client()
    if not client:
        raise RuntimeError("Milvus Lite client initialization failed")

    if not setup_milvus_collection(client):
        raise RuntimeError("Milvus Lite collection setup failed")

    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    if not embedding_model:
        raise RuntimeError("Embedding model loading failed")

    data = load_data(DATA_FILE)
    if not data:
        raise RuntimeError(f"No data loaded from {DATA_FILE}")

    if not index_data_if_needed(client, data, embedding_model):
        raise RuntimeError("Data indexing failed")

    return client, embedding_model


@app.get("/health")
def health():
    return {
        "status": "ok",
        "workflow": "langgraph-agentic-rag",
        "online_llm_model": ONLINE_LLM_MODEL,
        "top_k": TOP_K,
    }


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query cannot be empty")

    try:
        client, embedding_model = get_runtime()
        result = run_agentic_rag(query, client, embedding_model)
        retrieved_docs = [
            RetrievedDoc(
                id=doc["id"],
                title=doc["title"],
                distance=doc["distance"],
                abstract=doc["abstract"],
            )
            for doc in result["docs"]
        ]
        return AskResponse(
            query=query,
            answer=result["answer"],
            groundedness=result["groundedness"],
            trace=result["trace"],
            workflow_events=result["events"],
            route=result["route"],
            query_analysis=result["query_analysis"],
            query_complexity=result["query_complexity"],
            sub_questions=result["sub_questions"],
            tool_plan=result["tool_plan"],
            tool_calls=result["tool_calls"],
            search_queries=result["search_queries"],
            evidence_quality=result["evidence_quality"],
            generation_mode=result["generation_mode"],
            retrieval_round=result["retrieval_round"],
            retry_reason=result["retry_reason"],
            revision_count=result["revision_count"],
            selected_evidence=result["selected_evidence"],
            latency_seconds=result["latency_seconds"],
            retrieved_docs=retrieved_docs,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agentic RAG request failed: {exc}") from exc
