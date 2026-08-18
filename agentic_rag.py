import time
from typing import Any, Dict, List, Literal, TypedDict

from langgraph.graph import END, StateGraph

from config import EMBEDDING_MODEL_NAME, TOP_K, id_to_doc_map
from llm_client import chat_completion
from milvus_utils import search_similar_documents
from rag_core import generate_answer


class RetrievedDocument(TypedDict):
    id: int
    title: str
    distance: float
    abstract: str
    content: str


class WorkflowEvent(TypedDict):
    step: str
    status: str
    detail: str
    data: Dict[str, Any]


class AgentState(TypedDict):
    query: str
    route: str
    retrieved_ids: List[int]
    distances: List[float]
    docs: List[RetrievedDocument]
    answer: str
    groundedness: str
    trace: List[str]
    events: List[WorkflowEvent]
    latency_seconds: float


def append_event(
    state: AgentState,
    step: str,
    status: str,
    detail: str,
    data: Dict[str, Any] | None = None,
) -> List[WorkflowEvent]:
    return state["events"] + [
        {
            "step": step,
            "status": status,
            "detail": detail,
            "data": data or {},
        }
    ]


def route_query(state: AgentState) -> AgentState:
    query = state["query"].strip()
    route = "clarify" if len(query) < 4 else "retrieve"
    detail = (
        f"检测到问题长度为 {len(query)}，信息较少，进入澄清分支。"
        if route == "clarify"
        else f"检测到问题长度为 {len(query)}，进入知识库检索分支。"
    )
    return {
        **state,
        "route": route,
        "trace": state["trace"] + ["route_query"],
        "events": append_event(
            state,
            "route_query",
            "completed",
            detail,
            {"query_length": len(query), "next_node": route},
        ),
    }


def clarify(state: AgentState) -> AgentState:
    answer = "问题信息较少，请补充疾病、症状、检查或治疗背景后再提问。"
    return {
        **state,
        "answer": answer,
        "groundedness": "clarification_required",
        "trace": state["trace"] + ["clarify"],
        "events": append_event(
            state,
            "clarify",
            "completed",
            "未调用知识库和在线模型，直接提示用户补充问题背景。",
            {"answer": answer},
        ),
    }


def retrieve_context(client, embedding_model):
    def _node(state: AgentState) -> AgentState:
        started_at = time.time()
        retrieved_ids, distances = search_similar_documents(
            client, state["query"], embedding_model
        )
        docs = []
        for index, doc_id in enumerate(retrieved_ids):
            doc = id_to_doc_map.get(doc_id)
            if not doc:
                continue
            docs.append(
                {
                    "id": doc_id,
                    "title": doc.get("title", ""),
                    "distance": float(distances[index]) if index < len(distances) else 0.0,
                    "abstract": doc.get("abstract", ""),
                    "content": doc.get("content", ""),
                }
            )

        retrieval_ms = round((time.time() - started_at) * 1000, 1)
        documents = [
            {
                "rank": index + 1,
                "id": doc["id"],
                "title": doc["title"],
                "distance": round(doc["distance"], 4),
            }
            for index, doc in enumerate(docs)
        ]
        detail = (
            f"使用 {EMBEDDING_MODEL_NAME} 将问题向量化，"
            f"在 Milvus Lite 中检索 Top-{TOP_K}，命中 {len(docs)} 条文档，"
            f"耗时 {retrieval_ms} ms。"
        )
        next_state = {
            **state,
            "retrieved_ids": retrieved_ids,
            "distances": distances,
            "docs": docs,
            "trace": state["trace"] + [f"retrieve_context(top_k={TOP_K}, hits={len(docs)})"],
        }
        next_state["events"] = append_event(
            next_state,
            "retrieve_context",
            "completed",
            detail,
            {
                "top_k": TOP_K,
                "hits": len(docs),
                "retrieval_ms": retrieval_ms,
                "documents": documents,
            },
        )
        return next_state

    return _node


def answer_with_context(state: AgentState) -> AgentState:
    started_at = time.time()
    context_chars = sum(len(doc["content"]) for doc in state["docs"])
    answer = generate_answer(state["query"], state["docs"])
    generation_ms = round((time.time() - started_at) * 1000, 1)
    detail = (
        f"将 {len(state['docs'])} 条召回文档拼接为 {context_chars} 个字符的上下文，"
        f"调用在线大模型生成回答，耗时 {generation_ms} ms。"
    )
    next_state = {
        **state,
        "answer": answer,
        "trace": state["trace"] + ["answer_with_context"],
    }
    next_state["events"] = append_event(
        next_state,
        "answer_with_context",
        "completed",
        detail,
        {
            "context_documents": len(state["docs"]),
            "context_chars": context_chars,
            "generation_ms": generation_ms,
            "answer_chars": len(answer),
        },
    )
    return next_state


def verify_groundedness(state: AgentState) -> AgentState:
    if not state["docs"]:
        groundedness = "no_context"
        detail = "没有召回上下文，未执行依据校验。"
        data = {"groundedness": groundedness, "evaluation_called": False}
    else:
        snippets = "\n".join([doc["abstract"][:300] for doc in state["docs"][:3]])
        prompt = f"""Judge whether the answer is grounded in the retrieved context.
Return exactly one Chinese label: 有依据 / 依据不足 / 无法判断.

Question: {state['query']}

Retrieved context snippets:
{snippets}

Answer:
{state['answer']}"""
        started_at = time.time()
        evaluation = chat_completion(
            [
                {"role": "system", "content": "You are a strict RAG answer evaluator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=32,
        )
        verification_ms = round((time.time() - started_at) * 1000, 1)
        groundedness = next(
            (
                label
                for label in ("有依据", "依据不足", "无法判断")
                if label in evaluation
            ),
            "无法判断",
        )
        detail = (
            f"使用独立校验调用检查回答与召回上下文的一致性，"
            f"结果为“{groundedness}”，耗时 {verification_ms} ms。"
        )
        data = {
            "groundedness": groundedness,
            "evaluation_called": True,
            "verification_ms": verification_ms,
            "raw_evaluation": evaluation[:100],
        }

    next_state = {
        **state,
        "groundedness": groundedness,
        "trace": state["trace"] + ["verify_groundedness"],
    }
    next_state["events"] = append_event(
        next_state,
        "verify_groundedness",
        "completed",
        detail,
        data,
    )
    return next_state


def decide_route(state: AgentState) -> Literal["clarify", "retrieve"]:
    return "clarify" if state["route"] == "clarify" else "retrieve"


def build_agentic_rag_graph(client, embedding_model):
    graph = StateGraph(AgentState)
    graph.add_node("route_query", route_query)
    graph.add_node("clarify", clarify)
    graph.add_node("retrieve", retrieve_context(client, embedding_model))
    graph.add_node("answer", answer_with_context)
    graph.add_node("verify", verify_groundedness)

    graph.set_entry_point("route_query")
    graph.add_conditional_edges(
        "route_query",
        decide_route,
        {"clarify": "clarify", "retrieve": "retrieve"},
    )
    graph.add_edge("clarify", END)
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "verify")
    graph.add_edge("verify", END)
    return graph.compile()


def run_agentic_rag(query, client, embedding_model):
    started_at = time.time()
    app = build_agentic_rag_graph(client, embedding_model)
    result = app.invoke(
        {
            "query": query,
            "route": "",
            "retrieved_ids": [],
            "distances": [],
            "docs": [],
            "answer": "",
            "groundedness": "",
            "trace": [],
            "events": [],
            "latency_seconds": 0.0,
        }
    )
    result["latency_seconds"] = round(time.time() - started_at, 2)
    return result
