import time
from typing import List, Literal, TypedDict

from langgraph.graph import END, StateGraph

from config import TOP_K, id_to_doc_map
from llm_client import chat_completion
from milvus_utils import search_similar_documents
from rag_core import generate_answer


class RetrievedDocument(TypedDict):
    id: int
    title: str
    distance: float
    abstract: str
    content: str


class AgentState(TypedDict):
    query: str
    route: str
    retrieved_ids: List[int]
    distances: List[float]
    docs: List[RetrievedDocument]
    answer: str
    groundedness: str
    trace: List[str]
    latency_seconds: float


def route_query(state: AgentState) -> AgentState:
    query = state["query"].strip()
    trace = state["trace"] + ["route_query"]

    if len(query) < 4:
        return {**state, "route": "clarify", "trace": trace}

    return {**state, "route": "retrieve", "trace": trace}


def clarify(state: AgentState) -> AgentState:
    return {
        **state,
        "answer": "问题信息较少，请补充疾病、症状、检查或治疗背景后再提问。",
        "groundedness": "clarification_required",
        "trace": state["trace"] + ["clarify"],
    }


def retrieve_context(client, embedding_model):
    def _node(state: AgentState) -> AgentState:
        retrieved_ids, distances = search_similar_documents(client, state["query"], embedding_model)
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
        return {
            **state,
            "retrieved_ids": retrieved_ids,
            "distances": distances,
            "docs": docs,
            "trace": state["trace"] + [f"retrieve_context(top_k={TOP_K}, hits={len(docs)})"],
        }

    return _node


def answer_with_context(state: AgentState) -> AgentState:
    answer = generate_answer(state["query"], state["docs"])
    return {**state, "answer": answer, "trace": state["trace"] + ["answer_with_context"]}


def verify_groundedness(state: AgentState) -> AgentState:
    if not state["docs"]:
        groundedness = "no_context"
    else:
        snippets = "\n".join([doc["abstract"][:300] for doc in state["docs"][:3]])
        prompt = f"""Judge whether the answer is grounded in the retrieved context.
Return exactly one Chinese label: 有依据 / 依据不足 / 无法判断.

Question: {state['query']}

Retrieved context snippets:
{snippets}

Answer:
{state['answer']}"""
        evaluation = chat_completion(
            [
                {"role": "system", "content": "You are a strict RAG answer evaluator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=32,
        )
        groundedness = next(
            (label for label in ("有依据", "依据不足", "无法判断") if label in evaluation),
            "无法判断",
        )

    return {**state, "groundedness": groundedness, "trace": state["trace"] + ["verify_groundedness"]}


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
    graph.add_conditional_edges("route_query", decide_route, {"clarify": "clarify", "retrieve": "retrieve"})
    graph.add_edge("clarify", END)
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "verify")
    graph.add_edge("verify", END)
    return graph.compile()


def run_agentic_rag(query, client, embedding_model):
    start = time.time()
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
            "latency_seconds": 0.0,
        }
    )
    result["latency_seconds"] = round(time.time() - start, 2)
    return result
