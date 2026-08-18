import re
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


class SelectedEvidence(TypedDict):
    rank: int
    doc_id: int
    title: str
    distance: float
    snippet: str


class WorkflowEvent(TypedDict):
    step: str
    status: str
    detail: str
    data: Dict[str, Any]


class AgentState(TypedDict):
    query: str
    route: str
    query_analysis: Dict[str, Any]
    search_queries: List[str]
    retrieved_ids: List[int]
    distances: List[float]
    docs: List[RetrievedDocument]
    evidence_quality: str
    selected_evidence: List[SelectedEvidence]
    answer: str
    groundedness: str
    trace: List[str]
    events: List[WorkflowEvent]
    latency_seconds: float


MEDICAL_KEYWORDS = [
    "白血病",
    "发热",
    "发烧",
    "感染",
    "贫血",
    "出血",
    "化疗",
    "移植",
    "造血干细胞",
    "腰穿",
    "血常规",
    "骨痛",
    "血小板",
    "中性粒细胞",
]

INTENT_RULES = [
    ("cause", ["为什么", "原因", "导致", "引起", "怎么会"]),
    ("care", ["注意", "护理", "饮食", "生活", "怎么办"]),
    ("diagnosis", ["检查", "诊断", "血常规", "确诊", "判断"]),
    ("treatment", ["治疗", "化疗", "移植", "用药", "控制"]),
    ("symptom", ["症状", "表现", "信号"]),
]


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


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip())


def infer_intent(query: str) -> str:
    for intent, words in INTENT_RULES:
        if any(word in query for word in words):
            return intent
    return "general"


def detect_entities(query: str) -> List[str]:
    entities = [word for word in MEDICAL_KEYWORDS if word in query]
    if "发烧" in entities and "发热" not in entities:
        entities.append("发热")
    return list(dict.fromkeys(entities))


def build_query_variants(query: str, analysis: Dict[str, Any]) -> List[str]:
    variants = [query]
    entities = analysis.get("entities", [])
    intent = analysis.get("intent", "general")

    if entities:
        variants.append(" ".join(entities))

    if "白血病" in entities and intent == "cause":
        variants.extend(["白血病 发热 原因", "白血病 感染 发热", "白血病 为什么 发烧"])
    elif "白血病" in entities and intent == "care":
        variants.extend(["白血病 护理 注意事项", "白血病 化疗 注意", "白血病 饮食 生活指导"])
    elif intent == "diagnosis":
        variants.append(f"{query} 检查 诊断")
    elif intent == "treatment":
        variants.append(f"{query} 治疗 控制")

    normalized = []
    for item in variants:
        item = normalize_query(item)
        if item and item not in normalized:
            normalized.append(item)
    return normalized[:4]


def route_query(state: AgentState) -> AgentState:
    query = normalize_query(state["query"])
    if len(query) < 4:
        route = "clarify"
        detail = f"检测到问题长度为 {len(query)}，信息较少，进入澄清分支。"
    else:
        entities = detect_entities(query)
        route = "medical_rag" if entities else "medical_rag"
        detail = (
            f"检测到问题长度为 {len(query)}，进入医学知识库问答分支。"
            f"初步识别实体：{entities or ['未命中固定词表']}。"
        )

    return {
        **state,
        "query": query,
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
            "问题过短，未进入检索与生成流程，直接要求用户补充上下文。",
            {"answer": answer},
        ),
    }


def analyze_query(state: AgentState) -> AgentState:
    query = state["query"]
    entities = detect_entities(query)
    intent = infer_intent(query)
    analysis = {
        "intent": intent,
        "entities": entities,
        "needs_retrieval": True,
        "risk_control": "medical_answer_must_be_grounded",
    }
    detail = (
        f"完成查询分析：意图={intent}，实体={entities or ['未命中固定词表']}。"
        "该结果会用于后续检索 query 改写和证据质量判断。"
    )
    return {
        **state,
        "query_analysis": analysis,
        "trace": state["trace"] + ["analyze_query"],
        "events": append_event(
            state,
            "analyze_query",
            "completed",
            detail,
            analysis,
        ),
    }


def rewrite_query(state: AgentState) -> AgentState:
    search_queries = build_query_variants(state["query"], state["query_analysis"])
    detail = (
        f"根据识别出的实体和意图生成 {len(search_queries)} 个检索 query，"
        "用于提升召回覆盖率。"
    )
    return {
        **state,
        "search_queries": search_queries,
        "trace": state["trace"] + ["rewrite_query"],
        "events": append_event(
            state,
            "rewrite_query",
            "completed",
            detail,
            {"search_queries": search_queries},
        ),
    }


def retrieve_context(client, embedding_model):
    def _node(state: AgentState) -> AgentState:
        started_at = time.time()
        merged: Dict[int, Dict[str, Any]] = {}

        for search_query in state["search_queries"]:
            hit_ids, hit_distances = search_similar_documents(
                client, search_query, embedding_model
            )
            for index, doc_id in enumerate(hit_ids):
                distance = float(hit_distances[index]) if index < len(hit_distances) else 0.0
                existing = merged.get(doc_id)
                if existing is None or distance < existing["distance"]:
                    merged[doc_id] = {
                        "doc_id": doc_id,
                        "distance": distance,
                        "matched_query": search_query,
                    }

        ranked_hits = sorted(merged.values(), key=lambda item: item["distance"])[:TOP_K]
        docs = []
        for hit in ranked_hits:
            doc = id_to_doc_map.get(hit["doc_id"])
            if not doc:
                continue
            docs.append(
                {
                    "id": hit["doc_id"],
                    "title": doc.get("title", ""),
                    "distance": hit["distance"],
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
                "matched_query": next(
                    (
                        hit["matched_query"]
                        for hit in ranked_hits
                        if hit["doc_id"] == doc["id"]
                    ),
                    "",
                ),
            }
            for index, doc in enumerate(docs)
        ]
        detail = (
            f"对 {len(state['search_queries'])} 个 query 执行多路向量检索，"
            f"合并去重后保留 Top-{TOP_K}，命中 {len(docs)} 条文档，"
            f"耗时 {retrieval_ms} ms。"
        )
        next_state = {
            **state,
            "retrieved_ids": [doc["id"] for doc in docs],
            "distances": [doc["distance"] for doc in docs],
            "docs": docs,
            "trace": state["trace"] + [f"retrieve_context(queries={len(state['search_queries'])}, hits={len(docs)})"],
        }
        next_state["events"] = append_event(
            next_state,
            "retrieve_context",
            "completed",
            detail,
            {
                "embedding_model": EMBEDDING_MODEL_NAME,
                "search_queries": state["search_queries"],
                "top_k": TOP_K,
                "hits": len(docs),
                "retrieval_ms": retrieval_ms,
                "documents": documents,
            },
        )
        return next_state

    return _node


def grade_evidence(state: AgentState) -> AgentState:
    if not state["docs"]:
        quality = "no_context"
        reason = "没有召回到可用文档。"
    else:
        best_distance = min(state["distances"]) if state["distances"] else 999
        entity_hits = sum(
            1
            for doc in state["docs"]
            for entity in state["query_analysis"].get("entities", [])
            if entity and (entity in doc["title"] or entity in doc["abstract"])
        )
        if best_distance <= 0.75 or entity_hits >= 2:
            quality = "strong"
        elif best_distance <= 0.95:
            quality = "usable"
        else:
            quality = "weak"
        reason = (
            f"最佳距离={best_distance:.4f}，实体命中次数={entity_hits}，"
            f"证据质量判断为 {quality}。"
        )

    return {
        **state,
        "evidence_quality": quality,
        "trace": state["trace"] + ["grade_evidence"],
        "events": append_event(
            state,
            "grade_evidence",
            "completed",
            reason,
            {"evidence_quality": quality},
        ),
    }


def select_evidence(state: AgentState) -> AgentState:
    selected = []
    for index, doc in enumerate(state["docs"]):
        snippet = doc["abstract"][:500].replace("\n", " ")
        selected.append(
            {
                "rank": index + 1,
                "doc_id": doc["id"],
                "title": doc["title"],
                "distance": round(doc["distance"], 4),
                "snippet": snippet,
            }
        )
    detail = f"从 {len(state['docs'])} 条召回文档中整理 {len(selected)} 条证据片段，用于回答生成与校验。"
    return {
        **state,
        "selected_evidence": selected,
        "trace": state["trace"] + ["select_evidence"],
        "events": append_event(
            state,
            "select_evidence",
            "completed",
            detail,
            {"selected_evidence": selected},
        ),
    }


def build_extractive_fallback_answer(
    query: str,
    selected_evidence: List[SelectedEvidence],
    evidence_quality: str,
    error: str,
) -> str:
    if not selected_evidence:
        return "当前知识库没有召回到足够相关的文档，无法基于已有资料回答该问题。"

    lines = [
        "在线大模型调用超时或失败，系统已切换为证据摘录式回答。",
        f"问题：{query}",
        f"证据质量：{evidence_quality}",
        "",
        "可参考的召回证据：",
    ]
    for item in selected_evidence[:3]:
        snippet = item["snippet"][:180]
        lines.append(
            f"- [文档ID {item['doc_id']}] {item['title']}：{snippet}"
        )
    lines.extend(
        [
            "",
            "说明：该 fallback 不编造新医学结论，只展示知识库召回证据。"
            "恢复在线模型后，系统会自动回到生成式回答。",
            f"异常摘要：{error[:120]}",
        ]
    )
    return "\n".join(lines)


def answer_with_context(state: AgentState) -> AgentState:
    started_at = time.time()
    answer_docs = [
        {
            "id": item["doc_id"],
            "title": item["title"],
            "content": f"[文档ID: {item['doc_id']}] {item['title']}\n{item['snippet']}",
        }
        for item in state["selected_evidence"]
    ]
    context_chars = sum(len(doc["content"]) for doc in answer_docs)
    generation_mode = "online_llm"
    generation_error = ""
    try:
        answer = generate_answer(
            state["query"],
            answer_docs,
            query_analysis=state["query_analysis"],
            evidence_quality=state["evidence_quality"],
        )
    except Exception as exc:
        generation_mode = "extractive_fallback"
        generation_error = str(exc)
        answer = build_extractive_fallback_answer(
            state["query"],
            state["selected_evidence"],
            state["evidence_quality"],
            generation_error,
        )
    generation_ms = round((time.time() - started_at) * 1000, 1)
    detail = (
        f"将 {len(state['docs'])} 条证据文档拼接为 {context_chars} 个字符的上下文，"
        f"结合查询意图和证据质量生成回答，模式={generation_mode}，耗时 {generation_ms} ms。"
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
            "intent": state["query_analysis"].get("intent"),
            "evidence_quality": state["evidence_quality"],
            "context_documents": len(state["docs"]),
            "context_chars": context_chars,
            "generation_ms": generation_ms,
            "generation_mode": generation_mode,
            "generation_error": generation_error[:200],
            "answer_chars": len(answer),
        },
    )
    return next_state


def heuristic_groundedness(answer: str, docs: List[RetrievedDocument], evidence_quality: str) -> Dict[str, Any]:
    context = "".join(doc["abstract"] + doc["title"] for doc in docs)
    answer_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}", answer))
    context_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}", context))
    if not answer_tokens or not docs:
        ratio = 0.0
    else:
        ratio = len(answer_tokens & context_tokens) / max(len(answer_tokens), 1)

    if evidence_quality in {"strong", "usable"} and ratio >= 0.18:
        label = "有依据"
    elif docs:
        label = "依据不足"
    else:
        label = "无法判断"
    return {"label": label, "overlap_ratio": round(ratio, 3)}


def verify_groundedness(state: AgentState) -> AgentState:
    heuristic = heuristic_groundedness(
        state["answer"], state["docs"], state["evidence_quality"]
    )
    if not state["docs"]:
        groundedness = "no_context"
        detail = "没有召回上下文，未执行在线依据校验。"
        data = {"groundedness": groundedness, "evaluation_called": False}
    else:
        snippets = "\n".join([item["snippet"] for item in state["selected_evidence"][:3]])
        prompt = f"""你是严格的RAG答案评估器。请判断答案是否被召回证据支持。
只输出一个标签：有依据 / 依据不足 / 无法判断。

问题：{state['query']}

召回证据：
{snippets}

答案：
{state['answer']}"""
        started_at = time.time()
        evaluation_error = ""
        try:
            evaluation = chat_completion(
                [
                    {"role": "system", "content": "你只输出评估标签，不输出解释。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=64,
            )
        except Exception as exc:
            evaluation = ""
            evaluation_error = str(exc)
        verification_ms = round((time.time() - started_at) * 1000, 1)
        llm_label = next(
            (
                label
                for label in ("有依据", "依据不足", "无法判断")
                if label in evaluation
            ),
            "",
        )
        groundedness = llm_label if llm_label in {"有依据", "依据不足"} else heuristic["label"]
        detail = (
            f"综合在线评估和启发式覆盖率进行依据校验，最终结果为“{groundedness}”。"
            f"在线评估耗时 {verification_ms} ms。"
        )
        data = {
            "groundedness": groundedness,
            "llm_label": llm_label or "empty_or_unrecognized",
            "heuristic_label": heuristic["label"],
            "overlap_ratio": heuristic["overlap_ratio"],
            "evaluation_called": True,
            "evaluation_error": evaluation_error[:200],
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


def finalize_response(state: AgentState) -> AgentState:
    detail = (
        f"完成一次 Agentic RAG 调用：route={state['route']}，"
        f"intent={state['query_analysis'].get('intent')}，"
        f"evidence_quality={state['evidence_quality']}，"
        f"groundedness={state['groundedness']}。"
    )
    return {
        **state,
        "trace": state["trace"] + ["finalize_response"],
        "events": append_event(
            state,
            "finalize_response",
            "completed",
            detail,
            {
                "route": state["route"],
                "intent": state["query_analysis"].get("intent"),
                "evidence_quality": state["evidence_quality"],
                "groundedness": state["groundedness"],
            },
        ),
    }


def decide_route(state: AgentState) -> Literal["clarify", "analyze"]:
    return "clarify" if state["route"] == "clarify" else "analyze"


def build_agentic_rag_graph(client, embedding_model):
    graph = StateGraph(AgentState)
    graph.add_node("route_query", route_query)
    graph.add_node("clarify", clarify)
    graph.add_node("analyze", analyze_query)
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("retrieve", retrieve_context(client, embedding_model))
    graph.add_node("grade", grade_evidence)
    graph.add_node("select", select_evidence)
    graph.add_node("answer", answer_with_context)
    graph.add_node("verify", verify_groundedness)
    graph.add_node("finalize", finalize_response)

    graph.set_entry_point("route_query")
    graph.add_conditional_edges(
        "route_query",
        decide_route,
        {"clarify": "clarify", "analyze": "analyze"},
    )
    graph.add_edge("clarify", "finalize")
    graph.add_edge("analyze", "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_edge("grade", "select")
    graph.add_edge("select", "answer")
    graph.add_edge("answer", "verify")
    graph.add_edge("verify", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_agentic_rag(query, client, embedding_model):
    started_at = time.time()
    app = build_agentic_rag_graph(client, embedding_model)
    result = app.invoke(
        {
            "query": query,
            "route": "",
            "query_analysis": {},
            "search_queries": [],
            "retrieved_ids": [],
            "distances": [],
            "docs": [],
            "evidence_quality": "",
            "selected_evidence": [],
            "answer": "",
            "groundedness": "",
            "trace": [],
            "events": [],
            "latency_seconds": 0.0,
        }
    )
    result["latency_seconds"] = round(time.time() - started_at, 2)
    return result
