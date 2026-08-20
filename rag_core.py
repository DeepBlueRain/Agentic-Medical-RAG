from llm_client import chat_completion


def build_grounded_prompt(query, context_docs, query_analysis=None, evidence_quality="", strict=False):
    context = "\n\n---\n\n".join(
        [
            f"[文档ID: {doc.get('id')}] 标题: {doc.get('title', '')}\n{doc['content']}"
            for doc in context_docs
        ]
    )
    intent = (query_analysis or {}).get("intent", "general")
    strict_instruction = ""
    if strict:
        strict_instruction = (
            "This is a verification retry. Remove any claim that is not directly supported "
            "by the context, and explicitly state when the evidence is insufficient.\n"
        )
    return f"""You are a careful medical knowledge assistant.
Answer the user's question using ONLY the context documents below.
If the context does not contain enough evidence, say that the current knowledge base does not contain sufficient information.
Do not invent medical facts.
{strict_instruction}
The detected question intent is: {intent}.
The current evidence quality is: {evidence_quality}.
When possible, mention the supporting document ID in the answer.

Context Documents:
{context}

User Question:
{query}

Answer:"""


def generate_answer(query, context_docs, query_analysis=None, evidence_quality="", strict=False):
    """Generate a grounded answer with the online LLM."""
    if not context_docs:
        return "当前知识库没有召回到足够相关的文档，无法基于已有资料回答该问题。"

    prompt = build_grounded_prompt(
        query,
        context_docs,
        query_analysis=query_analysis,
        evidence_quality=evidence_quality,
        strict=strict,
    )
    return chat_completion(
        [
            {
                "role": "system",
                "content": "You answer in Chinese. Be concise, factual, and grounded in the provided context.",
            },
            {"role": "user", "content": prompt},
        ]
    )
