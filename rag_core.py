from llm_client import chat_completion


def build_grounded_prompt(query, context_docs):
    context = "\n\n---\n\n".join([doc["content"] for doc in context_docs])
    return f"""You are a careful medical knowledge assistant.
Answer the user's question using ONLY the context documents below.
If the context does not contain enough evidence, say that the current knowledge base does not contain sufficient information.
Do not invent medical facts.

Context Documents:
{context}

User Question:
{query}

Answer:"""


def generate_answer(query, context_docs):
    """Generate a grounded answer with the online LLM."""
    if not context_docs:
        return "当前知识库没有召回到足够相关的文档，无法基于已有资料回答该问题。"

    prompt = build_grounded_prompt(query, context_docs)
    return chat_completion(
        [
            {
                "role": "system",
                "content": "You answer in Chinese. You must be factual, concise, and grounded in the provided context.",
            },
            {"role": "user", "content": prompt},
        ]
    )
