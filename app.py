import os

import streamlit as st

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "./hf_cache")

from agentic_rag import run_agentic_rag
from config import (
    COLLECTION_NAME,
    DATA_FILE,
    EMBEDDING_MODEL_NAME,
    MAX_ARTICLES_TO_INDEX,
    MILVUS_LITE_DATA_PATH,
    ONLINE_LLM_MODEL,
    TOP_K,
    id_to_doc_map,
)
from data_utils import load_data
from milvus_utils import get_milvus_client, index_data_if_needed, setup_milvus_collection
from models import load_embedding_model


st.set_page_config(page_title="Agentic Medical RAG", layout="wide")
st.title("Agentic Medical RAG")
st.caption(f"Milvus Lite + {EMBEDDING_MODEL_NAME} + Online LLM: {ONLINE_LLM_MODEL}")

milvus_client = get_milvus_client()
if not milvus_client:
    st.error("Failed to initialize Milvus Lite client. Check logs and configuration.")
    st.stop()

collection_is_ready = setup_milvus_collection(milvus_client)
embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)

if not collection_is_ready or not embedding_model:
    st.error("Failed to load embedding model or set up the Milvus Lite collection.")
    st.stop()

pubmed_data = load_data(DATA_FILE)
if pubmed_data:
    indexing_successful = index_data_if_needed(milvus_client, pubmed_data, embedding_model)
else:
    st.warning(f"No data loaded from {DATA_FILE}. Please run preprocess.py first.")
    indexing_successful = False

st.divider()

if not indexing_successful and not id_to_doc_map:
    st.error("RAG is unavailable because the index or document map is not ready.")
    st.stop()

query = st.text_input("Enter a medical question", key="query_input")

if st.button("Run Agentic RAG", key="submit_button") and query:
    with st.spinner("Running Agentic RAG workflow..."):
        result = run_agentic_rag(query, milvus_client, embedding_model)

    st.subheader("Answer")
    st.write(result["answer"])

    col1, col2 = st.columns(2)
    col1.metric("Groundedness", result["groundedness"])
    col2.metric("Latency", f"{result['latency_seconds']:.2f}s")

    st.write(
        {
            "query_complexity": result["query_complexity"],
            "sub_questions": result["sub_questions"],
            "tool_plan": result["tool_plan"],
            "tool_calls": len(result["tool_calls"]),
            "generation_mode": result["generation_mode"],
            "retrieval_round": result["retrieval_round"],
            "retry_reason": result["retry_reason"],
            "revision_count": result["revision_count"],
        }
    )

    st.subheader("Agent Trace")
    st.code(" -> ".join(result["trace"]))

    st.subheader("Agent Tool Plan")
    st.json(result["tool_plan"])

    st.subheader("Tool Calls")
    for index, call in enumerate(result["tool_calls"]):
        with st.expander(f"{index + 1}. {call['tool']} - {call['status']}", expanded=True):
            st.write(call["detail"])
            if call["data"]:
                st.json(call["data"])

    st.subheader("Retrieved Context")
    st.write("Query analysis:", result["query_analysis"])
    st.write("Query complexity:", result["query_complexity"])
    st.write("Sub-questions:", result["sub_questions"])
    st.write("Search queries:", " | ".join(result["search_queries"]))
    st.write("Evidence quality:", result["evidence_quality"])
    st.subheader("Workflow Details")
    for index, event in enumerate(result["events"]):
        label = f"{index + 1}. {event['step']} · {event['status']}"
        with st.expander(label, expanded=True):
            st.write(event["detail"])
            if event["data"]:
                st.json(event["data"])

    st.subheader("Selected Evidence")
    if result["selected_evidence"]:
        st.json(result["selected_evidence"])
    else:
        st.info("No selected evidence.")

    if not result["docs"]:
        st.warning("No retrieved documents.")
    for index, doc in enumerate(result["docs"]):
        with st.expander(f"Document {index + 1} (ID: {doc['id']}, distance: {doc['distance']:.4f}) - {doc['title'][:60]}"):
            st.write(f"**Title:** {doc['title']}")
            st.write(f"**Content:** {doc['abstract']}")

st.sidebar.header("Configuration")
st.sidebar.markdown("**Workflow:** LangGraph Agentic RAG")
st.sidebar.markdown("**Vector store:** Milvus Lite")
st.sidebar.markdown(f"**Database path:** `{MILVUS_LITE_DATA_PATH}`")
st.sidebar.markdown(f"**Collection:** `{COLLECTION_NAME}`")
st.sidebar.markdown(f"**Data file:** `{DATA_FILE}`")
st.sidebar.markdown(f"**Embedding model:** `{EMBEDDING_MODEL_NAME}`")
st.sidebar.markdown(f"**Online LLM:** `{ONLINE_LLM_MODEL}`")
st.sidebar.markdown(f"**Max indexed documents:** `{MAX_ARTICLES_TO_INDEX}`")
st.sidebar.markdown(f"**Top K:** `{TOP_K}`")
