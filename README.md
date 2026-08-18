# Agentic Medical RAG

An agentic RAG assistant for medical science articles. The project combines local document retrieval with an online OpenAI-compatible LLM and exposes the workflow through Streamlit and FastAPI.

## What Changed

The original RAG pipeline has been upgraded from a single retrieval-and-generation flow to an Agentic RAG workflow:

```text
user query
  -> route query
  -> retrieve documents from Milvus Lite
  -> generate grounded answer with online LLM
  -> verify answer groundedness
  -> return answer, retrieved context, trace and latency
```

The workflow is implemented with LangGraph and exposes the following trace nodes:

- `route_query`
- `retrieve_context`
- `answer_with_context`
- `verify_groundedness`

## Features

- Parse and chunk medical HTML documents with BeautifulSoup.
- Generate embeddings with Sentence-Transformers.
- Store and retrieve vectors through Milvus Lite with IVF_FLAT indexing.
- Use a LangGraph workflow to route, retrieve, answer and verify each query.
- Call an online OpenAI-compatible LLM instead of loading a local generation model.
- Show step-by-step workflow details, retrieved documents, agent trace, groundedness label and latency in Streamlit.
- Provide an API endpoint for integration and evaluation.

## Tech Stack

- Python
- LangGraph
- FastAPI
- Streamlit
- Milvus Lite
- Sentence-Transformers
- OpenAI-compatible online LLM API
- BeautifulSoup / lxml

## Project Structure

```text
.
├── agentic_rag.py     # LangGraph workflow and trace collection
├── api.py             # FastAPI service
├── app.py             # Streamlit demo UI
├── config.py          # Data, Milvus and online LLM configuration
├── llm_client.py      # OpenAI-compatible chat completion client
├── milvus_utils.py    # Milvus Lite indexing and retrieval
├── models.py          # Embedding model loader
├── preprocess.py      # HTML parsing and chunking
├── rag_core.py        # Grounded answer prompt and generation
├── requirements.txt
└── .env.example
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local `.env` file from `.env.example`:

```dotenv
ONLINE_LLM_BASE_URL=https://apihub.agnes-ai.com/v1
ONLINE_LLM_API_KEY=your-api-key
ONLINE_LLM_MODEL=agnes-2.5-flash
```

Do not commit `.env`. It is ignored by `.gitignore`.

## Data Preparation

Put raw HTML files under `data/`, then run:

```bash
python preprocess.py
```

This generates `data/processed_data.json` for vector indexing. Large raw data, vector database files, caches and logs are intentionally excluded from the repository.

## Run Streamlit

```bash
streamlit run app.py
```

The page displays the answer, retrieved context, workflow trace, groundedness label and latency.

## Run API

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Ask a question:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"白血病患者化疗后需要注意什么？"}'
```

Example response fields:

```text
answer
groundedness
trace
workflow_events
latency_seconds
retrieved_docs
```

## Implementation Highlights

- Built a LangGraph Agentic RAG workflow with query routing, vector retrieval, grounded answer generation and answer verification.
- Encapsulated Milvus Lite retrieval as an agent workflow capability and exposed trace, context and latency for debugging.
- Replaced local generation-model loading with an OpenAI-compatible online LLM service, keeping API credentials outside the repository.
- Designed the system for later extensions such as query rewriting, conversation memory, reranking and evaluation datasets.
