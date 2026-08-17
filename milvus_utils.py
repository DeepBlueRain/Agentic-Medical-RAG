import os
import time
from functools import lru_cache

from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from config import (
    COLLECTION_NAME,
    EMBEDDING_DIM,
    INDEX_METRIC_TYPE,
    INDEX_PARAMS,
    INDEX_TYPE,
    MAX_ARTICLES_TO_INDEX,
    MILVUS_LITE_DATA_PATH,
    SEARCH_PARAMS,
    TOP_K,
    id_to_doc_map,
)


@lru_cache(maxsize=1)
def get_milvus_client():
    """Initialize and return a Milvus Lite client."""
    try:
        db_dir = os.path.dirname(MILVUS_LITE_DATA_PATH)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        return MilvusClient(uri=MILVUS_LITE_DATA_PATH)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Milvus Lite client: {exc}") from exc


def setup_milvus_collection(client):
    """Create the collection and vector index when they do not exist."""
    try:
        has_collection = COLLECTION_NAME in client.list_collections()

        if not has_collection:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                FieldSchema(name="content_preview", dtype=DataType.VARCHAR, max_length=500),
            ]
            schema = CollectionSchema(fields, f"Medical RAG collection, dim={EMBEDDING_DIM}")

            client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type=INDEX_TYPE,
                metric_type=INDEX_METRIC_TYPE,
                params=INDEX_PARAMS,
            )
            client.create_index(COLLECTION_NAME, index_params)

        try:
            client.load_collection(COLLECTION_NAME)
        except Exception:
            pass

        get_collection_count(client)
        return True

    except Exception as exc:
        raise RuntimeError(f"Error setting up Milvus collection '{COLLECTION_NAME}': {exc}") from exc


def get_collection_count(client):
    """Return collection entity count with compatibility fallbacks."""
    try:
        if hasattr(client, "num_entities"):
            return client.num_entities(COLLECTION_NAME)
        stats = client.get_collection_stats(COLLECTION_NAME)
        return int(stats.get("row_count", stats.get("rowCount", 0)))
    except Exception:
        return 0


def index_data_if_needed(client, data, embedding_model):
    """Index processed documents into Milvus Lite when the collection is empty."""
    current_count = get_collection_count(client)
    data_to_index = data[:MAX_ARTICLES_TO_INDEX]

    docs_for_embedding = []
    data_to_insert = []
    temp_id_map = {}

    for index, doc in enumerate(data_to_index):
        title = doc.get("title", "") or ""
        abstract = doc.get("abstract", "") or ""
        content = f"Title: {title}\nContent: {abstract}".strip()
        if not content:
            continue

        doc_id = index
        temp_id_map[doc_id] = {"title": title, "abstract": abstract, "content": content}
        docs_for_embedding.append(content)
        data_to_insert.append(
            {
                "id": doc_id,
                "embedding": None,
                "content_preview": content[:500],
            }
        )

    needed_count = len(docs_for_embedding)
    if current_count >= needed_count and needed_count > 0:
        if not id_to_doc_map:
            id_to_doc_map.update(temp_id_map)
        return True

    if not docs_for_embedding:
        raise RuntimeError("No valid document text found for indexing")

    start_embed = time.time()
    embeddings = embedding_model.encode(docs_for_embedding, show_progress_bar=True)
    embedding_seconds = time.time() - start_embed

    for index, embedding in enumerate(embeddings):
        data_to_insert[index]["embedding"] = embedding.tolist() if hasattr(embedding, "tolist") else embedding

    try:
        start_insert = time.time()
        client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
        id_to_doc_map.update(temp_id_map)
        insert_seconds = time.time() - start_insert
        print(
            f"Indexed {len(data_to_insert)} documents "
            f"(embedding: {embedding_seconds:.2f}s, insert: {insert_seconds:.2f}s)."
        )
        return True
    except Exception as exc:
        raise RuntimeError(f"Error inserting data into Milvus Lite: {exc}") from exc


def search_similar_documents(client, query, embedding_model):
    """Search similar documents from Milvus Lite."""
    try:
        try:
            client.load_collection(COLLECTION_NAME)
        except Exception:
            pass

        query_embedding = embedding_model.encode([query])[0]
        query_vector = query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding

        search_params = {
            "collection_name": COLLECTION_NAME,
            "data": [query_vector],
            "anns_field": "embedding",
            "limit": TOP_K,
            "output_fields": ["id"],
        }

        try:
            results = client.search(**search_params, search_params=SEARCH_PARAMS)
        except TypeError:
            results = client.search(**search_params)

        if not results or not results[0]:
            return [], []

        hit_ids = [hit["id"] for hit in results[0]]
        distances = [hit["distance"] for hit in results[0]]
        return hit_ids, distances
    except Exception as exc:
        raise RuntimeError(f"Error during Milvus Lite search: {exc}") from exc
