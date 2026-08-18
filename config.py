import os

from dotenv import load_dotenv

load_dotenv()


# Milvus Lite
MILVUS_LITE_DATA_PATH = os.getenv("MILVUS_LITE_DATA_PATH", "./data/medical_rag.db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "medical_rag_lite")

# Data
DATA_FILE = os.getenv("DATA_FILE", "./data/processed_data.json")

# Embedding model
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# Online LLM, compatible with OpenAI-style chat completions.
ONLINE_LLM_BASE_URL = os.getenv("ONLINE_LLM_BASE_URL", "https://apihub.agnes-ai.com/v1")
ONLINE_LLM_API_KEY = os.getenv("ONLINE_LLM_API_KEY", "")
ONLINE_LLM_MODEL = os.getenv("ONLINE_LLM_MODEL", "agnes-2.5-flash")
ONLINE_LLM_TIMEOUT = int(os.getenv("ONLINE_LLM_TIMEOUT", "60"))

# Indexing and search
MAX_ARTICLES_TO_INDEX = int(os.getenv("MAX_ARTICLES_TO_INDEX", "500"))
TOP_K = int(os.getenv("TOP_K", "3"))
INDEX_METRIC_TYPE = os.getenv("INDEX_METRIC_TYPE", "L2")
INDEX_TYPE = os.getenv("INDEX_TYPE", "IVF_FLAT")
INDEX_PARAMS = {"nlist": int(os.getenv("MILVUS_NLIST", "128"))}
SEARCH_PARAMS = {"nprobe": int(os.getenv("MILVUS_NPROBE", "16"))}

# Generation
MAX_NEW_TOKENS_GEN = int(os.getenv("MAX_NEW_TOKENS_GEN", "256"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.1"))

# Runtime document map, populated during indexing.
id_to_doc_map = {}
