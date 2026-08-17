from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=2)
def load_embedding_model(model_name):
    """Load the sentence transformer embedding model."""
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to load embedding model '{model_name}': {exc}") from exc
