"""Local text embeddings. The same model must encode documents and queries."""
from functools import lru_cache

DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


@lru_cache(maxsize=2)
def get_model(model_name: str = DEFAULT_MODEL):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def encode_texts(texts: list[str], model_name: str = DEFAULT_MODEL):
    model = get_model(model_name)
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=len(texts) > 10)
