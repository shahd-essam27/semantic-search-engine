"""Build/update a persistent local Chroma collection."""
from __future__ import annotations
import hashlib
from pathlib import Path
from src.data_loader import DEFAULT_DATASET, ROOT, load_documents, prepare_chunks
from src.embeddings import DEFAULT_MODEL, encode_texts

DEFAULT_DB = ROOT / 'chroma_db'
COLLECTION = 'week5_semantic_search_v1'


def open_collection(db_dir: Path = DEFAULT_DB, collection_name: str = COLLECTION):
    import chromadb
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=None,
        configuration={'hnsw': {'space': 'cosine'}},
    )
    return collection


def model_collection_name(model_name: str, chunk_words: int, overlap_words: int) -> str:
    key = f'{model_name}|{chunk_words}|{overlap_words}'
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return f'search_{digest}'


def index_documents(dataset: Path = DEFAULT_DATASET, db_dir: Path = DEFAULT_DB,
                    model_name: str = DEFAULT_MODEL, chunk_words: int = 120,
                    overlap_words: int = 20, batch_size: int = 64) -> dict:
    if batch_size < 1:
        raise ValueError('batch_size must be >= 1')
    records = load_documents(dataset)
    chunks = prepare_chunks(records, chunk_words, overlap_words)
    name = model_collection_name(model_name, chunk_words, overlap_words)
    collection = open_collection(db_dir, name)
    # Guard against stale index if CSV text or metadata changed.
    # Upsert updates existing IDs; stale IDs are removed.
    desired_ids = {item['id'] for item in chunks}
    existing_ids = set(collection.get(include=[])['ids'])
    stale = sorted(existing_ids - desired_ids)
    for start in range(0, len(stale), batch_size):
        collection.delete(ids=stale[start:start + batch_size])
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        vectors = encode_texts([item['text'] for item in batch], model_name)
        collection.upsert(
            ids=[item['id'] for item in batch],
            documents=[item['text'] for item in batch],
            embeddings=vectors.tolist(),
            metadatas=[{**item['metadata'], 'model_name': model_name,
                        'chunk_words': chunk_words, 'overlap_words': overlap_words}
                       for item in batch],
        )
    return {'documents': len(records), 'chunks': len(chunks), 'collection': name,
            'stored': collection.count(), 'db_dir': str(db_dir), 'model': model_name}
